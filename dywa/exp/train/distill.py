#!/usr/bin/env python3

from typing import Tuple, Dict, Optional, List, Union

from dataclasses import dataclass, fields, replace
from functools import partial

import torch as th
import torch.nn as nn

import pdb

from util.torch_util import dcn
from models.rl.net.base import FeatureBase
from models.common import (
    attention,
    MultiHeadLinear,
    grad_step,
    MLP,
    map_tensor,
    SingleGRU, SingleLSTM, DeepGRU
)
from pathlib import Path
from train.ckpt import save_ckpt, load_ckpt, last_ckpt
from util.path import ensure_directory
from models.cloud.point_mae import (
    PointMAEEncoder,
)
from util.config import recursive_replace_map
from env.env.wrap.normalize_env import NormalizeEnv
from icecream import ic
from train.losses import GaussianKLDivLoss
from train.metrics import pose_error
from models.modules import (PredictorHead, HistoryEncoder, PosePredictor,
                                PointEncoder, TokenEncoder, TokenDecoder, Aggregator, CLLoss)
from models.ttt import (DynamicsModel, OutcomeEncoder, FlowDecoder,
                        MacroActionEncoder, ttt_inner_update, TransformerPointEncoder,
                        FiLMPolicyTrunk)

from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext


@dataclass
class TTTParams:
    """Test-Time-Training belief/dynamics hyperparameters (see models/ttt.py).

    Only consumed by `StudentAgentTTT`; ignored by the vanilla `StudentAgentRMA`.
    Kept as a field on the shared student config so `++student.ttt.*` binds
    uniformly across all entry scripts.
    """
    latent_dim: int = 64      # belief dim D; == decoder FiLM cond_dim
    ttt_alpha: float = 0.5    # inner-loop step size
    k_inner_steps: int = 1    # number of inner TTT steps
    history_len: int = 10     # max macro-transitions kept for adaptation
    detach_q: bool = False    # False=second-order (offline), True=first-order (online)
    d_model: int = 128
    nhead: int = 4
    dyn_layers: int = 3
    outcome_layers: int = 2
    use_checkpoint: bool = False  # see models/ttt.py: avoids double-backward alloc assert
    # Policy point encoder: self-attention transformer (SelfAttnBlock, no FiLM)
    # instead of PointNet2. q is injected at the policy trunk, not here.
    transformer_point_encoder: bool = True
    point_enc_layers: int = 3
    # Policy trunk = the stage that turns the token sequence + belief `q` into the
    # action distribution.
    #   'dywa'            : TokenDecoder(film) -> Aggregator, plus a direct `q_head`.
    #   'film_transformer': doors-style FiLMPolicyTrunk (near-identity FiLM blocks +
    #                       concat-q head + zero-init q-residual). See plan REV 3 —
    #                       DyWA's FilmBlock attenuates dL/dq to ~1e-5, which freezes
    #                       q0/dyn/outcome at init.
    policy_trunk: str = 'dywa'
    policy_layers: int = 4          # FiLM blocks when policy_trunk=='film_transformer'
    # ── doors-inspired refactor (see plan REVISION 2) ────────────────────────
    window_k: int = 5        # K-step outcome window (macro-transition stride)
    # macro-action over the K-window: 'ee_only' (net EE motion, 6-dim = Δpos ‖ rotvec),
    # 'ee_mean_gains' (+ 14 window-mean gains, 20-dim), or 'seq' (MacroActionEncoder
    # over the K×20 raw actions)
    macro_action: str = 'ee_only'
    macro_embed_dim: int = 32       # macro-action width when macro_action=='seq'
    recon_weight: float = 1.0       # FlowDecoder reconstruction loss weight (anti-collapse)
    recon_on_history: bool = True   # also apply recon on the H support transitions

class StudentAgentRMA(nn.Module):
    '''
    model list:

    Transformer: 
        self.tokenizer: Linear layer for abs_goal, hand_state, robot_state, previous_action
        self.pos_embed: learnable positional embedding
        Vision:
            self.group: group point 
            self.patch_encoder: vision encoder, currently is pointnet
        self.encoder: self attention encoder, input various dimensions of information

    MLP_layers: 
        self.aggregator: Aggregate the current and historical memory information
        self.project: The last linear layer

    Predict
        self.pre_pose
    '''
    @dataclass
    class StudentAgentRMAConfig(FeatureBase.Config):

        horizon: int = 1
        p_drop: float = 0.0
        num_layer: int = 2                    ### useless
        pos_embed_type: Optional[str] = 'mlp'
        patch_type: str = 'mlp'               ### useless, mlp/knn/cnn
        batch_size: int = 4096                ### useless, 自动设为env数
        action_size: int = 20
        # reset delay for student
        # student is reset after t ~ U(0, T) step
        max_delay_steps: int = 7              ### useless
        estimate_level: str = 'state'         ### or action but action is not implemented yet
        without_teacher: bool = True
        
        use_gpcd: bool = True               ### Always be True
        use_interim_goal: bool = True
        ckpt: Optional[str] = None
        state_keys: Optional[List[str]] = None


        #### training parameters        
        learning_rate: float = 3e-4
        loss_type: str = "KLDiv" ## or MSE
        use_triplet_loss: bool = False ## useless
        use_amp: bool = False

        #########################
        ####      Model      ####
        #########################


        norm:str ='bn'  ### used in final mlp and patch encoder
        embed_size: int = 128 ### self attention embed size 

        #### state tokenizer
        shapes: Optional[Dict[str, int]] = None
        state_tokenizer_activate:bool=False
        state_tokenizer_hiddens:Optional[List[int]]=None

        #### Vision Encoder
        point_tokenizer: PointEncoder.Config = PointEncoder.Config()

        #### Encoder
        encoder: TokenEncoder.Config = TokenEncoder.Config()

        ###  Decoder
        decoder: TokenDecoder.Config = TokenDecoder.Config()

        ### Pose Predictor
        vision_pose_predictor: PosePredictor.PosePredictorConfig = PosePredictor.PosePredictorConfig(
            input='vision'
        )
        merge_pose_pred: bool = False

        ### final mlp
        aggregator: Aggregator.Config = Aggregator.Config()

        ## History
        use_history: bool = False
        history_tokenizer: HistoryEncoder.Config = HistoryEncoder.Config()
        constraint: CLLoss.Config = CLLoss.Config()

        ## Test-Time-Training (only used by StudentAgentTTT; ignored otherwise)
        ttt: TTTParams = TTTParams()


        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            p_drop = self.p_drop
            self.encoder.self_atten = recursive_replace_map(self.encoder.self_atten, {
                'layer.hidden_size': self.embed_size,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop
            })

    def __init__(self, cfg: StudentAgentRMAConfig,
                 writer, device):
        super().__init__()

        self.cfg = cfg
        self.writer = writer
        self.device = device
        self.input_keys = cfg.state_keys
        self.action_size = cfg.action_size

        ### amp
        self.scaler = GradScaler() if cfg.use_amp else None

        ## vision encoder
        self.point_tokenizer = PointEncoder(cfg.point_tokenizer, cfg.embed_size, cfg.norm)
        num_pcd_tokens = cfg.point_tokenizer.num_tokens
        num_vision_tokens = num_pcd_tokens * 2 if cfg.use_gpcd else num_pcd_tokens

        ## state tokenizer
        state_hiddens = () if cfg.state_tokenizer_hiddens is None else tuple(cfg.state_tokenizer_hiddens)
        self.tokenizer = nn.ModuleDict({
            k: MLP((cfg.shapes[k], ) + state_hiddens + (cfg.embed_size, ), 
                    use_ln=True, use_bn=False, activate_output=cfg.state_tokenizer_activate)
            for k in self.input_keys
        })

        ## encoder
        self.encoder = TokenEncoder(cfg.encoder, cfg.embed_size, num_tokens= num_vision_tokens, norm = cfg.norm)

        ## history encoder
        num_history_tokens = 0
        if cfg.use_history:
            history_tokenizer = replace(cfg.history_tokenizer, num_envs=cfg.batch_size)
            self.history_tokenizer = HistoryEncoder(history_tokenizer, embed_size=cfg.embed_size, 
                                                    num_tokens=num_pcd_tokens + len(self.input_keys), norm=cfg.norm)

            ### contrastive loss
            num_history_tokens = self.history_tokenizer.decoder.num_query_tokens
            self.constraint = CLLoss(cfg.constraint, anchor_dim= num_history_tokens * cfg.embed_size, 
                                        negative_dim= None, positive_dim= cfg.embed_size)
            
        ## decoder
        num_decoder_tokens = num_vision_tokens + len(self.input_keys)
        self.decoder = TokenDecoder(cfg.decoder, cfg.embed_size, num_tokens= num_decoder_tokens, norm= cfg.norm,
                                    cond_dim= num_history_tokens * cfg.embed_size if cfg.use_history else 0)

        ### final network
        num_agg_tokens = self.decoder.num_query_tokens
        self.aggregator = Aggregator(cfg.aggregator, num_agg_tokens, cfg.embed_size, cfg.action_size, cfg.norm, batch_size=cfg.batch_size)
       
        ## pose predictor
        self.vision_pose_predictor = PosePredictor(
            self.cfg.vision_pose_predictor,
            num_vision_tokens * cfg.embed_size,
            cfg.norm
        )

        ### loss
        if cfg.estimate_level == 'action':
            if self.cfg.loss_type == "KLDiv":
                self.loss = GaussianKLDivLoss()
            elif self.cfg.loss_type == "MSE":
                self.loss = nn.MSELoss()
            else:
                raise NotImplementedError
            self.action_pose_error = pose_error
        else:
            if cfg.use_triplet_loss:
                self.loss = nn.TripletMarginLoss(margin=0.0)
            else:
                ### need env.determistic action = True
                self.loss = nn.MSELoss()


        self.losses = 0 ###总体loss
        ## Useless
        self.pose_losses = {'state': 0,
                            "embed": 0,
                            "token": 0,
                            "vision": 0} 
        self.aux_losses = {'phys_params': 0}

        self.optimizer = th.optim.Adam(
            # OK?
            self.parameters(),
            self.cfg.learning_rate)
        
        self.need_goal = None

        self.vision_tokens = None

    def _forward_impl(self, obs):
        cfg = self.cfg

        aux = dict()
        ## input tokens
        ctx_tokens = [
            self.tokenizer[k](obs[k].detach().clone())[:, None]
            for k in self.input_keys if k in obs] 
        self.vision_tokens = pcd_tokens = self.point_tokenizer(obs['partial_cloud'])

        ## goal tokens
        if self.cfg.use_gpcd:
            gpcd_tokens = self.point_tokenizer(obs['goal_cloud'])
            self.vision_tokens = th.concat([pcd_tokens, gpcd_tokens], dim=-2) 

        ## encoder
        embed_tokens = self.encoder(self.vision_tokens)

        ## pose prediction
        self.pose_losses['vision'], pose_pred = self.vision_pose_predictor(self.vision_tokens, obs)
        if self.cfg.merge_pose_pred:
            pose_pred_tokens = self.tokenizer['pose_pred'](pose_pred)[:, None]
            embed_tokens = th.cat([pose_pred_tokens, embed_tokens], dim=-2)

        ### history condition
        cond = self.history_tokenizer(th.concat([pcd_tokens] + ctx_tokens, dim=-2)) if cfg.use_history else None
        aux['cond'] = cond

        ## decoder
        decoded_tokens = self.decoder(th.cat(ctx_tokens + [embed_tokens], dim=-2), cond)
        decoded_tokens = decoded_tokens.reshape(*decoded_tokens.shape[:-2], -1) ### B * 512
       
        if not cfg.use_interim_goal:
            self.need_goal.fill_(0)

        output = self.aggregator(decoded_tokens) 
        output = output.reshape(*output.shape[:-1], 2, -1)

        return output, aux

    def reset(self, obs):
        cfg = self.cfg
        device = obs['partial_cloud'].device

        if not cfg.without_teacher:
            teacher_state = obs.get('teacher_state', None)
            if teacher_state is not None:
                teacher_state = teacher_state.detach().clone()

            teacher_action = obs.get('teacher_action', None)
            if teacher_action is not None:
                teacher_action = teacher_action.detach().clone()

        if cfg.max_delay_steps > 0:
            self.delay_count = -th.randint(
                high=cfg.max_delay_steps,
                size=(cfg.batch_size,),
                device=device
            )

        if self.aggregator is not None:   # None when policy_trunk=='film_transformer'
            self.aggregator.reset()

        with autocast() if cfg.use_amp else nullcontext():
            output, _ = self._forward_impl(obs)

            if not cfg.without_teacher and cfg.max_delay_steps > 0:
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action
                )

        if not cfg.use_interim_goal:
            self.need_goal = th.zeros(cfg.batch_size,
                                      dtype=bool,
                                      device=obs['partial_cloud'].device)

       
        return output.detach().clone()


    def reset_state(self, done: th.Tensor):
        # reset memory
        cfg = self.cfg
        keep = (~done)[..., None]

        if self.aggregator is not None:   # None when policy_trunk=='film_transformer'
            self.aggregator.reset(keep)

        # reset counts
        if cfg.max_delay_steps > 0:
            num_reset: int = done.sum()
            self.delay_count[done] = -th.randint(
                high=cfg.max_delay_steps,
                size=(num_reset,),
                device=self.delay_count.device
            )
        if not cfg.use_interim_goal:
            self.need_goal |= done
        
        if cfg.use_history:
            self.history_tokenizer.reset_history(done)

    def get_output(self, obs):
        """
        Generate action and predicted state without backpropagation
        
        Args:
            obs: Environment observations
            
        Returns:
            tuple: (output, aux) where output contains action/state predictions
                  and aux contains auxiliary information like pose predictions
        """
        cfg = self.cfg
        
        with autocast() if cfg.use_amp else nullcontext():
            output, aux = self._forward_impl(obs)

        # Handle delay logic (maintain original logic)
        if not cfg.without_teacher and cfg.max_delay_steps > 0:
            if cfg.estimate_level == 'state':
                teacher_state = obs.get('teacher_state', None)
                if teacher_state is not None:
                    teacher_state = teacher_state.detach().clone()
                output = th.where(
                    self.delay_count[..., None] >= 0,
                    output, teacher_state
                )
            elif cfg.estimate_level == 'action':
                teacher_action = obs.get('teacher_action', None)
                if teacher_action is not None:
                    teacher_action = teacher_action.detach().clone()
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action
                )
        
        return output, aux
    
    def update_policy(self, obs, step, output, aux):
        """
        Execute backpropagation and parameter updates
        
        Args:
            obs: Environment observations
            step: Current training step
            done: Environment done flags
        """
        cfg = self.cfg
        if not cfg.without_teacher:
            teacher_state = obs.get('teacher_state', None)
            if teacher_state is not None:
                teacher_state = teacher_state.detach().clone()

            teacher_action = obs.get('teacher_action', None)
            if teacher_action is not None:
                teacher_action = teacher_action.detach().clone()

            if 'neg_teacher_state' in obs:
                neg_teacher_state = obs.get(
                    'neg_teacher_state').detach().clone()
                
        # compute pose losses
        self.pose_losses['vision'], _ = self.vision_pose_predictor(self.vision_tokens, obs)

        # Update only for the case where current timestep excced the delay
        if cfg.max_delay_steps > 0:
            step_indices = (self.delay_count >= 0).nonzero().flatten()
        else:
            step_indices = Ellipsis

        assert (cfg.max_delay_steps <= 0)

        if cfg.use_history:
            cond = aux['cond']
            self.aux_losses['cond_pos'] = self.constraint(cond, positive = teacher_state.detach())

        if cfg.estimate_level == 'state':
            if cfg.use_triplet_loss:
                self.losses = self.losses + self.loss(
                    output[step_indices],
                    teacher_state[step_indices],
                    neg_teacher_state[step_indices])
            else: 
                self.losses = self.losses + self.loss(
                    output[step_indices], teacher_state[step_indices])
        else: ### current : estimate action
            if self.cfg.loss_type == "KLDiv":
                self.losses = self.losses + self.loss(
                    # mu
                    output[step_indices][..., 0, :],
                    # ls
                    output[step_indices][..., 1, :],
                    # mu
                    teacher_action[step_indices][..., 0, :],
                    # ls
                    teacher_action[step_indices][..., 1, :],
                )
            else:
                self.losses = self.losses + self.loss(
                    # mu
                    output[step_indices][..., 0, :],
                    # mu
                    teacher_action[step_indices][..., 0, :],
                )
            pos_err, rot_err = self.action_pose_error(output[step_indices][..., 0, :6], teacher_action[step_indices][..., 0, :6]) ### teacher 和 student的pose error

        if (step + 1) % self.cfg.horizon == 0: ###当前horizon=1, 进入这个branch
            
            pose_loss = sum([v for k, v in self.pose_losses.items()]) 
            aux_loss = sum([v for k, v in self.aux_losses.items()])
            loss = self.losses + pose_loss + aux_loss
            
            if self.training:
                grad_step(loss, self.optimizer, scaler=self.scaler)
            else:
                self.optimizer.zero_grad()
                
            if self.writer is not None:
                with th.no_grad():
                    self.writer.add_scalar('loss/action',
                                           self.losses / cfg.horizon,
                                           global_step=step)
                    self.writer.add_scalar('loss/pose',
                                           pose_loss / cfg.horizon,
                                           global_step=step)
                    for k, v in self.pose_losses.items():
                        self.writer.add_scalar('loss/pose'+k,
                                           v / cfg.horizon,
                                           global_step=step)
                    for k, v in self.aux_losses.items():
                        self.writer.add_scalar('loss/aux_'+k,
                                           v / cfg.horizon,
                                           global_step=step)
        
                    self.writer.add_scalar('log/learning_rate',
                                           self.optimizer.param_groups[0]['lr'],
                                           global_step=step)
                    try:
                        pos_err, rot_err = dcn(pos_err).mean(), dcn(rot_err).mean()
                        self.writer.add_scalar('error/pos',
                                           pos_err,
                                           global_step=step)
                        self.writer.add_scalar('error/rot',
                                            rot_err,
                                            global_step=step)
                    except:
                        pass
                
            # if self.training: 
            self.losses = 0.
            for k in self.pose_losses:
                self.pose_losses[k] = 0.
            for k in self.aux_losses:
                self.aux_losses[k] = 0.

            if self.aggregator is not None:   # None when policy_trunk=='film_transformer'
                self.aggregator.memory_detach_()

    def forward(self, obs, step, done, aux=None):
        """
        Maintain backward compatibility, internally calls decoupled functions
        
        Args:
            obs: Environment observations
            step: Current training step
            done: Environment done flags
            aux: Auxiliary data (for compatibility)
            
        Returns:
            output: Action or state predictions
        """
        cfg = self.cfg
        
        # 1. Generate output (action/state)
        output, aux = self.get_output(obs)
        
        # 2. Execute backpropagation
        self.update_policy(obs, step, output, aux)

        ### useless
        if cfg.max_delay_steps > 0: 
            self.delay_count += 1
        # if aux is not None: 
        #     aux['pose'] = pose.clone()

        return output.detach().clone()

    def save(self, path: str):
        ensure_directory(Path(path).parent)
        save_ckpt(dict(self=self),
                  ckpt_file=path)

    def load(self, path: str, strict: bool = True):
        ckpt_path = last_ckpt(path)
        load_ckpt(dict(self=self),
                  ckpt_file=ckpt_path,
                  strict=strict,
                  exclude_keys=['history_tokenizer.history', 'aggregator.aggregator.memory'])

    def reset_optimizer(self):
        self.optimizer = th.optim.Adam(
            # OK?
            filter(lambda p: p.requires_grad, self.parameters()),
            self.cfg.learning_rate)


class StudentAgentTTT(StudentAgentRMA):
    """DyWA student with a Test-Time-Training latent belief `q`.

    Reuses the entire `StudentAgentRMA` trunk (point encoder, token encoder,
    FiLM decoder, aggregator, `GaussianKLDivLoss`) but:
      * drops the `HistoryEncoder`/`CLLoss` (`use_history=False`);
      * rebuilds the decoder so its FiLM `cond_dim == cfg.ttt.latent_dim`, and
        feeds it a per-env belief `q` instead of the history `cond`;
      * adds `q0` (meta-learned prior), `dyn` (`DynamicsModel`) and
        `outcome_enc` (`OutcomeEncoder`), used by the TTT inner loop.

    Only the imitation loss is kept; `dyn`/`outcome_enc`/`q0` are trained purely
    through the outer imitation loss via the inner-loop graph (as in the bars
    pipeline) — there is no separate dynamics loss term.
    """

    def __init__(self, cfg: 'StudentAgentRMA.StudentAgentRMAConfig', writer, device):
        # Force TTT-compatible settings on the (possibly shared) student config.
        cfg.use_history = False          # no HistoryEncoder / CLLoss
        cfg.estimate_level = 'action'    # so self.loss = GaussianKLDivLoss()
        cfg.loss_type = 'KLDiv'
        cfg.without_teacher = False
        # LayerNorm is mandatory: meta-training / per-env forwards run at batch
        # sizes (incl. B=1) that BatchNorm1d cannot handle. Fail loudly now.
        assert cfg.norm == 'ln', (
            f"StudentAgentTTT requires cfg.norm=='ln' (got {cfg.norm!r}); "
            "pass ++student.norm=ln")

        # The parent builds a throwaway FiLM decoder at cond_dim=0; force
        # film_mlp=False during that build so MLP_layer(0,0) is never created.
        orig_film_mlp = cfg.decoder.film_mlp
        cfg.decoder.film_mlp = False
        super().__init__(cfg, writer, device)
        cfg.decoder.film_mlp = orig_film_mlp

        tp = cfg.ttt

        if tp.policy_trunk not in ('dywa', 'film_transformer'):
            raise ValueError(f"unknown policy_trunk={tp.policy_trunk!r} "
                             "(expected 'dywa' or 'film_transformer')")

        num_pcd_tokens = cfg.point_tokenizer.num_tokens
        num_vision_tokens = num_pcd_tokens * 2 if cfg.use_gpcd else num_pcd_tokens
        num_decoder_tokens = num_vision_tokens + len(self.input_keys)

        if tp.policy_trunk == 'dywa':
            # Rebuild the decoder so the FiLM conditioning width == belief dim D.
            decoder_cfg = replace(cfg.decoder, decoder_type='film', film_mlp=True)
            self.decoder = TokenDecoder(
                decoder_cfg, cfg.embed_size, num_tokens=num_decoder_tokens,
                norm=cfg.norm, cond_dim=tp.latent_dim).to(device)
            self.film_trunk = None
        else:
            # doors-style trunk consumes the token sequence directly and emits the
            # action distribution, so TokenDecoder + Aggregator are both replaced.
            # Drop them so they contribute no dead parameters to the optimizer or
            # the checkpoint. (Only the parent's RMA-specific step/reset paths touch
            # them, and the TTT entry points never call those.)
            self.decoder = None
            self.aggregator = None
            self.film_trunk = FiLMPolicyTrunk(
                embed_size=cfg.embed_size,
                action_size=cfg.action_size,
                latent_dim=tp.latent_dim,
                num_layers=tp.policy_layers,
                nhead=tp.nhead).to(device)

        # Belief prior + dynamics / outcome / flow-decoder nets (double-backward safe).
        self.q0 = nn.Parameter(th.randn(tp.latent_dim, device=device) * 0.1)

        # Macro-action width over the K-window (see `build_macro`).
        # Net EE motion is the minimal relative pose: Δpos(3) ‖ rotvec(R_rel)(3).
        # (NOT the raw 9-dim hand_state difference — a rot6d difference is not a
        #  relative rotation; see models.ttt.ee_pose_delta.)
        self.ee_delta_dim = 6
        self.n_gains = cfg.action_size - 6      # 20-6 = 14 (KP7+KD7); action pose = first 6
        if tp.macro_action == 'ee_only':
            self.macro_dim = self.ee_delta_dim
        elif tp.macro_action == 'ee_mean_gains':
            self.macro_dim = self.ee_delta_dim + self.n_gains
        elif tp.macro_action == 'seq':
            self.macro_dim = tp.macro_embed_dim
        else:
            raise ValueError(f"unknown macro_action={tp.macro_action!r}")

        self.dyn = DynamicsModel(
            latent_dim=tp.latent_dim, action_dim=self.macro_dim,
            d_model=tp.d_model, num_layers=tp.dyn_layers, nhead=tp.nhead,
            use_checkpoint=tp.use_checkpoint).to(device)
        self.outcome_enc = OutcomeEncoder(
            d_model=tp.d_model, nhead=tp.nhead, num_layers=tp.outcome_layers,
            latent_dim=tp.latent_dim, use_checkpoint=tp.use_checkpoint).to(device)
        # Anti-collapse: reconstruct per-point flow from (pcd+flow, z). Outer-loop only.
        self.flow_decoder = FlowDecoder(
            latent_dim=tp.latent_dim, hidden=tp.d_model).to(device)
        # Only `seq` needs a learned temporal encoder over the K-action window.
        self.macro_encoder = None
        if tp.macro_action == 'seq':
            self.macro_encoder = MacroActionEncoder(
                action_raw_dim=cfg.action_size, out_dim=self.macro_dim,
                k_window=tp.window_k, d_model=tp.d_model, nhead=tp.nhead,
                use_checkpoint=tp.use_checkpoint).to(device)
        
        # Swap the policy's PointNet2 point encoder for a self-attention
        # transformer (no FiLM). Keeps the same token count so the downstream
        # TokenEncoder / TokenDecoder sizing (built by the parent) is unchanged.
        if tp.transformer_point_encoder:
            self.point_tokenizer = TransformerPointEncoder(
                num_tokens=cfg.point_tokenizer.num_tokens,
                embed_size=cfg.embed_size,
                num_layers=tp.point_enc_layers,
                nhead=tp.nhead,
                use_checkpoint=False).to(device)  # policy is outer-loop only (single backward)

        # Direct belief->action path (doors parity). The TokenDecoder FiLM is the
        # *only* other route from `q` to the output, and it is a dead end from a
        # cold init: `FilmBlock` computes `out = scale*out + bias` with `scale`
        # centered at 0 (not `1+gamma`), then passes it through an `MLP_layer`
        # ending in LayerNorm, then adds the q-INDEPENDENT `residual_conv(x)`
        # highway. Measured consequence: ||dmu|| saturates at ~1e-4 no matter how
        # far `q` is perturbed, and ||dL/dq|| ~ 2.6e-5 against ~31 for
        # `aggregator.project.weight`. Since q0, `dyn` and the inner loop are
        # trained ONLY through dL/dq, that near-zero gradient freezes all of them
        # at init — which is exactly what the epoch-1 checkpoint shows.
        #
        # `q_head` restores an O(1) gradient path, mirroring how doors' GMMHead
        # concatenates `q` into its heads and adds a q-only residual. Normal small
        # init (not zero) because the trunk is trained from scratch here, so there
        # is no pretrained behaviour to protect.
        #
        # Only needed for the 'dywa' trunk — `FiLMPolicyTrunk` already carries an
        # equivalent (and stronger) direct path via its concat-q head + q_res.
        self.q_head = (nn.Linear(tp.latent_dim, 2 * cfg.action_size).to(device)
                       if tp.policy_trunk == 'dywa' else None)

        # Rebuild the optimizer so it covers q0 / dyn / outcome_enc / q_head /
        # new decoder / new point encoder.
        self.optimizer = th.optim.Adam(self.parameters(), self.cfg.learning_rate)

    # ── belief-conditioned policy forward ────────────────────────────────────
    def belief_forward(self, obs, q):
        """Student action distribution conditioned on belief `q`.

        obs: dict with `partial_cloud [B,N,3]`, `goal_cloud [B,N,3]` (if use_gpcd)
             and the state keys in `self.input_keys`.
        q:   `[B, latent_dim]`.
        Returns `output [B, 2, action_size]` (`[...,0,:]`=mu, `[...,1,:]`=log_std).

        Stateless: the `dywa/base` aggregator is `mlp` (non-recurrent), so this is
        a pure function of `(obs, q)` — no memory to carry across steps.
        """
        cfg = self.cfg
        ctx_tokens = [
            self.tokenizer[k](obs[k])[:, None]
            for k in self.input_keys if k in obs
        ]
        pcd_tokens = self.point_tokenizer(obs['partial_cloud'])
        vision_tokens = pcd_tokens # torch.Size([1, 16, 128])
        if cfg.use_gpcd:
            gpcd_tokens = self.point_tokenizer(obs['goal_cloud'])
            vision_tokens = th.cat([pcd_tokens, gpcd_tokens], dim=-2) # torch.Size([1, 32, 128])

        embed_tokens = self.encoder(vision_tokens) # torch.Size([1, 32, 128])
        tokens = th.cat(ctx_tokens + [embed_tokens], dim=-2)  # torch.Size([1, 36, 128])

        if self.film_trunk is not None:
            return self.film_trunk(tokens, q)                 # [B, 2, action_size]

        decoded = self.decoder(tokens, q) # torch.Size([1, 37, 128])
        decoded = decoded.reshape(*decoded.shape[:-2], -1)
        output = self.aggregator(decoded) # torch.Size([1, 40])
        output = output.reshape(*output.shape[:-1], 2, -1)
        # Direct belief->action term. Without this the FiLM route attenuates `q`
        # to ~1e-4 of the output and dL/dq collapses to ~1e-5, starving q0/dyn/
        # outcome of gradient (see `q_head` in __init__).
        output = output + self.q_head(q).reshape(*q.shape[:-1], 2, -1)
        return output

    # ── TTT inner loop ───────────────────────────────────────────────────────
    def broadcast_q0(self, batch: int) -> th.Tensor:
        return self.q0.unsqueeze(0).expand(batch, -1)

    def build_macro(self, ee_delta, mean_gains=None, action_seq=None):
        """Assemble the K-window macro-action `[*, macro_dim]` per `macro_action`.

        ee_delta   : `[*, 6]`                 net EE motion (Δpos frame-normalized ‖ rotvec)
        mean_gains : `[*, n_gains]`           window-mean gains  (ee_mean_gains only)
        action_seq : `[*, K, action_size]`    raw per-step actions (seq only)

        For `seq` this runs the (double-backward-safe) `macro_encoder`; callers must
        invoke it *with grad enabled* so the outer meta-gradient reaches the encoder.
        """
        mode = self.cfg.ttt.macro_action
        if mode == 'ee_only':
            return ee_delta
        if mode == 'ee_mean_gains':
            return th.cat([ee_delta, mean_gains], dim=-1)
        lead = action_seq.shape[:-2]
        K, A = action_seq.shape[-2], action_seq.shape[-1]
        macro = self.macro_encoder(action_seq.reshape(-1, K, A))
        return macro.reshape(*lead, -1)

    def inner_update(self, q, hist_clouds, hist_macro, hist_flow,
                     create_graph: bool, detach_q: Optional[bool] = None):
        if detach_q is None:
            detach_q = self.cfg.ttt.detach_q
        return ttt_inner_update(
            q, self.dyn, self.outcome_enc,
            hist_clouds, hist_macro, hist_flow,
            ttt_alpha=self.cfg.ttt.ttt_alpha,
            create_graph=create_graph, detach_q=detach_q)

    def adapt(self, q, hist_clouds, hist_macro, hist_flow,
              k_steps: Optional[int] = None, create_graph: bool = True,
              detach_q: Optional[bool] = None):
        """Run `k_steps` TTT inner updates; returns the adapted belief `q_k`.

        `hist_macro` is the per-window macro-action `[*, H, macro_dim]` (see
        `build_macro`); `hist_flow` is the shared-frame rigid flow `[*, H, N, 3]`.
        """
        if k_steps is None:
            k_steps = self.cfg.ttt.k_inner_steps
        for _ in range(k_steps):
            q, _ = self.inner_update(
                q, hist_clouds, hist_macro, hist_flow,
                create_graph=create_graph, detach_q=detach_q)
        return q


def test_1():
    batch_size = 5
    cfg = StudentAgentRMA.StudentAgentRMAConfig(
        shapes={
            'goal': 7,
            'hand_state': 7,
            'robot_state': 14,
            'previous_action': 20
        },
        batch_size=batch_size,
        max_delay_steps=0,
        without_teacher=False,
        pose_dim=7
    )
    ic(cfg)
    student = StudentAgentRMA(cfg, None, None).to("cuda")
    ic(student)
    obs1 = {
        'goal': th.rand(batch_size, 7, device="cuda"),
        'hand_state': th.rand(batch_size, 7, device="cuda"),
        'robot_state': th.rand(batch_size, 14, device="cuda"),
        'previous_action': th.rand(batch_size, 20, device="cuda"),
        'teacher_state': th.rand(batch_size, 128, device="cuda"),
        'partial_cloud': th.rand(batch_size, 84, 3, device="cuda")

    }
    state = student.reset(obs1)
    print(state.shape)
    obs2 = {
        'goal': th.rand(batch_size, 7, device="cuda"),
        'hand_state': th.rand(batch_size, 7, device="cuda"),
        'robot_state': th.rand(batch_size, 14, device="cuda"),
        'previous_action': th.rand(batch_size, 20, device="cuda"),
        'teacher_state': th.rand(batch_size, 128, device="cuda"),
        'partial_cloud': th.rand(batch_size, 310, 3, device="cuda")
    }
    done = th.zeros(batch_size, dtype=th.bool, device="cuda")
    state2 = student(obs2, 1, done)
    print(state2.shape)


def test_deep_gru():
    B: int = 1
    D_X: int = 4
    D_S: int = 8
    N_L: int = 2

    gru_1 = DeepGRU(D_X, D_S, N_L)
    gru_2 = SingleGRU(D_X, D_S)

    x = th.zeros((B, D_X))
    h_1 = th.zeros((N_L, B, D_S))
    h_2 = th.zeros((B, D_S))

    y_1, h_1 = gru_1(x, h_1)
    y_2, h_2 = gru_2(x, h_2)
    print(h_1.shape)
    print(h_2.shape)


def main():
    test_1()
    # test_deep_gru()


if __name__ == "__main__":
    main()