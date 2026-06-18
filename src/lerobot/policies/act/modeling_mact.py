#!/usr/bin/env python

# MACT (Memory-Enhanced Action Chunking Transformer) Correct Implementation
# 包含 FeatureCache, TimeEmbedding, 和正确的 Reset 逻辑

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.configuration_mact import MACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# === [核心组件] 特征缓存与时间编码 ===
class FeatureCache(nn.Module):
    def __init__(self, max_length, feature_dim):
        super().__init__()
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=max_length)
        # 时间编码表：0~max_length
        self.time_embed = nn.Embedding(max_length + 1, feature_dim)

    def reset(self):
        self.buffer.clear()

    def add(self, feature):
        if self.buffer and feature.shape[0] != self.buffer[0].shape[0]:
            self.reset()
        self.buffer.append(feature.detach())

    def get_sequence_with_time_embed(self, current_feature):
        """返回带有时间编码的序列 [T, B, C]"""
        seq = []
        device = current_feature.device
        available_frames = list(self.buffer) + [current_feature]
        
        target_len = self.max_length + 1
        missing = target_len - len(available_frames)
        padding_frame = available_frames[0]
        
        # 填充缺失帧
        for t in range(missing):
            time_code = self.time_embed(torch.tensor(t, device=device))
            seq.append(padding_frame + time_code.view(1, -1, 1, 1))
            
        # 填充真实帧
        for i, feat in enumerate(available_frames):
            t = missing + i 
            time_code = self.time_embed(torch.tensor(t, device=device))
            seq.append(feat + time_code.view(1, -1, 1, 1))

        return seq

class MACTPolicy(PreTrainedPolicy):
    config_class = MACTConfig
    name = "mact"

    def __init__(self, config: MACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = MACT(config)
        
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {"params": [p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad], "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        # [关键修复] 显式重置模型记忆
        if hasattr(self.model, "reset"):
            self.model.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        l1_loss = (F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()
        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()
    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None
    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones((self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device)
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat([self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])])
        action, self.ensembled_actions, self.ensembled_actions_count = (self.ensembled_actions[:, 0], self.ensembled_actions[:, 1:], self.ensembled_actions_count[1:])
        return action

class MACT(nn.Module):
    def __init__(self, config: MACTConfig):
        super().__init__()
        self.config = config
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(self.config.action_feature.shape[0], config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size + (1 if self.config.robot_state_feature else 0)
            self.register_buffer("vae_encoder_pos_enc", create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0))

        # Backbone
        backbone_out_channels = 512
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights, norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            backbone_out_channels = backbone_model.fc.in_features

        # === [核心] Feature Cache ===
        if getattr(config, "use_memory", False) and config.image_features:
            history_len = max(0, config.n_obs_steps - 1)
            self.feature_caches = nn.ModuleList([FeatureCache(max_length=history_len, feature_dim=backbone_out_channels) for _ in config.image_features])

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_out_channels, config.dim_model, kernel_size=1)

        n_1d_tokens = 1 + (1 if self.config.robot_state_feature else 0) + (1 if self.config.env_state_feature else 0)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def reset(self):
        if getattr(self.config, "use_memory", False) and hasattr(self, "feature_caches"):
            for cache in self.feature_caches: cache.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        # 修正: 只取 State 的最后一帧
        if OBS_STATE in batch and batch[OBS_STATE].ndim == 3: batch[OBS_STATE] = batch[OBS_STATE][:, -1]
        if OBS_ENV_STATE in batch and batch[OBS_ENV_STATE].ndim == 3: batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE][:, -1]

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            vae_input = [cls_embed, robot_state_embed, action_embed] if self.config.robot_state_feature else [cls_embed, action_embed]
            vae_input = torch.cat(vae_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full((batch_size, 2 if self.config.robot_state_feature else 1), False, device=batch[OBS_STATE].device)
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(vae_input.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2), key_padding_mask=key_padding_mask)[0]
            latent_pdf = self.vae_encoder_latent_output_proj(cls_token_out)
            mu, log_sigma_x2 = latent_pdf[:, : self.config.latent_dim], latent_pdf[:, self.config.latent_dim:]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(batch[OBS_STATE].device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature: encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature: encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            use_memory = getattr(self.config, "use_memory", False)
            for i, img in enumerate(batch[OBS_IMAGES]):
                if use_memory:
                    cache = self.feature_caches[i]
                    if self.training: # Training: Flatten T into Sequence
                        b, t, c, h, w = img.shape
                        img_flat = img.view(b * t, c, h, w)
                        feat_flat = self.backbone(img_flat)["feature_map"]
                        feat = feat_flat.view(b, t, *feat_flat.shape[1:])
                        feat_list = []
                        for t_idx in range(t):
                            time_code = cache.time_embed(torch.tensor(t_idx, device=img.device))
                            feat_list.append(feat[:, t_idx] + time_code.view(1, -1, 1, 1))
                        cam_features = torch.stack(feat_list, dim=1).view(b * t, *feat_flat.shape[1:])
                    else: # Inference: Use Cache
                        if img.ndim == 5: img = img[:, -1]
                        curr_feat = self.backbone(img)["feature_map"]
                        seq_list = cache.get_sequence_with_time_embed(curr_feat)
                        cache.add(curr_feat)
                        cam_features = torch.stack(seq_list, dim=1)
                        b_curr, t_curr = cam_features.shape[:2]
                        cam_features = cam_features.view(b_curr * t_curr, *cam_features.shape[2:])
                else:
                    cam_features = self.backbone(img)["feature_map"]

                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "n c h w -> (h w) n c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "n c h w -> (h w) n c")

                if use_memory:
                    total_bt = cam_features.shape[1]
                    t_steps = total_bt // batch_size
                    cam_features = einops.rearrange(cam_features, "p (b t) c -> (t p) b c", b=batch_size, t=t_steps)
                   # ✅ 正确代码 (保持 batch=1，只在时间维度复制)
                    cam_pos_embed = einops.repeat(cam_pos_embed, "p 1 c -> (t p) 1 c", t=t_steps)
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros((self.config.chunk_size, batch_size, self.config.dim_model), dtype=encoder_in_pos_embed.dtype, device=encoder_in_pos_embed.device)
        decoder_out = self.decoder(decoder_in, encoder_out, encoder_pos_embed=encoder_in_pos_embed, decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1))
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)

class ACTEncoder(nn.Module):
    def __init__(self, config: MACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()
    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers: x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

class ACTEncoderLayer(nn.Module):
    def __init__(self, config: MACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm: x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm: skip = x; x = self.norm2(x)
        else: x = self.norm1(x); skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm: x = self.norm2(x)
        return x

class ACTDecoder(nn.Module):
    def __init__(self, config: MACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)
    def forward(self, x: Tensor, encoder_out: Tensor, decoder_pos_embed: Tensor | None = None, encoder_pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers: x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None: x = self.norm(x)
        return x

class ACTDecoderLayer(nn.Module):
    def __init__(self, config: MACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed
    def forward(self, x: Tensor, encoder_out: Tensor, decoder_pos_embed: Tensor | None = None, encoder_pos_embed: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm: x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm: skip = x; x = self.norm2(x)
        else: x = self.norm1(x); skip = x
        x = self.multihead_attn(query=self.maybe_add_pos_embed(x, decoder_pos_embed), key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed), value=encoder_out)[0]
        x = skip + self.dropout2(x)
        if self.pre_norm: skip = x; x = self.norm3(x)
        else: x = self.norm2(x); skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm: x = self.norm3(x)
        return x

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000
    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension)
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed

def get_activation_fn(activation: str) -> Callable:
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")