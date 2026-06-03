# 04 Target snippets

These are selected snippets from the uploaded consolidated repo, with line numbers for quick orientation. Always inspect the live repo file before editing.

## configs/crohme_config.yaml

```python
0001: ================================================================================
0002: seed_everything: 7
0003: 
0004: trainer:
0005:   checkpoint_callback: true
0006:   callbacks:
0007:     - class_path: pytorch_lightning.callbacks.LearningRateMonitor
0008:       init_args:
0009:         logging_interval: epoch
0010:     - class_path: pytorch_lightning.callbacks.ModelCheckpoint
0011:       init_args:
0012:         save_top_k: 1
0013:         monitor: val_ExpRate # val_ExpRate
0014:         mode: max
0015:         filename: '{epoch}-{val_ExpRate:.4f}'
0016:   # gpus: 1
0017:   gpus: 0,1
0018:   accelerator: ddp #gpu
0019:   check_val_every_n_epoch: 2
0020:   max_epochs: 300
0021:   deterministic: true
0022:   resume_from_checkpoint: null
0023:   val_check_interval: 1.0
0024:   default_root_dir: "checkpoints"
0025:   log_grad_norm: false
0026: 
0027: model:
0028:   d_model: 256
0029:   # false: vanilla left-to-right transformer targets/search.
0030:   # true: legacy BTTR bidirectional targets/search. Bidirectional tree bias is supported.
0031:   use_bidirectional: false
0032:   # -------- tree relative bias --------
0033:   # Supported in both L2R and bidirectional modes.
0034:   use_tree_bias: true
0035:   tree_bias_num_buckets: 16
0036:   tree_bias_mode: full        # dist_only | type_only | full
0037:   tree_bias_layers: all       # all | last1
0038:   tree_bias_rel_set: core     # script | fraction | core
0039:   # -------- encoder --------
0040:   growth_rate: 24
0041:   num_layers: 16
0042:   reduction: 0.5
0043:   bottleneck: true
0044:   use_dropout: true
0045:   encoder_dropout: 0.2
0046:   # -------- decoder --------
0047:   nhead: 8
0048:   num_decoder_layers: 3
0049:   dim_feedforward: 1024
0050:   decoder_dropout: 0.3
0051:   dc: 32
0052:   cross_coverage: true
0053:   self_coverage: true
0054:   # -------- beam search --------
0055:   beam_size: 10
0056:   max_len: 200
0057:   alpha: 1.0
0058:   early_stopping: false
0059:   temperature: 1.0
0060:   # -------- training --------
0061:   optimizer:
0062:     use: SGD # SGD, Adam, AdamW, Adadelta
0063:     # init for SGD
0064:     SGD:
0065:       lr: 0.24
0066:       momentum: 0.9
0067:       weight_decay: 0.0001
0068:     # init for Adam
0069:     Adam:
0070:       lr: 0.12
0071:       betas: [0.9, 0.999]
0072:     # init for AdamW
0073:     AdamW:
0074:       lr: 0.12
0075:       betas: [0.9, 0.999]
0076:       weight_decay: 1e-4
0077:     # init for Adadelta
0078:     Adadelta:
0079:       lr: 1
0080:       weight_decay: 1e-4
0081:       eps: 1e-6
0082: 
0083:   scheduler:
0084:     use: ReduceLROnPlateau # ReduceLROnPlateau
0085:     interval: epoch
0086:     monitor: val_ExpRate
0087:     # init for ReduceLROnPlateau
0088:     ReduceLROnPlateau:
0089:       mode: 'max'
0090:       factor: 0.25
```

## models/comer.py

```python
0001: ================================================================================
0002: from typing import List, Dict, Any
0003: 
0004: import pytorch_lightning as pl
0005: import torch
0006: from torch import FloatTensor, LongTensor
0007: 
0008: from utils.utils import Hypothesis
0009: 
0010: from .decoder import Decoder
0011: from .encoder import Encoder
0012: 
0013: 
0014: from datamodule.vocab import VocabInfo
0015: 
0016: class CoMER(pl.LightningModule):
0017:     def __init__(
0018:         self,
0019:         config: Dict[str, Any],
0020:         vocab_info: VocabInfo,
0021:     ):
0022:         super().__init__()
0023:         mcfg = config["model"]
0024:         # encoder
0025:         d_model             = mcfg.get("d_model", 256)
0026:         growth_rate         = mcfg.get("growth_rate")
0027:         num_layers          = mcfg.get("num_layers", 16)
0028:         reduction           = mcfg.get("reduction", 0.5)
0029:         bottleneck          = mcfg.get("bottleneck", True)
0030:         use_dropout         = mcfg.get("use_dropout", True)
0031:         densenet_dropout    = mcfg.get("encoder_dropout", 0.2)
0032:         # decoder
0033:         nhead               = mcfg.get("nhead", 8)
0034:         num_decoder_layers  = mcfg.get("num_decoder_layers", 3)
0035:         dim_feedforward     = mcfg.get("dim_feedforward", 1024)
0036:         dropout             = mcfg.get("decoder_dropout", 0.3)
0037:         dc                  = mcfg.get("dc", 32)
0038:         cross_coverage      = mcfg.get("cross_coverage", True)
0039:         self_coverage       = mcfg.get("self_coverage", True)
0040: 
0041:         self.use_bidirectional = bool(mcfg.get("use_bidirectional", False))
0042:         use_tree_bias       = mcfg.get("use_tree_bias", True)
0043:         tree_bias_num_buckets = mcfg.get("tree_bias_num_buckets", 16)
0044:         tree_bias_mode      = mcfg.get("tree_bias_mode", "full")
0045:         tree_bias_layers    = mcfg.get("tree_bias_layers", "all")
0046:         tree_bias_rel_set   = mcfg.get("tree_bias_rel_set", "full")
0047: 
0048:         self.encoder = Encoder(
0049:             d_model=d_model, 
0050:             growth_rate=growth_rate, 
0051:             num_layers=num_layers,
0052:             reduction=reduction,
0053:             bottleneck=bottleneck,
0054:             use_dropout=use_dropout,
0055:             densenet_dropout=densenet_dropout
0056:         )
0057:         self.decoder = Decoder(
0058:             d_model=d_model,
0059:             nhead=nhead,
0060:             num_decoder_layers=num_decoder_layers,
0061:             dim_feedforward=dim_feedforward,
0062:             dropout=dropout,
0063:             dc=dc,
0064:             cross_coverage=cross_coverage,
0065:             self_coverage=self_coverage,
0066:             vocab_info=vocab_info,
0067:             use_tree_bias=use_tree_bias,
0068:             tree_bias_num_buckets=tree_bias_num_buckets,
0069:             tree_bias_mode=tree_bias_mode,
0070:             tree_bias_layers=tree_bias_layers,
0071:             tree_bias_rel_set=tree_bias_rel_set,
0072:             use_bidirectional=self.use_bidirectional,
0073:         )
0074: 
0075:     def forward(
0076:         self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, rel_ids: torch.LongTensor = None
0077:     ) -> FloatTensor:
0078:         """run img and tgt
0079: 
0080:         Parameters
0081:         ----------
0082:         img : FloatTensor
0083:             [b, 1, h, w]
0084:         img_mask: LongTensor
0085:             [b, h, w]
0086:         tgt : LongTensor
0087:             [b, l] in L2R mode, [2b, l] in bidirectional mode
0088: 
0089:         Returns
0090:         -------
0091:         FloatTensor
0092:             [b, l, vocab_size] in L2R mode, [2b, l, vocab_size] in bidirectional mode
0093:         """
0094:         feature, mask = self.encoder(img, img_mask)  # [b, t, d]
0095:         if self.use_bidirectional:
0096:             feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
0097:             mask = torch.cat((mask, mask), dim=0)
0098: 
0099:         out = self.decoder(feature, mask, tgt, rel_ids=rel_ids)
0100: 
0101:         return out
0102: 
0103:     def beam_search(
0104:         self,
0105:         img: FloatTensor,
0106:         img_mask: LongTensor,
0107:         beam_size: int,
0108:         max_len: int,
0109:         alpha: float,
0110:         early_stopping: bool,
```

## models/decoder.py

```python
0001: ================================================================================
0002: from typing import List, Optional, Tuple
0003: 
0004: import torch
0005: import torch.nn as nn
0006: from einops import rearrange
0007: from torch import FloatTensor, LongTensor
0008: 
0009: from utils.vocab_info import VocabInfo
0010: 
0011: from .pos_enc import WordPosEnc
0012: from .transformer.arm import AttentionRefinementModule
0013: from .transformer.transformer_decoder import (
0014:     TransformerDecoder,
0015:     TransformerDecoderLayer,
0016: )
0017: from .transformer.tree_bias import TreeRelationBuilder, TreeRelativeBias, CausalR2LTreeRelationBuilder
0018: from utils.generation_utils import DecodeModel
0019: 
0020: 
0021: def _build_transformer_decoder(
0022:     d_model: int,
0023:     nhead: int,
0024:     num_decoder_layers: int,
0025:     dim_feedforward: int,
0026:     dropout: float,
0027:     dc: int,
0028:     cross_coverage: bool,
0029:     self_coverage: bool,
0030:     tree_bias_layers: str = "all",
0031: ) -> nn.TransformerDecoder:
0032:     decoder_layer = TransformerDecoderLayer(
0033:         d_model=d_model,
0034:         nhead=nhead,
0035:         dim_feedforward=dim_feedforward,
0036:         dropout=dropout,
0037:     )
0038:     if cross_coverage or self_coverage:
0039:         arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
0040:     else:
0041:         arm = None
0042: 
0043:     decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm, tree_bias_layers=tree_bias_layers)
0044:     return decoder
0045: 
0046: 
0047: class Decoder(DecodeModel):
0048:     def __init__(
0049:         self,
0050:         d_model: int,
0051:         nhead: int,
0052:         num_decoder_layers: int,
0053:         dim_feedforward: int,
0054:         dropout: float,
0055:         dc: int,
0056:         cross_coverage: bool,
0057:         self_coverage: bool,
0058:         vocab_info: VocabInfo,
0059:         use_tree_bias: bool = True,
0060:         tree_bias_num_buckets: int = 16,
0061:         tree_bias_mode: str = "full",
0062:         tree_bias_layers: str = "all",
0063:         tree_bias_rel_set: str = "full",
0064:         use_bidirectional: bool = False,
0065:     ):
0066:         super().__init__()
0067:         self.vocab_info = vocab_info
0068:         self.use_bidirectional = bool(use_bidirectional)
0069: 
0070:         self.word_embed = nn.Sequential(
0071:             nn.Embedding(vocab_info.vocab_size, d_model), nn.LayerNorm(d_model)
0072:         )
0073: 
0074:         self.pos_enc = WordPosEnc(d_model=d_model)
0075: 
0076:         self.norm = nn.LayerNorm(d_model)
0077: 
0078:         self.model = _build_transformer_decoder(
0079:             d_model=d_model,
0080:             nhead=nhead,
0081:             num_decoder_layers=num_decoder_layers,
0082:             dim_feedforward=dim_feedforward,
0083:             dropout=dropout,
0084:             dc=dc,
0085:             cross_coverage=cross_coverage,
0086:             self_coverage=self_coverage,
0087:             tree_bias_layers=tree_bias_layers,
0088:         )
0089: 
0090:         self.proj = nn.Linear(d_model, vocab_info.vocab_size)
0091: 
0092:         # -----------------------------
0093:         # Tree-structure relative bias
0094:         # -----------------------------
0095:         self.use_tree_bias = bool(use_tree_bias)
0096:         self.tree_bias_layers = tree_bias_layers
0097: 
0098:         if self.use_tree_bias:
0099:             if vocab_info is None or vocab_info.words is None or not hasattr(vocab_info.words, "idx2word"):
0100:                 raise ValueError("Tree bias requires vocab_info.words.idx2word")
0101: 
0102:             type_size = 6 if self.use_bidirectional else 5
0103:             self._tree_builder = TreeRelationBuilder(
0104:                 id2tok=vocab_info.words.idx2word,
0105:                 pad_id=vocab_info.pad_id,
0106:                 num_buckets=tree_bias_num_buckets,
0107:                 mode=tree_bias_mode,
0108:                 rel_set=tree_bias_rel_set,
0109:                 type_size=type_size,
0110:             )
0111:             if self.use_bidirectional:
0112:                 self._tree_builder_r2l = CausalR2LTreeRelationBuilder(
0113:                     id2tok=vocab_info.words.idx2word,
0114:                     pad_id=vocab_info.pad_id,
0115:                     num_buckets=tree_bias_num_buckets,
0116:                     mode=tree_bias_mode,
0117:                     rel_set=tree_bias_rel_set,
0118:                     type_size=type_size,
0119:                 )
0120:             else:
0121:                 self._tree_builder_r2l = None
0122: 
0123:             self._tree_rel_bias = TreeRelativeBias(
0124:                 num_heads=nhead,
0125:                 num_relations=self._tree_builder.num_relations,
0126:             )
0127:         else:
0128:             self._tree_builder = None
0129:             self._tree_builder_r2l = None
0130:             self._tree_rel_bias = None
0131:         # Causal mask cache: keyed by (device_type, device_index, dtype_str)
0132:         # so CPU->CUDA or dtype changes don't reuse a stale/wrong-device mask.
0133:         self._causal_mask_cache = {}
0134: 
0135:     def _build_attention_mask(self, length, device=None, dtype=torch.bool):
0136:         if device is None:
0137:             device = self.device
0138:         cache_key = (device.type, device.index, str(dtype))
0139:         cached = self._causal_mask_cache.get(cache_key)
0140:         if cached is not None and cached.size(0) >= length:
0141:             return cached[:length, :length]
0142: 
0143:         # lazily create causal attention mask (upper triangular = True)
0144:         mask = torch.full(
0145:             (length, length), fill_value=1, dtype=dtype, device=device
0146:         )
0147:         mask.triu_(1)  # zero out the lower diagonal
0148:         self._causal_mask_cache[cache_key] = mask
0149:         return mask
0150: 
0151:     def _build_rel_ids_for_tgt(self, tgt: torch.LongTensor) -> torch.LongTensor:
0152:         if self.use_bidirectional:
0153:             half_B = tgt.shape[0] // 2
0154:             rel_ids_l2r = self._tree_builder.build(tgt[:half_B])
0155:             rel_ids_r2l = self._tree_builder_r2l.build(tgt[half_B:])
0156:             return torch.cat([rel_ids_l2r, rel_ids_r2l], dim=0)
0157:         else:
0158:             return self._tree_builder.build(tgt)
0159: 
0160:     def forward(
0161:         self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor, rel_ids: Optional[LongTensor] = None
0162:     ) -> FloatTensor:
0163:         """generate output for tgt
0164: 
0165:         Parameters
0166:         ----------
0167:         src : FloatTensor
0168:             [b, h, w, d]
0169:         src_mask: LongTensor
0170:             [b, h, w]
0171:         tgt : LongTensor
0172:             [b, l]
0173: 
0174:         Returns
0175:         -------
0176:         FloatTensor
0177:             [b, l, vocab_size]
0178:         """
0179:         B_tgt, l = tgt.size()
0180:         tgt_mask = self._build_attention_mask(l)
0181:         tgt_pad_mask = tgt == self.vocab_info.pad_id
0182: 
0183:         rel_bias = None
0184:         if self.use_tree_bias and self._tree_rel_bias is not None:
0185:             if rel_ids is None:
0186:                 rel_ids = self._build_rel_ids_for_tgt(tgt)
0187:             rel_bias = self._tree_rel_bias(rel_ids, flatten=True)
0188: 
0189:         tgt = self.word_embed(tgt)  # [b, l, d]
0190:         tgt = self.pos_enc(tgt)  # [b, l, d]
0191:         tgt = self.norm(tgt)
0192: 
0193:         h = src.shape[1]
0194:         src = rearrange(src, "b h w d -> (h w) b d")
0195:         src_mask = rearrange(src_mask, "b h w -> b (h w)")
0196:         tgt = rearrange(tgt, "b l d -> l b d")
0197: 
0198:         out = self.model(
0199:             tgt=tgt,
0200:             memory=src,
0201:             height=h,
0202:             tgt_mask=tgt_mask,
0203:             tgt_key_padding_mask=tgt_pad_mask,
0204:             memory_key_padding_mask=src_mask,
0205:             rel_bias=rel_bias,
0206:         )
0207: 
0208:         out = rearrange(out, "l b d -> b l d")
0209:         out = self.proj(out)
0210: 
```

## models/transformer/arm.py

```python
0001: ================================================================================
0002: import torch
0003: import torch.nn as nn
0004: from einops import rearrange, repeat
0005: from torch import Tensor
0006: 
0007: 
0008: class MaskBatchNorm2d(nn.Module):
0009:     def __init__(self, num_features: int):
0010:         super().__init__()
0011:         self.bn = nn.BatchNorm1d(num_features)
0012: 
0013:     def forward(self, x: Tensor, mask: Tensor) -> Tensor:
0014:         """
0015:         Parameters
0016:         ----------
0017:         x: Tensor
0018:             [b, d, h, w]
0019:         mask: Tensor
0020:             [b, 1, h, w], True means padded/invalid.
0021: 
0022:         Returns
0023:         -------
0024:         Tensor
0025:             [b, d, h, w]
0026:         """
0027:         x = rearrange(x, "b d h w -> b h w d")
0028:         mask = mask.squeeze(1)
0029:         not_mask = ~mask
0030: 
0031:         flat_x = x[not_mask, :]
0032:         flat_x = self.bn(flat_x)
0033:         x[not_mask, :] = flat_x
0034: 
0035:         x = rearrange(x, "b h w d -> b d h w")
0036:         return x
0037: 
0038: 
0039: class AttentionRefinementModule(nn.Module):
0040:     def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
0041:         super().__init__()
0042:         assert cross_coverage or self_coverage
0043:         self.nhead = nhead
0044:         self.cross_coverage = cross_coverage
0045:         self.self_coverage = self_coverage
0046: 
0047:         if cross_coverage and self_coverage:
0048:             in_chs = 2 * nhead
0049:         else:
0050:             in_chs = nhead
0051: 
0052:         self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
0053:         self.act = nn.ReLU(inplace=True)
0054: 
0055:         self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
0056:         self.post_norm = MaskBatchNorm2d(nhead)
0057: 
0058:     def forward(
0059:         self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor
0060:     ) -> Tensor:
0061:         """
0062:         Parameters
0063:         ----------
0064:         prev_attn : Tensor
0065:             [(b * nhead), t, l]
0066:         key_padding_mask : Tensor
0067:             [b, l]
0068:         h : int
0069: 
0070:         Returns
0071:         -------
0072:         Tensor
0073:             [(b * nhead), t, l]
0074:         """
0075:         t = curr_attn.shape[1]
0076:         mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)
0077: 
0078:         curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
0079:         prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)
0080: 
0081:         attns = []
0082:         if self.cross_coverage:
0083:             attns.append(prev_attn)
0084:         if self.self_coverage:
0085:             attns.append(curr_attn)
0086:         attns = torch.cat(attns, dim=1)
0087: 
0088:         attns = attns.cumsum(dim=2) - attns
0089:         attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)
0090: 
0091:         cov = self.conv(attns)
0092:         cov = self.act(cov)
0093: 
0094:         cov = cov.masked_fill(mask, 0.0)
0095:         cov = self.proj(cov)
0096: 
0097:         cov = self.post_norm(cov, mask)
0098: 
0099:         cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
0100:         return cov
```

## models/transformer/attention.py

```python
0001: ================================================================================
0002: import warnings
0003: from typing import Optional, Tuple
0004: 
0005: import torch
0006: import torch.nn as nn
0007: import torch.nn.functional as F
0008: from torch import Tensor
0009: from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
0010: 
0011: from .arm import AttentionRefinementModule
0012: 
0013: 
0014: class MultiheadAttention(nn.Module):
0015:     bias_k: Optional[torch.Tensor]
0016:     bias_v: Optional[torch.Tensor]
0017: 
0018:     def __init__(
0019:         self,
0020:         embed_dim,
0021:         num_heads,
0022:         dropout=0.0,
0023:         bias=True,
0024:         add_bias_kv=False,
0025:         add_zero_attn=False,
0026:         kdim=None,
0027:         vdim=None,
0028:     ):
0029:         super(MultiheadAttention, self).__init__()
0030:         self.embed_dim = embed_dim
0031:         self.kdim = kdim if kdim is not None else embed_dim
0032:         self.vdim = vdim if vdim is not None else embed_dim
0033:         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
0034: 
0035:         self.num_heads = num_heads
0036:         self.dropout = dropout
0037:         self.head_dim = embed_dim // num_heads
0038:         assert (
0039:             self.head_dim * num_heads == self.embed_dim
0040:         ), "embed_dim must be divisible by num_heads"
0041: 
0042:         if self._qkv_same_embed_dim is False:
0043:             self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
0044:             self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
0045:             self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
0046:             self.register_parameter("in_proj_weight", None)
0047:         else:
0048:             self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
0049:             self.register_parameter("q_proj_weight", None)
0050:             self.register_parameter("k_proj_weight", None)
0051:             self.register_parameter("v_proj_weight", None)
0052: 
0053:         if bias:
0054:             self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
0055:         else:
0056:             self.register_parameter("in_proj_bias", None)
0057:         self.out_proj = nn.Linear(embed_dim, embed_dim)
0058: 
0059:         if add_bias_kv:
0060:             self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
0061:             self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
0062:         else:
0063:             self.bias_k = self.bias_v = None
0064: 
0065:         self.add_zero_attn = add_zero_attn
0066: 
0067:         self._reset_parameters()
0068: 
0069:     def _reset_parameters(self):
0070:         if self._qkv_same_embed_dim:
0071:             xavier_uniform_(self.in_proj_weight)
0072:         else:
0073:             xavier_uniform_(self.q_proj_weight)
0074:             xavier_uniform_(self.k_proj_weight)
0075:             xavier_uniform_(self.v_proj_weight)
0076: 
0077:         if self.in_proj_bias is not None:
0078:             constant_(self.in_proj_bias, 0.0)
0079:             constant_(self.out_proj.bias, 0.0)
0080:         if self.bias_k is not None:
0081:             xavier_normal_(self.bias_k)
0082:         if self.bias_v is not None:
0083:             xavier_normal_(self.bias_v)
0084: 
0085:     def __setstate__(self, state):
0086:         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
0087:         if "_qkv_same_embed_dim" not in state:
0088:             state["_qkv_same_embed_dim"] = True
0089: 
0090:         super(MultiheadAttention, self).__setstate__(state)
0091: 
0092:     def forward(
0093:         self,
0094:         query: Tensor,
0095:         key: Tensor,
0096:         value: Tensor,
0097:         arm: Optional[AttentionRefinementModule] = None,
0098:         key_padding_mask: Optional[Tensor] = None,
0099:         need_weights: bool = True,
0100:         attn_mask: Optional[Tensor] = None,
0101:         rel_bias: Optional[Tensor] = None,
0102:     ) -> Tuple[Tensor, Optional[Tensor]]:
0103:         if not self._qkv_same_embed_dim:
0104:             return multi_head_attention_forward(
0105:                 query,
0106:                 key,
0107:                 value,
0108:                 arm,
0109:                 self.embed_dim,
0110:                 self.num_heads,
0111:                 self.in_proj_weight,
0112:                 self.in_proj_bias,
0113:                 self.bias_k,
0114:                 self.bias_v,
0115:                 self.add_zero_attn,
0116:                 self.dropout,
0117:                 self.out_proj.weight,
0118:                 self.out_proj.bias,
0119:                 training=self.training,
0120:                 key_padding_mask=key_padding_mask,
0121:                 need_weights=need_weights,
0122:                 attn_mask=attn_mask,
0123:                 use_separate_proj_weight=True,
0124:                 q_proj_weight=self.q_proj_weight,
0125:                 k_proj_weight=self.k_proj_weight,
0126:                 v_proj_weight=self.v_proj_weight,
0127:             )
0128:         else:
0129:             return multi_head_attention_forward(
0130:                 query,
0131:                 key,
0132:                 value,
0133:                 arm,
0134:                 self.embed_dim,
0135:                 self.num_heads,
0136:                 self.in_proj_weight,
0137:                 self.in_proj_bias,
0138:                 self.bias_k,
0139:                 self.bias_v,
0140:                 self.add_zero_attn,
0141:                 self.dropout,
0142:                 self.out_proj.weight,
0143:                 self.out_proj.bias,
0144:                 training=self.training,
0145:                 key_padding_mask=key_padding_mask,
0146:                 need_weights=need_weights,
0147:                 attn_mask=attn_mask,
0148:                 rel_bias=rel_bias,
0149:             )
0150: 
0151: 
0152: 
0153: 
0154: def multi_head_attention_forward(
0155:     query: Tensor,
0156:     key: Tensor,
0157:     value: Tensor,
0158:     arm: Optional[AttentionRefinementModule],
0159:     embed_dim_to_check: int,
0160:     num_heads: int,
0161:     in_proj_weight: Tensor,
0162:     in_proj_bias: Tensor,
0163:     bias_k: Optional[Tensor],
0164:     bias_v: Optional[Tensor],
0165:     add_zero_attn: bool,
0166:     dropout_p: float,
0167:     out_proj_weight: Tensor,
0168:     out_proj_bias: Tensor,
0169:     training: bool = True,
0170:     key_padding_mask: Optional[Tensor] = None,
0171:     need_weights: bool = True,
0172:     attn_mask: Optional[Tensor] = None,
0173:     use_separate_proj_weight: bool = False,
0174:     q_proj_weight: Optional[Tensor] = None,
0175:     k_proj_weight: Optional[Tensor] = None,
0176:     v_proj_weight: Optional[Tensor] = None,
0177:     static_k: Optional[Tensor] = None,
0178:     static_v: Optional[Tensor] = None,
0179:     rel_bias: Optional[Tensor] = None,
0180: ) -> Tuple[Tensor, Optional[Tensor]]:
0181:     tgt_len, bsz, embed_dim = query.size()
0182:     assert embed_dim == embed_dim_to_check
0183:     # allow MHA to have different sizes for the feature dimension
0184:     assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
0185: 
0186:     head_dim = embed_dim // num_heads
0187:     assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
0188:     scaling = float(head_dim) ** -0.5
0189: 
0190:     if not use_separate_proj_weight:
0191:         if (query is key) and (key is value):
0192:             # self-attention
0193:             q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
0194: 
0195:         elif key is value:
0196:             # encoder-decoder attention
0197:             # This is inline in_proj function with in_proj_weight and in_proj_bias
0198:             _b = in_proj_bias
0199:             _start = 0
0200:             _end = embed_dim
0201:             _w = in_proj_weight[_start:_end, :]
0202:             if _b is not None:
0203:                 _b = _b[_start:_end]
0204:             q = F.linear(query, _w, _b)
0205: 
0206:             if key is None:
0207:                 assert value is None
0208:                 k = None
0209:                 v = None
0210:             else:
0211: 
0212:                 # This is inline in_proj function with in_proj_weight and in_proj_bias
0213:                 _b = in_proj_bias
0214:                 _start = embed_dim
0215:                 _end = None
0216:                 _w = in_proj_weight[_start:, :]
0217:                 if _b is not None:
0218:                     _b = _b[_start:]
0219:                 k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
0220: 
```

```python
0220: 
0221:         else:
0222:             # This is inline in_proj function with in_proj_weight and in_proj_bias
0223:             _b = in_proj_bias
0224:             _start = 0
0225:             _end = embed_dim
0226:             _w = in_proj_weight[_start:_end, :]
0227:             if _b is not None:
0228:                 _b = _b[_start:_end]
0229:             q = F.linear(query, _w, _b)
0230: 
0231:             # This is inline in_proj function with in_proj_weight and in_proj_bias
0232:             _b = in_proj_bias
0233:             _start = embed_dim
0234:             _end = embed_dim * 2
0235:             _w = in_proj_weight[_start:_end, :]
0236:             if _b is not None:
0237:                 _b = _b[_start:_end]
0238:             k = F.linear(key, _w, _b)
0239: 
0240:             # This is inline in_proj function with in_proj_weight and in_proj_bias
0241:             _b = in_proj_bias
0242:             _start = embed_dim * 2
0243:             _end = None
0244:             _w = in_proj_weight[_start:, :]
0245:             if _b is not None:
0246:                 _b = _b[_start:]
0247:             v = F.linear(value, _w, _b)
0248:     else:
0249:         q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
0250:         len1, len2 = q_proj_weight_non_opt.size()
0251:         assert len1 == embed_dim and len2 == query.size(-1)
0252: 
0253:         k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
0254:         len1, len2 = k_proj_weight_non_opt.size()
0255:         assert len1 == embed_dim and len2 == key.size(-1)
0256: 
0257:         v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
0258:         len1, len2 = v_proj_weight_non_opt.size()
0259:         assert len1 == embed_dim and len2 == value.size(-1)
0260: 
0261:         if in_proj_bias is not None:
0262:             q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
0263:             k = F.linear(
0264:                 key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
0265:             )
0266:             v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
0267:         else:
0268:             q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
0269:             k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
0270:             v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
0271:     q = q * scaling
0272: 
0273:     if attn_mask is not None:
0274:         assert (
0275:             attn_mask.dtype == torch.float32
0276:             or attn_mask.dtype == torch.float64
0277:             or attn_mask.dtype == torch.float16
0278:             or attn_mask.dtype == torch.uint8
0279:             or attn_mask.dtype == torch.bool
0280:         ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
0281:             attn_mask.dtype
0282:         )
0283:         if attn_mask.dtype == torch.uint8:
0284:             warnings.warn(
0285:                 "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
0286:             )
0287:             attn_mask = attn_mask.to(torch.bool)
0288: 
0289:         if attn_mask.dim() == 2:
0290:             attn_mask = attn_mask.unsqueeze(0)
0291:             if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
0292:                 raise RuntimeError("The size of the 2D attn_mask is not correct.")
0293:         elif attn_mask.dim() == 3:
0294:             if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
0295:                 raise RuntimeError("The size of the 3D attn_mask is not correct.")
0296:         else:
0297:             raise RuntimeError(
0298:                 "attn_mask's dimension {} is not supported".format(attn_mask.dim())
0299:             )
0300:         # attn_mask's dim is 3 now.
0301: 
0302:     # convert ByteTensor key_padding_mask to bool
0303:     if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
0304:         warnings.warn(
0305:             "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
0306:         )
0307:         key_padding_mask = key_padding_mask.to(torch.bool)
0308: 
0309:     if bias_k is not None and bias_v is not None:
0310:         if static_k is None and static_v is None:
0311:             k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
0312:             v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
0313:             if attn_mask is not None:
0314:                 attn_mask = F.pad(attn_mask, (0, 1))
0315:             if key_padding_mask is not None:
0316:                 key_padding_mask = F.pad(key_padding_mask, (0, 1))
0317:         else:
0318:             assert static_k is None, "bias cannot be added to static key."
0319:             assert static_v is None, "bias cannot be added to static value."
0320:     else:
0321:         assert bias_k is None
0322:         assert bias_v is None
0323: 
0324:     q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
0325:     if k is not None:
0326:         k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
0327:     if v is not None:
0328:         v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
0329: 
0330:     if static_k is not None:
0331:         assert static_k.size(0) == bsz * num_heads
0332:         assert static_k.size(2) == head_dim
0333:         k = static_k
0334: 
0335:     if static_v is not None:
0336:         assert static_v.size(0) == bsz * num_heads
0337:         assert static_v.size(2) == head_dim
0338:         v = static_v
0339: 
0340:     src_len = k.size(1)
0341: 
0342:     if key_padding_mask is not None:
0343:         assert key_padding_mask.size(0) == bsz
0344:         assert key_padding_mask.size(1) == src_len
0345: 
0346:     if add_zero_attn:
0347:         src_len += 1
0348:         k = torch.cat(
0349:             [
0350:                 k,
0351:                 torch.zeros(
0352:                     (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
0353:                 ),
0354:             ],
0355:             dim=1,
0356:         )
0357:         v = torch.cat(
0358:             [
0359:                 v,
0360:                 torch.zeros(
0361:                     (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
0362:                 ),
0363:             ],
0364:             dim=1,
0365:         )
0366:         if attn_mask is not None:
0367:             attn_mask = F.pad(attn_mask, (0, 1))
0368:         if key_padding_mask is not None:
0369:             key_padding_mask = F.pad(key_padding_mask, (0, 1))
0370: 
0371:     attn_output_weights = torch.bmm(q, k.transpose(1, 2))
0372:     assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
0373: 
0374:     if rel_bias is not None:
0375:         if rel_bias.dim() == 3:
0376:             if rel_bias.size(0) == bsz:
0377:                 rel_bias = rel_bias.unsqueeze(1).expand(bsz, num_heads, tgt_len, src_len)
0378:                 rel_bias = rel_bias.contiguous().view(bsz * num_heads, tgt_len, src_len)
0379:             elif rel_bias.size(0) == bsz * num_heads:
0380:                 pass
0381:             else:
0382:                 raise RuntimeError(
0383:                     f"rel_bias has invalid first dim: {rel_bias.size(0)} "
0384:                     f"expected {bsz} or {bsz * num_heads}"
0385:                 )
0386:         elif rel_bias.dim() == 4:
0387:             if rel_bias.size(0) == bsz and rel_bias.size(1) == num_heads:
0388:                 rel_bias = rel_bias.contiguous().view(bsz * num_heads, tgt_len, src_len)
0389:             else:
0390:                 raise RuntimeError(
0391:                     f"rel_bias has invalid shape: {tuple(rel_bias.shape)}; "
0392:                     f"expected ({bsz}, {num_heads}, {tgt_len}, {src_len})"
0393:                 )
0394:         else:
0395:             raise RuntimeError(f"rel_bias must have dim 3 or 4, got {rel_bias.dim()}")
0396: 
0397:         if rel_bias.size(1) != tgt_len or rel_bias.size(2) != src_len:
0398:             raise RuntimeError(
0399:                 f"rel_bias length mismatch: got {tuple(rel_bias.shape)}, "
0400:                 f"expected (*, {tgt_len}, {src_len})"
0401:             )
0402: 
0403:         attn_output_weights.add_(rel_bias.to(dtype=attn_output_weights.dtype))
0404: 
0405:     def mask_softmax_dropout(dots):
0406:         if attn_mask is not None:
0407:             if attn_mask.dtype == torch.bool:
0408:                 dots.masked_fill_(attn_mask, float("-inf"))
0409:             else:
0410:                 dots += attn_mask
0411: 
0412:         if key_padding_mask is not None:
0413:             dots = dots.view(bsz, num_heads, tgt_len, src_len)
0414:             dots = dots.masked_fill(
0415:                 key_padding_mask.unsqueeze(1).unsqueeze(2),
0416:                 float("-inf"),
0417:             )
0418:             dots = dots.view(bsz * num_heads, tgt_len, src_len)
0419: 
0420:         attn = F.softmax(dots, dim=-1)
0421:         attn = F.dropout(attn, p=dropout_p, training=training)
0422:         return attn
0423: 
0424:     attention = mask_softmax_dropout(attn_output_weights)
0425:     if arm is not None:
0426:         attn_output_weights -= arm(attention)
0427:         attention = mask_softmax_dropout(attn_output_weights)
0428: 
0429:     attn_output = torch.bmm(attention, v)
0430:     assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
0431:     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
0432:     attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
0433: 
0434:     if need_weights:
0435:         return attn_output, attention
0436:     else:
0437:         return attn_output, None
```

## models/transformer/transformer_decoder.py

```python
0001: ================================================================================
0002: import copy
0003: from functools import partial
0004: from typing import Optional, Tuple
0005: 
0006: import torch.nn as nn
0007: import torch.nn.functional as F
0008: from torch import Tensor
0009: 
0010: from .arm import AttentionRefinementModule
0011: from .attention import MultiheadAttention
0012: 
0013: 
0014: def _get_clones(module, N):
0015:     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
0016: 
0017: 
0018: class TransformerDecoder(nn.Module):
0019:     def __init__(
0020:         self,
0021:         decoder_layer,
0022:         num_layers: int,
0023:         arm: Optional[AttentionRefinementModule],
0024:         norm=None,
0025:         tree_bias_layers: str = "all",
0026:     ):
0027:         super(TransformerDecoder, self).__init__()
0028:         self.layers = _get_clones(decoder_layer, num_layers)
0029:         self.num_layers = num_layers
0030:         self.norm = norm
0031:         self.tree_bias_layers = tree_bias_layers
0032: 
0033:         self.arm = arm
0034: 
0035:     def forward(
0036:         self,
0037:         tgt: Tensor,
0038:         memory: Tensor,
0039:         height: int,
0040:         tgt_mask: Optional[Tensor] = None,
0041:         memory_mask: Optional[Tensor] = None,
0042:         tgt_key_padding_mask: Optional[Tensor] = None,
0043:         memory_key_padding_mask: Optional[Tensor] = None,
0044:         rel_bias: Optional[Tensor] = None,
0045:     ) -> Tensor:
0046:         output = tgt
0047: 
0048:         arm = None
0049:         for i, mod in enumerate(self.layers):
0050:             layer_rel_bias = rel_bias
0051:             if self.tree_bias_layers == "last1" and i != self.num_layers - 1:
0052:                 layer_rel_bias = None
0053: 
0054:             output, attn = mod(
0055:                 output,
0056:                 memory,
0057:                 arm,
0058:                 tgt_mask=tgt_mask,
0059:                 memory_mask=memory_mask,
0060:                 tgt_key_padding_mask=tgt_key_padding_mask,
0061:                 memory_key_padding_mask=memory_key_padding_mask,
0062:                 rel_bias=layer_rel_bias,
0063:             )
0064:             if i != len(self.layers) - 1 and self.arm is not None:
0065:                 arm = partial(self.arm, attn, memory_key_padding_mask, height)
0066: 
0067:         if self.norm is not None:
0068:             output = self.norm(output)
0069: 
0070:         return output
0071: 
0072: 
0073: 
0074: 
0075: class TransformerDecoderLayer(nn.Module):
0076:     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
0077:         super(TransformerDecoderLayer, self).__init__()
0078:         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
0079:         self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
0080:         # Implementation of Feedforward model
0081:         self.linear1 = nn.Linear(d_model, dim_feedforward)
0082:         self.dropout = nn.Dropout(dropout)
0083:         self.linear2 = nn.Linear(dim_feedforward, d_model)
0084: 
0085:         self.norm1 = nn.LayerNorm(d_model)
0086:         self.norm2 = nn.LayerNorm(d_model)
0087:         self.norm3 = nn.LayerNorm(d_model)
0088: 
0089:         self.dropout1 = nn.Dropout(dropout)
0090:         self.dropout2 = nn.Dropout(dropout)
0091:         self.dropout3 = nn.Dropout(dropout)
0092: 
0093:         self.activation = F.relu
0094: 
0095:     def __setstate__(self, state):
0096:         if "activation" not in state:
0097:             state["activation"] = F.relu
0098:         super(TransformerDecoderLayer, self).__setstate__(state)
0099: 
0100:     def forward(
0101:         self,
0102:         tgt: Tensor,
0103:         memory: Tensor,
0104:         arm: Optional[AttentionRefinementModule],
0105:         tgt_mask: Optional[Tensor] = None,
0106:         memory_mask: Optional[Tensor] = None,
0107:         tgt_key_padding_mask: Optional[Tensor] = None,
0108:         memory_key_padding_mask: Optional[Tensor] = None,
0109:         rel_bias: Optional[Tensor] = None,
0110:     ) -> Tensor:
0111:         r"""Pass the inputs (and mask) through the decoder layer.
0112: 
0113:         Args:
0114:             tgt: the sequence to the decoder layer (required).
0115:             memory: the sequence from the last layer of the encoder (required).
0116:             tgt_mask: the mask for the tgt sequence (optional).
0117:             memory_mask: the mask for the memory sequence (optional).
0118:             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
0119:             memory_key_padding_mask: the mask for the memory keys per batch (optional).
0120: 
0121:         Shape:
0122:             see the docs in Transformer class.
0123:         """
0124:         tgt2 = self.self_attn(
0125:             tgt, tgt, tgt,
0126:             attn_mask=tgt_mask,
0127:             key_padding_mask=tgt_key_padding_mask,
0128:             rel_bias=rel_bias,
0129:         )[0]
0130:         tgt = tgt + self.dropout1(tgt2)
0131:         tgt = self.norm1(tgt)
0132:         tgt2, attn = self.multihead_attn(
0133:             tgt,
0134:             memory,
0135:             memory,
0136:             arm=arm,
0137:             attn_mask=memory_mask,
0138:             key_padding_mask=memory_key_padding_mask,
0139:         )
0140:         tgt = tgt + self.dropout2(tgt2)
0141:         tgt = self.norm2(tgt)
0142:         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
0143:         tgt = tgt + self.dropout3(tgt2)
0144:         tgt = self.norm3(tgt)
0145:         return tgt, attn
```

## models/transformer/tree_bias.py

```python
1090:     max_len: int
1091: 
1092: 
1093: class TreeRelationBuilder:
1094:     """
1095:     Build relation ids (B, L, L) from LaTeX token ids (B, L) using a simple
1096:     stack-based parser that tracks nested contexts: SUP/SUB and FRAC (NUM/DEN).
1097: 
1098:     This is an *approximate structural tree* built from nesting contexts.
1099:     """
1100: 
1101:     def __init__(
1102:         self,
1103:         id2tok: Union[Dict[int, str], Sequence[str]],
1104:         pad_id: int = 0,
1105:         num_buckets: int = 8,
1106:         type_size: int = 5,
1107:         mode: str = "full",
1108:         rel_set: str = "full",
1109:     ) -> None:
1110:         self.pad_id = int(pad_id)
1111:         self.num_buckets = int(num_buckets)
1112:         self.type_size = int(type_size)
1113:         self.mode = mode
1114:         rel_set_alias = {
1115:             "supsub": "script",
1116:             "numden": "fraction",
1117:             "frac": "fraction",
1118:             "supsub_frac": "core",
1119:             "script_frac": "core",
1120:             "supsub_numden": "core",
1121:             "full": "core",   
1122:             "all": "core",    
1123:         }
1124:         self.rel_set = rel_set_alias.get(rel_set, rel_set)
1125:         if self.rel_set not in {"script", "fraction", "core"}:
1126:             raise ValueError(
1127:                 f"Unknown rel_set={rel_set!r}. "
1128:                 "Expected one of: script, fraction, core "
1129:                 "(aliases: supsub, numden, supsub_frac, full, all)."
1130:             )
1131: 
1132:         # Normalize id2tok to a list-like sequence
1133:         if isinstance(id2tok, dict):
1134:             max_id = max(id2tok.keys()) if len(id2tok) else -1
1135:             seq: List[str] = [""] * (max_id + 1)
1136:             for k, v in id2tok.items():
1137:                 if 0 <= k <= max_id:
1138:                     seq[k] = v
1139:             self.id2tok = seq
1140:         else:
1141:             self.id2tok = list(id2tok)
1142: 
1143:         # Precompute special token ids once (FAST)
1144:         self.sup_ids = self._find_all({"^", "^{"})
1145:         self.sub_ids = self._find_all({"_", "_{"})
1146:         self.lbrace_ids = self._find_all({"{"})
1147:         self.rbrace_ids = self._find_all({"}"})
1148: 
1149:         self.frac_ids: Set[int] = self._find_all({"\\frac", "\\dfrac", "\\tfrac"})
1150: 
1151:         # Relation id space size
1152:         if self.mode == "dist_only":
1153:             self.num_relations = self.num_buckets
1154:         elif self.mode == "type_only":
1155:             self.num_relations = self.type_size * self.type_size
1156:         else: # "full"
1157:             self.num_relations = self.num_buckets * (self.type_size * self.type_size)
1158: 
1159:     def _pair_to_rel_id(self, pi: Tuple[int, ...], pj: Tuple[int, ...]) -> int:
1160:         pi_clean = () if pi == (TYPE_ROOT,) else pi
1161:         pj_clean = () if pj == (TYPE_ROOT,) else pj
1162: 
1163:         min_l = min(len(pi_clean), len(pj_clean))
1164:         lcp = 0
1165:         while lcp < min_l and pi_clean[lcp] == pj_clean[lcp]:
1166:             lcp += 1
1167: 
1168:         d = len(pi_clean) + len(pj_clean) - 2 * lcp
1169:         db = min(max(d, 0), self.num_buckets - 1)
1170: 
1171:         ti = pi_clean[lcp] if lcp < len(pi_clean) else TYPE_ROOT
1172:         tj = pj_clean[lcp] if lcp < len(pj_clean) else TYPE_ROOT
1173: 
1174:         if self.rel_set == "script":
1175:             keep_i = (ti == TYPE_ROOT) or (ti == TYPE_SUP) or (ti == TYPE_SUB) 
1176:             keep_j = (tj == TYPE_ROOT) or (tj == TYPE_SUP) or (tj == TYPE_SUB)
1177:             if not keep_i:
1178:                 ti = TYPE_ROOT
1179:             if not keep_j:
1180:                 tj = TYPE_ROOT
1181:         elif self.rel_set == "fraction":
1182:             keep_i = (ti == TYPE_ROOT) or (ti == TYPE_NUM) or (ti == TYPE_DEN) 
1183:             keep_j = (tj == TYPE_ROOT) or (tj == TYPE_NUM) or (tj == TYPE_DEN) 
1184:             if not keep_i:
1185:                 ti = TYPE_ROOT
1186:             if not keep_j:
1187:                 tj = TYPE_ROOT
1188: 
1189:         if self.mode == "dist_only":
1190:             return db
1191:         elif self.mode == "type_only":
1192:             return ti * self.type_size + tj
1193:         else:
1194:             return db * (self.type_size * self.type_size) + ti * self.type_size + tj
1195: 
1196:     def init_state(self, max_len: int, start_token: int) -> L2RState:
1197:         state = L2RState(
1198:             tokens=[],
1199:             ctx_stack=[],
1200:             ctx_marks=[],
1201:             brace_depth=0,
1202:             pending_ctx=None,
1203:             paths=[],
1204:             rel_cpu=None,
1205:             max_len=max_len,
1206:         )
1207:         self.append_state(state, start_token)
1208:         return state
1209: 
1210:     def append_state(self, state: L2RState, token_id: int) -> None:
1211:         tid = int(token_id)
1212:         state.tokens.append(tid)
1213:         pos = len(state.tokens) - 1
1214: 
1215:         if tid == self.pad_id:
1216:             state.paths.append((TYPE_ROOT,))
1217:             return
1218: 
1219:         has_braces = len(self.lbrace_ids) > 0 and len(self.rbrace_ids) > 0
1220: 
1221:         if tid in self.frac_ids:
1222:             state.pending_ctx = TYPE_NUM
1223:             state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1224:             self._fill_relation_row(state, pos)
1225:             return
1226: 
1227:         if tid in self.sup_ids:
1228:             state.pending_ctx = TYPE_SUP
1229:             state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1230:             self._fill_relation_row(state, pos)
1231:             return
1232:         if tid in self.sub_ids:
1233:             state.pending_ctx = TYPE_SUB
1234:             state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1235:             self._fill_relation_row(state, pos)
1236:             return
1237: 
1238:         if has_braces and tid in self.lbrace_ids:
1239:             state.brace_depth += 1
1240:             if state.pending_ctx is not None:
1241:                 state.ctx_stack.append(state.pending_ctx)
1242:                 state.ctx_marks.append(_CtxMark(ctx=state.pending_ctx, start_depth=state.brace_depth))
1243:                 state.pending_ctx = None
1244:             state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1245:             self._fill_relation_row(state, pos)
1246:             return
1247: 
1248:         if has_braces and tid in self.rbrace_ids:
1249:             if state.ctx_marks and state.brace_depth == state.ctx_marks[-1].start_depth:
1250:                 closed = state.ctx_marks.pop().ctx
1251:                 if state.ctx_stack:
1252:                     state.ctx_stack.pop()
1253:                 if closed == TYPE_NUM:
1254:                     state.pending_ctx = TYPE_DEN
1255:                 elif closed == TYPE_DEN:
1256:                     state.pending_ctx = None
1257:             state.brace_depth = max(0, state.brace_depth - 1)
1258:             state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1259:             self._fill_relation_row(state, pos)
1260:             return
1261: 
1262:         if not has_braces:
1263:             tok = self.id2tok[tid] if tid < len(self.id2tok) else ""
1264:             if tok in ("{", "\\lbrace"):
1265:                 state.brace_depth += 1
1266:                 if state.pending_ctx is not None:
1267:                     state.ctx_stack.append(state.pending_ctx)
1268:                     state.ctx_marks.append(_CtxMark(ctx=state.pending_ctx, start_depth=state.brace_depth))
1269:                     state.pending_ctx = None
1270:                 state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1271:                 self._fill_relation_row(state, pos)
1272:                 return
1273:             if tok in ("}", "\\rbrace"):
1274:                 if state.ctx_marks and state.brace_depth == state.ctx_marks[-1].start_depth:
1275:                     closed = state.ctx_marks.pop().ctx
1276:                     if state.ctx_stack:
1277:                         state.ctx_stack.pop()
1278:                     if closed == TYPE_NUM:
1279:                         state.pending_ctx = TYPE_DEN
1280:                     elif closed == TYPE_DEN:
1281:                         state.pending_ctx = None
1282:                 state.brace_depth = max(0, state.brace_depth - 1)
1283:                 state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1284:                 self._fill_relation_row(state, pos)
1285:                 return
1286: 
1287:         # content token:
1288:         if state.pending_ctx is not None:
1289:             state.ctx_stack.append(state.pending_ctx)
1290:             state.paths.append(tuple(state.ctx_stack))
1291:             state.ctx_stack.pop()
1292: 
1293:             if state.pending_ctx == TYPE_NUM:
1294:                 state.pending_ctx = TYPE_DEN
1295:             elif state.pending_ctx == TYPE_DEN:
1296:                 state.pending_ctx = None
1297:             else:
1298:                 state.pending_ctx = None
1299:             self._fill_relation_row(state, pos)
1300:             return
1301: 
1302:         state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
1303:         self._fill_relation_row(state, pos)
1304: 
1305:     def _fill_relation_row(self, state: L2RState, pos: int) -> None:
1306:         if state.rel_cpu is None:
1307:             return
1308:         pi = state.paths[pos]
1309:         for j in range(pos + 1):
1310:             if state.tokens[j] == self.pad_id:
1311:                 continue
1312:             pj = state.paths[j]
1313:             state.rel_cpu[pos, j] = self._pair_to_rel_id(pi, pj)
1314:             state.rel_cpu[j, pos] = self._pair_to_rel_id(pj, pi)
1315: 
1316:     def write_relation_cache(
1317:         self,
1318:         state: L2RState,
1319:         rel_cache: torch.Tensor,
1320:         batch_idx: int,
1321:         pos: Optional[int] = None,
1322:     ) -> None:
1323:         """Write the newest L2R relation row/column into a batched cache.
1324: 
1325:         rel_cache: (B, max_len, max_len), usually on the decoder device.
1326:         This updates only O(cur_len) entries for one hypothesis and avoids
1327:         storing/cloning a full relation matrix inside every parser state.
1328:         """
1329:         if pos is None:
1330:             pos = len(state.tokens) - 1
1331:         if pos < 0 or pos >= len(state.tokens) or state.tokens[pos] == self.pad_id:
1332:             return
1333: 
1334:         pi = state.paths[pos]
1335:         js: List[int] = []
1336:         row_vals: List[int] = []
1337:         col_vals: List[int] = []
1338:         for j in range(pos + 1):
1339:             if state.tokens[j] == self.pad_id:
1340:                 continue
1341:             pj = state.paths[j]
1342:             js.append(j)
1343:             row_vals.append(self._pair_to_rel_id(pi, pj))
1344:             col_vals.append(self._pair_to_rel_id(pj, pi))
1345: 
1346:         if not js:
1347:             return
1348: 
1349:         device = rel_cache.device
1350:         idx = torch.tensor(js, dtype=torch.long, device=device)
1351:         row = torch.tensor(row_vals, dtype=torch.long, device=device)
1352:         col = torch.tensor(col_vals, dtype=torch.long, device=device)
1353:         rel_cache[batch_idx, pos, idx] = row
1354:         rel_cache[batch_idx, idx, pos] = col
1355: 
1356:     def materialize_state(self, state: L2RState, cur_len: int, device: torch.device) -> torch.Tensor:
1357:         if state.rel_cpu is not None:
1358:             return state.rel_cpu[:cur_len, :cur_len].to(device, non_blocking=True)
1359: 
1360:         # Slow fallback kept for callers outside the optimized beam-search path.
1361:         rel_cpu = torch.zeros((cur_len, cur_len), dtype=torch.long)
1362:         for pos in range(min(cur_len, len(state.tokens))):
1363:             if state.tokens[pos] == self.pad_id:
1364:                 continue
1365:             pi = state.paths[pos]
1366:             for j in range(pos + 1):
1367:                 if state.tokens[j] == self.pad_id:
1368:                     continue
1369:                 pj = state.paths[j]
1370:                 rel_cpu[pos, j] = self._pair_to_rel_id(pi, pj)
```

```python
1650:         is_pad = (tgt_ids == self.pad_id)                                 # (B, L)
1651:         valid = (~is_pad).unsqueeze(2) & (~is_pad).unsqueeze(1)           # (B, L, L)
1652:         rid = rid.masked_fill(~valid, 0).to(torch.long)
1653: 
1654:         return rid
1655: 
1656: 
1657: class TreeRelativeBias(nn.Module):
1658:     """
1659:     Convert discrete relation ids (B, L, L) into per-head bias (B, H, L, L)
1660:     using a learnable embedding table.
1661:     """
1662: 
1663:     def __init__(self, num_heads: int, num_relations: int) -> None:
1664:         super().__init__()
1665:         self.num_heads = int(num_heads)
1666:         self.num_relations = int(num_relations)
1667: 
1668:         self.emb = nn.Embedding(self.num_relations, self.num_heads)
1669:         nn.init.zeros_(self.emb.weight)
1670: 
1671:     def forward(self, rel_ids: torch.LongTensor, flatten: bool = False) -> torch.Tensor:
1672:         if rel_ids.dim() != 3:
1673:             raise ValueError(f"rel_ids must be (B, L, L), got {tuple(rel_ids.shape)}")
1674:         B, L, S = rel_ids.shape
1675:         if L != S:
1676:             raise ValueError(f"rel_ids must be square (B, L, L), got {tuple(rel_ids.shape)}")
1677: 
1678:         # (B, L, L, H) -> (B, H, L, L)
1679:         bias = self.emb(rel_ids).permute(0, 3, 1, 2).contiguous()
1680: 
1681:         if flatten:
1682:             # (B*H, L, L) - matches attn_output_weights shape
1683:             return bias.view(B * self.num_heads, L, L)
1684: 
1685:         return bias
1686: 
1687: 
1688: @dataclass
1689: class CtxNode:
1690:     type: int
```

```python
1830:                 state.rel_cpu[i, j] = self._pair_to_rel_id(pi, pj)
1831: 
1832: 
1833:     def write_relation_cache(
1834:         self,
1835:         state: R2LState,
1836:         rel_cache: torch.Tensor,
1837:         batch_idx: int,
1838:         pos: Optional[int] = None,
1839:     ) -> None:
1840:         """Write the newest causal R2L relation row into a batched cache.
1841: 
1842:         R2L uses a lower-triangular causal relation matrix; the upper triangle
1843:         remains zero and is masked by causal self-attention.
1844:         """
1845:         if pos is None:
1846:             pos = len(state.tokens) - 1
1847:         if pos < 0 or pos >= len(state.tokens) or state.tokens[pos] == self.pad_id:
1848:             return
1849: 
1850:         pi = state.paths[pos]
1851:         js: List[int] = []
1852:         row_vals: List[int] = []
1853:         for j in range(pos + 1):
1854:             if state.tokens[j] == self.pad_id:
1855:                 continue
1856:             pj = state.paths[j]
1857:             js.append(j)
1858:             row_vals.append(self._pair_to_rel_id(pi, pj))
1859: 
1860:         if not js:
1861:             return
1862: 
1863:         device = rel_cache.device
1864:         idx = torch.tensor(js, dtype=torch.long, device=device)
1865:         row = torch.tensor(row_vals, dtype=torch.long, device=device)
1866:         rel_cache[batch_idx, pos, idx] = row
1867: 
1868:     def materialize_state(self, state: R2LState, cur_len: int, device: torch.device) -> torch.Tensor:
1869:         if state.rel_cpu is not None:
1870:             return state.rel_cpu[:cur_len, :cur_len].to(device, non_blocking=True)
1871: 
1872:         # Slow fallback kept for callers outside the optimized beam-search path.
1873:         # Replaying is important for R2L because the parser can mutate earlier
1874:         # token paths after a later operator is seen; the original incremental
1875:         # rel_cpu semantics only wrote the current row at each step.
1876:         replay = R2LState(
1877:             tokens=[],
1878:             frames=[Frame(node=None, operands=[], token_indices=[])],
1879:             path_refs=[],
1880:             paths=[],
```

## utils/generation_utils.py

```python
0511: class DecodeModel(nn.Module):
0512:     @property
0513:     def device(self):
0514:         return next(self.parameters()).device
0515: 
0516:     @abstractmethod
0517:     def transform(
0518:         self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor, rel_ids: Optional[LongTensor] = None
0519:     ) -> FloatTensor:
0520:         """decode one step
0521: 
0522:         Parameters
0523:         ----------
0524:         src : List[FloatTensor]
0525:             [b, t, d]
0526:         src_mask : List[LongTensor]
0527:             [b, t]
0528:         input_ids : LongTensor
0529:             [b, l]
0530: 
0531:         Returns
0532:         -------
0533:         FloatTensor
0534:             [b, l, vocab_size]
0535:         """
0536:         raise NotImplementedError("This is an abstract method.")
0537: 
0538: 
0539:     def beam_search(
0540:         self,
0541:         src: List[FloatTensor],
0542:         src_mask: List[LongTensor],
0543:         beam_size: int,
0544:         max_len: int,
0545:         alpha: float,
0546:         early_stopping: bool,
0547:         temperature: float,
0548:         use_cache: bool = True,
0549:     ) -> List[Hypothesis]:
0550:         """run beam search to decode
0551: 
0552:         Parameters
0553:         ----------
0554:         src : List[FloatTensor]
0555:             [b, t, d]
0556:         src_mask : List[LongTensor]
0557:             [b, t]
0558:         beam_size : int
0559:         max_len : int
0560:         alpha : float
0561:         early_stopping : bool
0562: 
0563:         Returns
0564:         -------
0565:         List[Hypothesis]: [batch_size,]
0566:         """
0567:         if getattr(self, "use_bidirectional", False):
0568:             return self._bidirectional_beam_search(
0569:                 src=src,
0570:                 src_mask=src_mask,
0571:                 beam_size=beam_size,
0572:                 max_len=max_len,
0573:                 alpha=alpha,
0574:                 early_stopping=early_stopping,
0575:                 temperature=temperature,
0576:                 use_cache=use_cache,
0577:             )
0578:         return self._l2r_beam_search(
0579:             src=src,
0580:             src_mask=src_mask,
0581:             beam_size=beam_size,
0582:             max_len=max_len,
0583:             alpha=alpha,
0584:             early_stopping=early_stopping,
0585:             temperature=temperature,
0586:             use_cache=use_cache,
0587:         )
0588: 
0589:     def _l2r_beam_search(
0590:         self,
0591:         src: List[FloatTensor],
0592:         src_mask: List[LongTensor],
0593:         beam_size: int,
0594:         max_len: int,
0595:         alpha: float,
0596:         early_stopping: bool,
0597:         temperature: float,
0598:         use_cache: bool = True,
0599:     ) -> List[Hypothesis]:
0600:         batch_size = src[0].shape[0]
0601:         input_ids = torch.full(
0602:             (batch_size, 1),
0603:             fill_value=self.vocab_info.sos_id,
0604:             dtype=torch.long,
0605:             device=self.device,
0606:         )
0607: 
0608:         beam_scorer = BeamSearchScorer(
0609:             batch_size, beam_size, alpha, early_stopping, self.device, self.vocab_info
0610:         )
0611:         hyps, scores = self._beam_search(
0612:             src=list(src),
0613:             src_mask=list(src_mask),
0614:             input_ids=input_ids,
0615:             beam_scorer=beam_scorer,
0616:             beam_size=beam_size,
0617:             max_len=max_len,
0618:             temperature=temperature,
0619:             use_cache=use_cache,
0620:         )
```

```python
0738:     def _beam_search(
0739:         self,
0740:         src: List[FloatTensor],
0741:         src_mask: List[LongTensor],
0742:         input_ids: LongTensor,
0743:         beam_scorer: BeamSearchScorer,
0744:         beam_size: int,
0745:         max_len: int,
0746:         temperature: float,
0747:         use_cache: bool = True,
0748:     ) -> Tuple[List[LongTensor], FloatTensor]:
0749:         batch_size, cur_len = input_ids.shape
0750:         vocab_size = self.vocab_info.vocab_size
0751: 
0752:         beam_scores = torch.zeros(batch_size, dtype=torch.float, device=self.device)
0753: 
0754:         use_tree_bias = getattr(self, "use_tree_bias", False) and use_cache
0755:         is_bidirectional = getattr(self, "use_bidirectional", False)
0756: 
0757:         relation_states = []
0758:         rel_cache = None
0759:         if use_tree_bias:
0760:             # Keep one materialized relation tensor per active hypothesis on the
0761:             # decoder device. Parser states remain lightweight; only this cache
0762:             # carries the O(max_len^2) data needed by the existing full-prefix
0763:             # decoder. During beam reorder we copy only the active cur_len block.
0764:             rel_cache = torch.zeros(
0765:                 (batch_size, max_len, max_len),
0766:                 dtype=torch.long,
0767:                 device=self.device,
0768:             )
0769:             for i in range(batch_size):
0770:                 start_tok = int(input_ids[i, 0].item())
0771:                 if is_bidirectional and i >= batch_size // 2:
0772:                     state = self._tree_builder_r2l.init_state(max_len, start_tok)
0773:                     self._tree_builder_r2l.write_relation_cache(state, rel_cache, i)
0774:                 else:
0775:                     state = self._tree_builder.init_state(max_len, start_tok)
0776:                     self._tree_builder.write_relation_cache(state, rel_cache, i)
0777:                 relation_states.append(state)
0778: 
0779:         while cur_len < max_len and not beam_scorer.is_done():
0780:             rel_ids = None
0781:             if use_tree_bias:
0782:                 rel_ids = rel_cache[: len(relation_states), :cur_len, :cur_len]
0783: 
0784:             next_token_logits = (
0785:                 self.transform(src, src_mask, input_ids, rel_ids=rel_ids)[:, -1, :] / temperature
0786:             )
0787:             next_token_scores = F.log_softmax(next_token_logits, dim=-1)
0788: 
0789:             next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
0790:                 next_token_scores
0791:             )
0792:             
0793:             reshape_size = next_token_scores.shape[0] // batch_size
0794:             next_token_scores = rearrange(
0795:                 next_token_scores,
0796:                 "(b m) v -> b (m v)",
0797:                 m=reshape_size,
0798:             )
0799: 
0800:             next_token_scores, next_tokens = torch.topk(
0801:                 next_token_scores, 2 * beam_size, dim=1
0802:             )
0803: 
0804:             next_indices = next_tokens // vocab_size
0805:             next_tokens = next_tokens % vocab_size
0806: 
0807:             if cur_len == 1:
0808:                 input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
0809:                 for i in range(len(src)):
0810:                     src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
0811:                     src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)
0812: 
0813:                 if use_tree_bias:
0814:                     # Expand the batched relation cache once, matching the
0815:                     # repeat(input_ids, "b l -> (b m) l") order.
0816:                     parent_idx = torch.arange(
0817:                         len(relation_states), dtype=torch.long, device=self.device
0818:                     ).repeat_interleave(beam_size)
0819:                     expanded_cache = rel_cache.new_zeros(
0820:                         (len(relation_states) * beam_size, max_len, max_len)
0821:                     )
0822:                     expanded_cache[:, :cur_len, :cur_len] = rel_cache[
0823:                         :, :cur_len, :cur_len
0824:                     ].index_select(0, parent_idx)
0825:                     rel_cache = expanded_cache
0826: 
0827:                     new_states = []
0828:                     for i, state in enumerate(relation_states):
0829:                         for _ in range(beam_size):
0830:                             if is_bidirectional and i >= batch_size // 2:
0831:                                 cloned = self._tree_builder_r2l.clone_state(state)
0832:                             else:
0833:                                 cloned = self._tree_builder.clone_state(state)
0834:                             new_states.append(cloned)
0835:                     relation_states = new_states
0836: 
0837:             beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
0838:                 input_ids=input_ids,
0839:                 next_scores=next_token_scores,
0840:                 next_tokens=next_tokens,
0841:                 next_indices=next_indices,
0842:             )
0843: 
0844:             input_ids = torch.cat(
0845:                 (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
0846:             )
0847: 
0848:             if use_tree_bias:
0849:                 reordered_states = []
0850:                 beam_idx_list = beam_idx.cpu().tolist()
0851:                 beam_next_tokens_list = beam_next_tokens.cpu().tolist()
0852: 
0853:                 # Reorder only the active prefix block, not the whole max_len^2
0854:                 # cache. clone() protects against parent duplicates and in-place
0855:                 # overwrite when several children come from the same parent.
0856:                 active = len(beam_idx_list)
0857:                 parent_block = rel_cache[:, :cur_len, :cur_len].index_select(0, beam_idx).clone()
0858:                 rel_cache[:active, :cur_len, :cur_len] = parent_block
0859:                 if cur_len < max_len:
0860:                     rel_cache[:active, cur_len, :cur_len + 1].zero_()
0861:                     rel_cache[:active, :cur_len + 1, cur_len].zero_()
0862: 
0863:                 for i, parent_idx in enumerate(beam_idx_list):
0864:                     parent_state = relation_states[parent_idx]
0865:                     is_r2l_direction = is_bidirectional and (i >= len(beam_idx_list) // 2)
0866:                     if is_r2l_direction:
0867:                         cloned = self._tree_builder_r2l.clone_state(parent_state)
0868:                         self._tree_builder_r2l.append_state(cloned, beam_next_tokens_list[i])
0869:                         self._tree_builder_r2l.write_relation_cache(cloned, rel_cache, i, pos=cur_len)
0870:                     else:
0871:                         cloned = self._tree_builder.clone_state(parent_state)
0872:                         self._tree_builder.append_state(cloned, beam_next_tokens_list[i])
0873:                         self._tree_builder.write_relation_cache(cloned, rel_cache, i, pos=cur_len)
0874:                     reordered_states.append(cloned)
0875:                 relation_states = reordered_states
0876: 
0877:             cur_len += 1
0878: 
0879:         return beam_scorer.finalize(input_ids, beam_scores)
0880: 
0881:     def _rate(
0882:         self,
0883:         src: List[FloatTensor],
0884:         src_mask: List[LongTensor],
0885:         tgt: LongTensor,
0886:         out: LongTensor,
0887:         alpha: float,
0888:         temperature: float,
0889:     ) -> FloatTensor:
0890:         """rate tgt and output
```

## utils/beam_search.py

```python
0001: ================================================================================
0002: from typing import List, Optional, Tuple
0003: 
0004: import torch
0005: from torch import FloatTensor, LongTensor
0006: 
0007: # modified from
0008: # https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_beam_search.py#L206
0009: class BeamSearchScorer:
0010:     """Active beam-search scorer.
0011: 
0012:     Pass ``vocab`` explicitly rather than relying on a global class attribute.
0013:     All `.item()` / `.tolist()` calls inside this class are acceptable because
0014:     they live outside the hot generation loop used at training time.
0015:     """
0016: 
0017:     def __init__(
0018:         self,
0019:         batch_size: int,
0020:         beam_size: int,
0021:         alpha: float,
0022:         do_early_stopping: bool,
0023:         device: torch.device,
0024:         vocab,  # pass explicitly; do NOT read CROHMEDatamodule.shared_vocab here
0025:     ) -> None:
0026:         self.batch_size = batch_size
0027:         self.beam_size = beam_size
0028:         self.alpha = alpha
0029:         self.device = device
0030:         # vocab must be passed explicitly; do not rely on global shared_vocab
0031:         self.vocab = vocab
0032: 
0033:         self._beam_hyps = [
0034:             BeamHypotheses(beam_size, alpha, do_early_stopping)
0035:             for _ in range(batch_size)
0036:         ]
0037: 
0038:         self._done = torch.tensor(
0039:             [False for _ in range(batch_size)], dtype=torch.bool, device=self.device
0040:         )
0041: 
0042:     def is_done(self) -> bool:
0043:         return self._done.all()
0044: 
0045:     def process(
0046:         self,
0047:         input_ids: LongTensor,
0048:         next_scores: FloatTensor,
0049:         next_tokens: LongTensor,
0050:         next_indices: LongTensor,
0051:     ) -> Tuple[FloatTensor, LongTensor, LongTensor]:
0052:         """score for each beam
0053: 
0054:         Parameters
0055:         ----------
0056:         input_ids : LongTensor
0057:             [b * beam_size, l]
0058:         next_scores : FloatTensor
0059:             [b, 2 * beam_size]
0060:         next_tokens : LongTensor
0061:             [b, 2 * beam_size]
0062:         next_indices : LongTensor
0063:             [b, 2 * beam_size]
0064: 
0065:         Returns
0066:         -------
0067:         Tuple[FloatTensor, LongTensor, LongTensor]
0068:             next_scores: [b * beam_size]
0069:             next_tokens: [b * beam_size]
0070:             next_indices: [b * beam_size]
0071:         """
0072:         next_beam_scores = torch.zeros(
0073:             (self.batch_size, self.beam_size),
0074:             dtype=next_scores.dtype,
0075:             device=self.device,
0076:         )
0077:         next_beam_tokens = torch.zeros(
0078:             (self.batch_size, self.beam_size),
0079:             dtype=next_tokens.dtype,
0080:             device=self.device,
0081:         )
0082:         next_beam_indices = torch.zeros(
0083:             (self.batch_size, self.beam_size),
0084:             dtype=next_indices.dtype,
0085:             device=self.device,
0086:         )
0087: 
0088:         vocab = self.vocab
0089:         for batch_idx, beam_hyp in enumerate(self._beam_hyps):
0090:             if self._done[batch_idx]:
0091:                 assert len(beam_hyp) >= self.beam_size
0092:                 # pad the batch
0093:                 next_beam_scores[batch_idx, :] = 0
0094:                 next_beam_tokens[batch_idx, :] = vocab.pad_id
0095:                 next_beam_indices[batch_idx, :] = batch_idx * self.beam_size
0096:                 continue
0097: 
0098:             beam_idx = 0
0099:             for beam_token_rank, (next_score, next_token, next_index) in enumerate(
0100:                 zip(
0101:                     next_scores[batch_idx],
0102:                     next_tokens[batch_idx],
0103:                     next_indices[batch_idx],
0104:                 )
0105:             ):
0106:                 batch_beam_idx = batch_idx * self.beam_size + next_index
0107:                 # NOTE: .item() calls here are acceptable — this class is NOT on the
0108:                 # hot training/inference path (see DEPRECATED notice at top of file).
0109:                 l2r_done = (
0110:                     input_ids[batch_beam_idx][0].item() == vocab.sos_id
0111:                     and next_token.item() == vocab.eos_id
0112:                 )
0113:                 r2l_done = (
0114:                     input_ids[batch_beam_idx][0].item() == vocab.eos_id
0115:                     and next_token.item() == vocab.sos_id
0116:                 )
0117:                 if l2r_done or r2l_done:
0118:                     if beam_token_rank >= self.beam_size:
0119:                         # if beam_token does not belong to top num_beams tokens, it should not be added
0120:                         continue
0121:                     beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.item())
0122:                 else:
0123:                     # add next predicted token since it is not eos_token
0124:                     next_beam_scores[batch_idx, beam_idx] = next_score
0125:                     next_beam_tokens[batch_idx, beam_idx] = next_token
0126:                     next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
0127:                     beam_idx += 1
0128: 
0129:                 # once the beam for next step is full, don't add more tokens to it.
0130:                 if beam_idx == self.beam_size:
0131:                     break
0132: 
0133:             assert beam_idx == self.beam_size
0134: 
0135:             self._done[batch_idx] = beam_hyp.is_done(
0136:                 best_sum_logprobs=next_beam_scores[batch_idx].max().item(),
0137:                 cur_len=input_ids.shape[-1],
0138:             )
0139: 
0140:         return (
0141:             next_beam_scores.view(-1),
0142:             next_beam_tokens.view(-1),
0143:             next_beam_indices.view(-1),
0144:         )
0145: 
0146:     def finalize(
0147:         self,
0148:         input_ids: LongTensor,
0149:         final_scores: FloatTensor,
0150:     ) -> Tuple[List[LongTensor], FloatTensor]:
0151:         """generate final output
0152: 
0153:         Parameters
0154:         ----------
0155:         input_ids : LongTensor
0156:             [b * beam_size, l]
0157:         final_scores : FloatTensor
0158:             [b * beam_size]
0159: 
0160:         Returns
0161:         -------
0162:         Tuple[List[LongTensor], FloatTensor]
0163:             List[LongTensor]: [b * beam_size] without SOS or EOS token
0164:             FloatTensor: [b * beam_size] corresponding scores
0165:         """
0166:         # finalize all open beam hypotheses and add to generated hypotheses
0167:         for batch_idx, beam_hyp in enumerate(self._beam_hyps):
0168:             if self._done[batch_idx]:
0169:                 continue
0170: 
0171:             # all open beam hypotheses are added to the beam hypothesis
0172:             # beam hypothesis class automatically keeps the best beams
0173:             for beam_id in range(self.beam_size):
0174:                 batch_beam_idx = batch_idx * self.beam_size + beam_id
0175:                 final_score = final_scores[batch_beam_idx].item()
0176:                 final_tokens = input_ids[batch_beam_idx]
0177:                 beam_hyp.add(final_tokens, final_score)
0178: 
0179:         all_hyps: List[LongTensor] = []
0180:         scores: FloatTensor = torch.zeros(
```
