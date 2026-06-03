# 01 Repo map for KV-cache task

## Relevant repo shape
- `configs/crohme_config.yaml`: default model settings. ARM and tree-bias are enabled by default: `use_tree_bias: true`, `tree_bias_layers: all`, `cross_coverage: true`, `self_coverage: true`, `beam_size: 10`, `max_len: 200`.
- `models/comer.py`: encodes images, optionally duplicates features for bidirectional mode, then calls `Decoder.forward(...)` for training or `Decoder.beam_search(...)` for inference.
- `models/decoder.py`: owns word/position embeddings, tree relation builders, full-prefix `forward`, and `transform` wrapper.
- `models/transformer/transformer_decoder.py`: full-prefix decoder stack. Layer 0 has no ARM; later layers use ARM built from previous layer attention.
- `models/transformer/attention.py`: custom MHA implementation with ARM and tree `rel_bias`; do not replace it with PyTorch SDPA.
- `models/transformer/arm.py`: ARM computes cumulative coverage by target time, then returns an attention-logit correction.
- `models/transformer/tree_bias.py`: builds relation IDs and maps them to per-head bias. Needs non-square row support for incremental decode.
- `utils/generation_utils.py`: beam search and tree relation state/cache integration. This is where `use_kv_cache` fallback and beam cache reorder must be wired.
- `utils/beam_search.py`: beam scorer; should not need semantic changes.

## Existing call graph
Training/validation loss:
`LitCoMER.training_step/validation_step -> CoMER.forward -> Encoder.forward -> Decoder.forward -> TransformerDecoder.forward -> TransformerDecoderLayer.forward -> MultiheadAttention.forward`

Inference:
`CoMER.beam_search -> Encoder.forward -> Decoder.beam_search -> DecodeModel._beam_search -> Decoder.transform -> Decoder.forward -> full prefix decoder`

New cached inference path:
`CoMER.beam_search -> Encoder.forward -> DecodeModel._beam_search -> Decoder.init_decode_cache -> Decoder.transform_step -> TransformerDecoder.forward_step -> TransformerDecoderLayer.forward_step -> MultiheadAttention.forward_cached_self/cross`

## Target-file risks
- `attention.py`: projection slicing must exactly match existing `multi_head_attention_forward` behavior.
- `arm.py`: update coverage sums only after current cross-attention is computed.
- `generation_utils.py`: beam `reorder_` must happen after `beam_scorer.process(...)` and before appending selected token.
- `tree_bias.py`: full square behavior must remain byte/float exact.
- `decoder.py`: position encoding must use only row `pos`, not recompute prefix.
