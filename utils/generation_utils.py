from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .utils import Hypothesis, ce_loss, to_tgt_output
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor


# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


def _strip_generated_boundaries(
    seq: torch.Tensor,
    sos_id: int,
    eos_id: int,
) -> torch.Tensor:
    """Strip boundary tokens (leading start token, trailing terminal token)
    from a 1-D generated sequence tensor.

    Rules
    -----
    - Remove the first token if it equals sos_id or eos_id (start token).
    - Remove the last token if it equals eos_id or sos_id (terminal token).
    - Interior tokens are never removed (even if they happen to be sos/eos).
    - Returns empty tensor safely.

    Parameters
    ----------
    seq : torch.Tensor
        1-D tensor of token ids (no PAD, already filtered).
    sos_id : int
    eos_id : int

    Returns
    -------
    torch.Tensor
        Cleaned 1-D tensor comparable to ground-truth label indices.
    """
    if seq.numel() == 0:
        return seq

    boundary_ids = {sos_id, eos_id}

    # Remove leading start token
    start = 0
    if seq[0].item() in boundary_ids:
        start = 1

    # Remove trailing terminal token
    end = seq.numel()
    if end > start and seq[end - 1].item() in boundary_ids:
        end -= 1

    return seq[start:end]


class DecodeModel(pl.LightningModule):
    @abstractmethod
    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """decode one step

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        raise NotImplementedError("This is an abstract method.")

    def decode_step(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        cache: Optional[Dict] = None,
        step: int = 0,
    ) -> Tuple[FloatTensor, Optional[Dict]]:
        """Incremental decode interface — Stage 1 (full-prefix fallback).

        This method produces the same next-token logits as
        ``self.transform(src, src_mask, input_ids)[:, -1, :]``
        but exposes a ``cache`` argument for future KV-cache integration.

        Stage 1 (current):
            Falls back to full-prefix decoding.  ``cache`` is accepted
            but not used internally.

        Stage 2 (TODO — KV-cache):
            Cache per-layer self-attention K/V projections in ``cache``.
            Reorder cache after beam selection.
            Reuse encoder cross-attention projections.
            Only run the last token through each decoder layer.

        Parameters
        ----------
        src : List[FloatTensor]
            Encoder features (possibly beam-expanded).
        src_mask : List[LongTensor]
            Encoder masks (possibly beam-expanded).
        input_ids : LongTensor
            [B, L] full prefix including all previous tokens.
        cache : Optional[Dict]
            KV-cache dict.  Stage 1 ignores this.
        step : int
            Current decode step (0-indexed).

        Returns
        -------
        Tuple[FloatTensor, Optional[Dict]]
            - next_token_logits: [B, vocab_size]
            - updated cache (None in Stage 1)
        """
        # Stage 1: full-prefix fallback.
        # TODO Stage 2: implement per-layer K/V caching here.
        #   - In each TransformerDecoderLayer, cache the self-attention
        #     K and V tensors (shape [B*nhead, L, head_dim]) in
        #     cache["layer_{i}_self_k"], cache["layer_{i}_self_v"].
        #   - On step > 0, run only the last token through embedding,
        #     concatenate with cached K/V, and update cache.
        #   - Cross-attention K/V from encoder can also be cached since
        #     encoder features do not change across decode steps.
        logits = self.transform(src, src_mask, input_ids)[:, -1, :]
        return logits, cache

    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
    ) -> List[Hypothesis]:
        """run beam search to decode

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        beam_size : int
        max_len : int
        alpha : float
        early_stopping : bool

        Returns
        -------
        List[Hypothesis]: [batch_size,]
        """
        batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction
        batch_beam_size = batch_size * beam_size
        half_bb_size = batch_beam_size // 2

        for i in range(len(src)):
            # Bidirectional beam search: duplicate encoder features for l2r + r2l directions.
            # This copy is done ONCE here, before the decode loop, not inside it.
            # TODO: if memory is very tight, keep src as [B,...] and use batch-index
            #       indirection inside the loop instead of materialising the copy.
            src[i] = torch.cat((src[i], src[i]), dim=0)
            src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

        l2r = torch.full(
            (batch_size // 2, 1),
            fill_value=self.vocab_info.sos_id,
            dtype=torch.long,
            device=self.device,
        )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=self.vocab_info.eos_id,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)

        # first beam search
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            temperature=temperature,
        )

        # reverse half last
        for i in range(half_bb_size, batch_beam_size):
            hyps[i] = torch.flip(hyps[i], dims=[0])

        lens = [len(h) + 1 for h in hyps]  # plus to append start token
        r2l_tgt, r2l_out = to_tgt_output(
            hyps[:half_bb_size], "r2l", self.device, self.vocab_info.sos_id, self.vocab_info.eos_id, self.vocab_info.pad_id, pad_to_len=max(lens)
        )
        l2r_tgt, l2r_out = to_tgt_output(
            hyps[half_bb_size:], "l2r", self.device, self.vocab_info.sos_id, self.vocab_info.eos_id, self.vocab_info.pad_id, pad_to_len=max(lens)
        )
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        # calculate final score
        rev_scores = self._rate(src, src_mask, tgt, out, alpha, temperature)
        rev_scores = torch.cat(
            (rev_scores[half_bb_size:], rev_scores[:half_bb_size]), dim=0
        )
        scores = scores + rev_scores

        # [2 * b, beam_size]
        scores = rearrange(scores, "(b m) -> b m", b=batch_size)
        l2r_scores, r2l_scores = torch.chunk(scores, 2, dim=0)
        # [b, 2 * beam_size]
        scores = torch.cat((l2r_scores, r2l_scores), dim=1)
        # [batch_size, ]
        best_scores, best_indices = torch.max(scores, dim=1)
        best_split = best_indices // beam_size
        best_indices = best_indices % beam_size
        batch_indices = torch.arange(
            0, batch_size // 2, dtype=torch.long, device=self.device
        )
        best_indices = (
            best_split * half_bb_size + batch_indices * beam_size + best_indices
        )

        # Post-decode CPU conversion — .cpu().tolist() is allowed here (outside hot loop)
        best_indices_cpu = best_indices.cpu().tolist()
        best_scores_cpu = best_scores.cpu().tolist()

        ret: List[Hypothesis] = []
        for idx, score in zip(best_indices_cpu, best_scores_cpu):
            hpy = Hypothesis(hyps[idx].cpu(), score, "l2r")
            ret.append(hpy)
        return ret

    def _beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        batch_size = input_ids.shape[0]
        half = batch_size // 2

        end_tokens = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        end_tokens[:half] = self.vocab_info.eos_id
        end_tokens[half:] = self.vocab_info.sos_id

        # Expand for beams
        input_ids = input_ids.repeat_interleave(beam_size, dim=0)  # [batch_size * beam_size, seq_len]
        end_tokens = end_tokens.repeat_interleave(beam_size)       # [batch_size * beam_size]

        # ----- B2: Beam-expand encoder features ONCE, before the decode loop -----
        # This ensures transform() is never called with B_tgt > B_src inside
        # the per-token loop, avoiding repeated expand/reshape of encoder features.
        beam_src = []
        beam_src_mask = []
        for s in src:
            beam_src.append(
                s.unsqueeze(1)
                .expand(-1, beam_size, *( [-1] * (s.dim() - 1) ))
                .reshape(batch_size * beam_size, *s.shape[1:])
            )
        for sm in src_mask:
            beam_src_mask.append(
                sm.unsqueeze(1)
                .expand(-1, beam_size, *( [-1] * (sm.dim() - 1) ))
                .reshape(batch_size * beam_size, *sm.shape[1:])
            )
        # Encoder features are now [batch_size * beam_size, ...].
        # Memory cost: one copy (reshape may copy if expand produced stride-0 dim).
        # This copy is NOT repeated every decode step.

        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        done_mask = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=self.device)

        # KV-cache placeholder (Stage 1: unused, passed through for interface compat)
        cache: Optional[Dict] = None

        for step in range(max_len):
            # decode_step uses full-prefix in Stage 1; ready for KV-cache in Stage 2
            next_token_logits, cache = self.decode_step(
                beam_src, beam_src_mask, input_ids, cache=cache, step=step
            )
            next_token_logits = next_token_logits / temperature
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Done beams can only generate PAD with 0 cost
            next_token_scores[done_mask, :] = -1e9
            next_token_scores[done_mask, self.vocab_info.pad_id] = 0.0

            next_scores = beam_scores.unsqueeze(-1) + next_token_scores
            next_scores = next_scores.view(batch_size, beam_size * self.vocab_info.vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, beam_size, dim=1)

            beam_indices = next_tokens // self.vocab_info.vocab_size
            token_indices = next_tokens % self.vocab_info.vocab_size

            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, beam_size)
            flat_indices = (batch_indices * beam_size + beam_indices).view(-1)

            input_ids = input_ids[flat_indices]
            done_mask = done_mask[flat_indices]
            end_tokens = end_tokens[flat_indices]
            beam_scores = next_scores.view(-1)

            # Reorder beam-expanded encoder features after beam selection
            beam_src = [s[flat_indices] for s in beam_src]
            beam_src_mask = [sm[flat_indices] for sm in beam_src_mask]

            token_indices = token_indices.view(-1, 1)
            input_ids = torch.cat([input_ids, token_indices], dim=1)

            is_end_token = token_indices.squeeze(-1) == end_tokens
            done_mask = done_mask | is_end_token

            # NOTE: Early stopping via `done_mask.all()` is intentionally
            # removed.  When done_mask is a CUDA tensor, calling .all()
            # forces a CPU-GPU sync on every decode step, serialising
            # the entire decode loop.  Finished beams are handled purely
            # through tensor masking above (scores set to -1e9 / pad
            # forced to 0.0).

        seq_lens = (input_ids != self.vocab_info.pad_id).sum(dim=1).float()
        final_scores = beam_scores / (seq_lens ** alpha)

        # --- A1: Strip generated boundary tokens ---
        sos_id = self.vocab_info.sos_id
        eos_id = self.vocab_info.eos_id
        pad_id = self.vocab_info.pad_id

        all_hyps = []
        for i in range(batch_size * beam_size):
            seq = input_ids[i]
            non_pad = seq[seq != pad_id]
            stripped = _strip_generated_boundaries(non_pad, sos_id, eos_id)
            all_hyps.append(stripped)

        return all_hyps, final_scores

    def _rate(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        tgt: LongTensor,
        out: LongTensor,
        alpha: float,
        temperature: float,
    ) -> FloatTensor:
        """rate tgt and output

        Parameters
        ----------
        src : List[FloatTensor]
            [b * beam_size, t, d]
        src_mask : List[LongTensor]
            [b * beam_size, t]
        tgt : LongTensor
            [b * beam_size, l]
        out : LongTensor
            [b * beam_size, l]
        alpha : float
        temperature : float

        Returns
        -------
        FloatTensor
            [b * beam_size]
        """
        b = tgt.shape[0]
        beam_size = b // src[0].shape[0]
        chunk_size = beam_size * max(1, 32 // beam_size)

        losses = []
        for i in range(0, b, chunk_size):
            tgt_chunk = tgt[i : i + chunk_size]
            out_chunk = out[i : i + chunk_size]

            src_start = i // beam_size
            src_end = src_start + (tgt_chunk.shape[0] // beam_size)

            src_chunk = [s[src_start:src_end] for s in src]
            src_mask_chunk = [sm[src_start:src_end] for sm in src_mask]

            out_hat = self.transform(src_chunk, src_mask_chunk, tgt_chunk) / temperature

            loss = ce_loss(out_hat, out_chunk, ignore_idx=self.vocab_info.pad_id, reduction="none")
            loss = rearrange(loss, "(b l) -> b l", b=tgt_chunk.shape[0])

            mask = tgt_chunk == self.vocab_info.pad_id
            penalty = (~mask).sum(dim=1) ** alpha
            loss = -torch.sum(loss, dim=1) / penalty

            losses.append(loss)

        return torch.cat(losses, dim=0)
