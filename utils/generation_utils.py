from abc import abstractmethod
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datamodule.datamodule import CROHMEDatamodule
vocab = CROHMEDatamodule.shared_vocab
vocab_size = len(vocab)

from .utils import Hypothesis, ce_loss
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor
from .beam_search import BeamSearchScorer

# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


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
        device = self.device
        batch_size = src[0].shape[0] 

        # Khởi tạo <sos>
        input_ids = torch.full(
            (batch_size, 1),
            fill_value=vocab.SOS_IDX,
            dtype=torch.long,
            device=device,
        )

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            beam_size=beam_size,
            alpha=alpha,
            do_early_stopping=early_stopping,
            device=device,
        )

        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )

        scores = rearrange(scores, "(b m) -> b m", b=batch_size)  # [b, beam]
        best_scores, best_beam = torch.max(scores, dim=1)

        ret = []
        for b in range(batch_size):
            flat_idx = b * beam_size + best_beam[b].item()
            seq = hyps[flat_idx]
            ret.append(Hypothesis(seq, best_scores[b].item(), "l2r"))
        return ret

    def _beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        beam_scorer: BeamSearchScorer,
        beam_size: int,
        max_len: int,
        temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """inner beam search

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids: LongTensor
            [b, 1]
        beam_size : int
        max_len : int

        Returns
        _______
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: [b * beam_size] without SOS or EOS token
            FloatTensor: [b * beam_size] corresponding scores
        """
        batch_size, cur_len = input_ids.shape

        beam_scores = torch.zeros(batch_size, dtype=torch.float, device=self.device)

        while cur_len < max_len and not beam_scorer.is_done():
            next_token_logits = (
                self.transform(src, src_mask, input_ids)[:, -1, :] / temperature
            )
            # [b *, l, v]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            # [batch_size, beam_size * vocab_size]
            reshape_size = next_token_scores.shape[0] // batch_size
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )
            

            # [b, 2 * beam_size]
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1
            )
            # vocab_sz = next_token_logits.shape[-1]

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            if cur_len == 1:
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)

            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )

            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
            )

            cur_len += 1

        return beam_scorer.finalize(input_ids, beam_scores)

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
        out_hat = self.transform(src, src_mask, tgt) / temperature

        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        mask = tgt == vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss
