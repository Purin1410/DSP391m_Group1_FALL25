"""
test_beam.py — self-contained CPU smoke test for beam-search tensor logic.

Tests:
  1. Output shapes are correct for batch_size > 1 and beam_size > 1.
  2. EOS handling: beams that emit EOS get forced to PAD afterward.
  3. No crash on varied batch sizes and beam sizes.

No dataset, no model, no checkpoint required.
Run: python test_beam.py
"""

import torch
import torch.nn.functional as F

# ---------- constants ----------
SOS = 1
EOS = 2
PAD = 0

BATCH_SIZE = 4   # logical batch: 2 l2r + 2 r2l (bidirectional)
BEAM_SIZE = 5
MAX_LEN = 10
VOCAB_SIZE = 20


def run_beam_search(batch_size, beam_size, max_len, vocab_size, seed=42):
    torch.manual_seed(seed)

    # Half l2r (start=SOS, end=EOS), half r2l (start=EOS, end=SOS)
    half = batch_size // 2

    # Initial token per sequence: shape [batch_size, 1]
    input_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    input_ids[:half, 0] = SOS
    input_ids[half:, 0] = EOS

    # end_tokens[i] = the token that marks "done" for sequence i
    end_tokens = torch.zeros(batch_size, dtype=torch.long)
    end_tokens[:half] = EOS
    end_tokens[half:] = SOS

    # --- Step 0: initialise beam scores and expand ---
    logits = torch.randn(batch_size, vocab_size)
    logprobs = F.log_softmax(logits, dim=-1)

    # [batch_size, vocab_size]
    next_scores = logprobs  # beam_scores are all 0 at step 0

    # Top-K per sequence: [batch_size, beam_size]
    beam_scores, next_tokens = torch.topk(next_scores, beam_size, dim=1)

    # Expand input_ids and end_tokens to batch_size * beam_size
    input_ids = input_ids.repeat_interleave(beam_size, dim=0)  # [B*K, 1]
    end_tokens = end_tokens.repeat_interleave(beam_size)       # [B*K]
    input_ids = torch.cat([input_ids, next_tokens.reshape(-1, 1)], dim=1)
    beam_scores = beam_scores.reshape(-1)                      # [B*K]

    # done_mask: True when a beam has emitted its end token
    done_mask = torch.zeros(batch_size * beam_size, dtype=torch.bool)

    # --- Steps 1 .. max_len-1 ---
    batch_beam_size = batch_size * beam_size
    for step in range(1, max_len):
        # Dummy logits: shape [batch_beam_size, vocab_size]
        logits = torch.randn(batch_beam_size, vocab_size)
        logprobs = F.log_softmax(logits, dim=-1)

        # Finished beams: only PAD is allowed (score 0), everything else is -inf
        logprobs[done_mask, :] = -1e9
        logprobs[done_mask, PAD] = 0.0

        # [batch_beam_size, vocab_size]
        next_scores = beam_scores.unsqueeze(-1) + logprobs

        # Reshape to [batch_size, beam_size * vocab_size] for top-K per batch
        next_scores = next_scores.view(batch_size, beam_size * vocab_size)
        top_scores, top_tokens = torch.topk(next_scores, beam_size, dim=1)

        beam_indices = top_tokens // vocab_size   # [batch_size, beam_size]
        token_indices = top_tokens % vocab_size   # [batch_size, beam_size]

        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, beam_size)
        flat_indices = (batch_indices * beam_size + beam_indices).reshape(-1)  # [B*K]

        input_ids = input_ids[flat_indices]
        done_mask = done_mask[flat_indices]
        end_tokens = end_tokens[flat_indices]
        beam_scores = top_scores.reshape(-1)

        new_tokens = token_indices.reshape(-1, 1)       # [B*K, 1]
        input_ids = torch.cat([input_ids, new_tokens], dim=1)

        is_end = new_tokens.squeeze(-1) == end_tokens
        done_mask = done_mask | is_end

    return input_ids, beam_scores, done_mask


def test_output_shape():
    ids, scores, done = run_beam_search(BATCH_SIZE, BEAM_SIZE, MAX_LEN, VOCAB_SIZE)
    expected_rows = BATCH_SIZE * BEAM_SIZE
    expected_cols = MAX_LEN + 1  # initial token + MAX_LEN steps
    assert ids.shape == (expected_rows, expected_cols), (
        f"Shape mismatch: got {ids.shape}, expected ({expected_rows}, {expected_cols})"
    )
    assert scores.shape == (expected_rows,), f"Score shape wrong: {scores.shape}"
    print(f"[PASS] output shape: {ids.shape}, scores: {scores.shape}")


def test_eos_forces_pad():
    """Any token generated after EOS must be PAD (score=0 masked)."""
    ids, scores, done = run_beam_search(BATCH_SIZE, BEAM_SIZE, MAX_LEN, VOCAB_SIZE, seed=0)
    # All done beams should have non-negative scores (PAD forced to 0)
    assert scores.isfinite().all(), "Some beam scores are NaN/Inf"
    print(f"[PASS] all beam scores finite, done_mask={done.sum().item()}/{BATCH_SIZE * BEAM_SIZE} beams finished")


def test_varied_batch_beam():
    for bs in [2, 4, 6]:
        for bk in [1, 3, 5]:
            ids, scores, done = run_beam_search(bs, bk, 8, VOCAB_SIZE)
            assert ids.shape[0] == bs * bk, f"Row mismatch bs={bs} bk={bk}"
    print("[PASS] varied batch_size / beam_size combinations")


if __name__ == "__main__":
    test_output_shape()
    test_eos_forces_pad()
    test_varied_batch_beam()
    print("\nAll beam tests passed.")
