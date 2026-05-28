import time
import torch
from models.transformer.tree_bias import CausalR2LTreeRelationBuilder

def run_benchmark():
    # Vocabulary setup containing required symbols
    id2tok = ["<pad>", "<sos>", "<eos>", "x", "^", "{", "}", "2", "a", "b", "_", "i", "\\frac", "\\{", "\\}"]
    pad_id = 0
    
    # Instantiate builder
    builder = CausalR2LTreeRelationBuilder(
        id2tok=id2tok,
        pad_id=pad_id,
        num_buckets=8,
        type_size=6,
        mode="full",
        rel_set="full",
    )
    
    # 4 sequences of length 200 containing brace matching, fractions, subscripts, and padding
    # Let's construct a representative seq
    base_seq = [1] + [6, 7, 5, 4, 3] * 38 + [2] + [0] * 8
    # Length of base_seq is 1 + 190 + 1 + 8 = 200
    
    tgt_ids = torch.tensor([base_seq] * 4, dtype=torch.long)
    
    # Warmup
    _ = builder.build(tgt_ids)
    
    # Run 50 iterations
    t0 = time.perf_counter()
    for _ in range(50):
        _ = builder.build(tgt_ids)
    t1 = time.perf_counter()
    
    dur = (t1 - t0) / 50.0
    print(f"Average duration per build call (B=4, L=200): {dur * 1000:.3f} ms")

if __name__ == "__main__":
    run_benchmark()
