import time
import torch
import torch.nn as nn
from models.decoder import Decoder
from utils.vocab_info import VocabInfo

# Dummy Vocab setup
tok2id = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "x": 3,
    "^": 4,
    "{": 5,
    "}": 6,
    "2": 7,
    "\\frac": 8,
    "y": 9,
    "_": 10,
    "a": 11,
    "i": 12,
    "b": 13,
}
id2tok = [tok for tok in tok2id.keys()] + [f"dummy_{i}" for i in range(50)]

class DummyWords:
    idx2word = id2tok

def make_vocab_info() -> VocabInfo:
    return VocabInfo(
        pad_id=tok2id["<pad>"],
        sos_id=tok2id["<sos>"],
        eos_id=tok2id["<eos>"],
        vocab_size=len(id2tok),
        words=DummyWords(),
    )

def run_benchmark():
    print("=== Running Beam Cache Overhead Benchmark ===")
    vocab_info = make_vocab_info()
    
    # 1. Micro-benchmark comparing old Python-list vs new tensor CPU operations
    print("\n1. Micro-benchmark (B=4, max_len=100, beam=10, 100 steps):")
    device = torch.device("cpu")
    N = 100 * 4 * 10  # 4000 operations
    
    # Python-list simulation
    t0 = time.perf_counter()
    for _ in range(N):
        # init
        rel_list = [0] * (100 * 100)
        # clone
        cloned = rel_list.copy()
        # materialize
        rel_tensor = torch.tensor(cloned, dtype=torch.long).view(100, 100)
        rel_sliced = rel_tensor[:50, :50].to(device)
    list_time = time.perf_counter() - t0
    
    # Tensor simulation
    t0 = time.perf_counter()
    for _ in range(N):
        # init
        rel_cpu = torch.zeros((100, 100), dtype=torch.long)
        # clone
        cloned = rel_cpu.clone()
        # materialize
        rel_sliced = cloned[:50, :50].to(device)
    tensor_time = time.perf_counter() - t0
    
    print(f"Python-list clone & materialize time: {list_time * 1000:.2f} ms")
    print(f"Tensor-based clone & materialize time: {tensor_time * 1000:.2f} ms")
    print(f"Speedup of Tensor vs Python-list: {list_time / tensor_time:.2f}x")
    
    # 2. Decoder beam search profiling
    decoder = Decoder(
        d_model=16,
        nhead=2,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        dc=4,
        cross_coverage=False,
        self_coverage=False,
        vocab_info=vocab_info,
        use_tree_bias=True,
        use_bidirectional=True,
        tree_bias_num_buckets=8,
        tree_bias_mode="full",
    )
    decoder.eval()
    
    # Inputs: B=2 (which is 4 in bidirectional mode)
    src = [torch.randn(2, 2, 2, 16)]
    src_mask = [torch.zeros(2, 2, 2, dtype=torch.bool)]
    
    original_clone = decoder._tree_builder.clone_state
    original_materialize = decoder._tree_builder.materialize_state
    original_clone_r2l = decoder._tree_builder_r2l.clone_state
    original_materialize_r2l = decoder._tree_builder_r2l.materialize_state
    original_forward = decoder.forward

    times = {
        "clone": 0.0,
        "materialize": 0.0,
        "forward": 0.0,
    }

    def wrapped_clone(state):
        t0 = time.perf_counter()
        res = original_clone(state)
        times["clone"] += time.perf_counter() - t0
        return res

    def wrapped_materialize(state, cur_len, device):
        t0 = time.perf_counter()
        res = original_materialize(state, cur_len, device)
        times["materialize"] += time.perf_counter() - t0
        return res

    def wrapped_clone_r2l(state):
        t0 = time.perf_counter()
        res = original_clone_r2l(state)
        times["clone"] += time.perf_counter() - t0
        return res

    def wrapped_materialize_r2l(state, cur_len, device):
        t0 = time.perf_counter()
        res = original_materialize_r2l(state, cur_len, device)
        times["materialize"] += time.perf_counter() - t0
        return res

    def wrapped_forward(*args, **kwargs):
        t0 = time.perf_counter()
        res = original_forward(*args, **kwargs)
        times["forward"] += time.perf_counter() - t0
        return res

    decoder._tree_builder.clone_state = wrapped_clone
    decoder._tree_builder.materialize_state = wrapped_materialize
    decoder._tree_builder_r2l.clone_state = wrapped_clone_r2l
    decoder._tree_builder_r2l.materialize_state = wrapped_materialize_r2l
    decoder.forward = wrapped_forward
    
    print("\n2. Full beam search benchmark (B=2 bidirectional (=4), beam=10, max_len=100):")
    
    # Warmup
    with torch.inference_mode():
        _ = decoder.beam_search(src, src_mask, beam_size=10, max_len=10, alpha=1.0, early_stopping=False, temperature=1.0, use_cache=True)
    
    # Reset times
    times["clone"] = 0.0
    times["materialize"] = 0.0
    times["forward"] = 0.0
    
    # Measure use_cache=True
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = decoder.beam_search(src, src_mask, beam_size=10, max_len=100, alpha=1.0, early_stopping=False, temperature=1.0, use_cache=True)
    cached_beam_time = time.perf_counter() - t0
    
    cached_forward_time = times["forward"]
    cached_clone_time = times["clone"]
    cached_materialize_time = times["materialize"]
    
    # Measure use_cache=False (falls back to build on each step)
    times["forward"] = 0.0
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = decoder.beam_search(src, src_mask, beam_size=10, max_len=100, alpha=1.0, early_stopping=False, temperature=1.0, use_cache=False)
    nocached_beam_time = time.perf_counter() - t0
    nocached_forward_time = times["forward"]
    
    print(f"Beam search time with use_cache=False (no cache, fallback rebuild): {nocached_beam_time * 1000:.2f} ms")
    print(f"Beam search time with use_cache=True (optimized cache): {cached_beam_time * 1000:.2f} ms")
    print(f"Time spent in materialize_state: {cached_materialize_time * 1000:.2f} ms")
    print(f"Time spent in clone_state: {cached_clone_time * 1000:.2f} ms")
    print(f"Time spent in decoder forward (with cache): {cached_forward_time * 1000:.2f} ms")
    print(f"Time spent in decoder forward (without cache): {nocached_forward_time * 1000:.2f} ms")
    
    # Assertions to ensure benchmark is correct
    assert cached_beam_time > 0
    assert nocached_beam_time > 0
    assert tensor_time < list_time, "Tensor cache operations should be faster than list operations"

if __name__ == "__main__":
    run_benchmark()
