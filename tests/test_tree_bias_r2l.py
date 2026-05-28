import torch
import unittest
from models.transformer.tree_bias import (
    CausalR2LTreeRelationBuilder,
    TYPE_ROOT,
    TYPE_SUP,
    TYPE_SUB,
    TYPE_NUM,
    TYPE_DEN,
    TYPE_UNK,
)
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
    "\\dfrac": 9,
    "\\tfrac": 10,
    "y": 11,
    "_": 12,
    "a": 13,
    "i": 14,
    "b": 15,
    "\\{": 16,
    "\\}": 17,
    "+": 18,
}
id2tok = [tok for tok in tok2id.keys()]

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


class TestTreeBiasR2L(unittest.TestCase):
    def setUp(self):
        self.vocab_info = make_vocab_info()
        self.builder = CausalR2LTreeRelationBuilder(
            id2tok=id2tok,
            pad_id=tok2id["<pad>"],
            num_buckets=8,
            type_size=6,
            mode="full",
            rel_set="full",
        )

    def _path_to_str(self, path_list):
        # Convert path list at a step to a string of leaf context letters
        mapping = {
            TYPE_ROOT: "R",
            TYPE_SUP: "S",
            TYPE_SUB: "S",  # SUP and SUB both map to S in output string for tests
            TYPE_NUM: "N",
            TYPE_DEN: "D",
            TYPE_UNK: "U",
        }
        res = []
        for path in path_list:
            leaf = path[-1]
            res.append(mapping.get(leaf, "?"))
        return "".join(res)

    def test_golden_x_sup_2(self):
        # x ^ { 2 } -> R2L tokens: } 2 { ^ x
        seq = ["}", "2", "{", "^", "x"]
        seq_ids = torch.tensor([tok2id[t] for t in seq], dtype=torch.long)
        
        all_paths = self.builder._paths_for_seq_ids(seq_ids)
        
        expected = [
            "R",
            "RU",
            "RUU",
            "RSSR",
            "RSSRR"
        ]
        
        for i, exp in enumerate(expected):
            step_str = self._path_to_str(all_paths[i])
            self.assertEqual(step_str, exp, f"Failed at step {i} for x ^ {{ 2 }}: got {step_str}, expected {exp}")
            
    def test_golden_nested_x_sup_a_sub_i(self):
        # x ^ { a _ { i } } -> R2L tokens: } } i { _ a { ^ x
        seq = ["}", "}", "i", "{", "_", "a", "{", "^", "x"]
        seq_ids = torch.tensor([tok2id[t] for t in seq], dtype=torch.long)
        
        all_paths = self.builder._paths_for_seq_ids(seq_ids)
        
        # Check final paths (at last index 8)
        # Token 2 ("i") is inside inner group (SUB) nested in outer group (SUP)
        # Expected path of "i" (index 2) is (SUP, SUB)
        final_paths = all_paths[-1]
        
        # Token 0 ("}"): ROOT
        self.assertEqual(final_paths[0], (TYPE_ROOT,))
        # Token 1 ("}"): SUP (outer group opened, resolved to SUP)
        self.assertEqual(final_paths[1], (TYPE_SUP,))
        # Token 2 ("i"): (SUP, SUB) (inner group resolved to SUB, outer is SUP)
        self.assertEqual(final_paths[2], (TYPE_SUP, TYPE_SUB))
        # Token 3 ("{"): (SUP, SUB)
        self.assertEqual(final_paths[3], (TYPE_SUP, TYPE_SUB))
        # Token 4 ("_"): SUP
        self.assertEqual(final_paths[4], (TYPE_SUP,))
        # Token 5 ("a"): SUP
        self.assertEqual(final_paths[5], (TYPE_SUP,))
        # Token 6 ("{"): SUP
        self.assertEqual(final_paths[6], (TYPE_SUP,))
        # Token 7 ("^"): ROOT
        self.assertEqual(final_paths[7], (TYPE_ROOT,))
        # Token 8 ("x"): ROOT
        self.assertEqual(final_paths[8], (TYPE_ROOT,))

    def test_golden_frac(self):
        # \frac { a } { b } -> R2L tokens: } b { } a { \frac
        seq = ["}", "b", "{", "}", "a", "{", "\\frac"]
        seq_ids = torch.tensor([tok2id[t] for t in seq], dtype=torch.long)
        
        all_paths = self.builder._paths_for_seq_ids(seq_ids)
        
        expected = [
            "R",
            "RU",
            "RUU",
            "RUUR",
            "RUURU",
            "RUURUU",
            "RDDRNNR"
        ]
        
        for i, exp in enumerate(expected):
            step_str = self._path_to_str(all_paths[i])
            self.assertEqual(step_str, exp, f"Failed at step {i} for frac: got {step_str}, expected {exp}")

    def test_escaped_braces(self):
        # x + \{ y \} -> R2L tokens: \} y \{ + x
        # Escaped braces should be treated as normal tokens and not push/pop frames
        seq = ["\\}", "y", "\\{", "+", "x"]
        seq_ids = torch.tensor([tok2id[t] for t in seq], dtype=torch.long)
        
        all_paths = self.builder._paths_for_seq_ids(seq_ids)
        
        for i in range(len(seq)):
            step_str = self._path_to_str(all_paths[i])
            expected_str = "R" * (i + 1)
            self.assertEqual(step_str, expected_str)

    def test_pad_and_special_tokens(self):
        # Tokens with <pad> inside: } <pad> 2 { ^ x
        seq = ["}", "<pad>", "2", "{", "^", "x"]
        seq_ids = torch.tensor([tok2id[t] for t in seq], dtype=torch.long)
        
        all_paths = self.builder._paths_for_seq_ids(seq_ids)
        
        expected = [
            "R",
            "RR",      # pad at index 1 gets ROOT
            "RRU",     # 2 at index 2 gets UNK
            "RRUU",    # { at index 3 gets UNK
            "RRSSR",   # ^ at index 4 resolves
            "RRSSRR",  # x at index 5
        ]
        
        for i, exp in enumerate(expected):
            step_str = self._path_to_str(all_paths[i])
            self.assertEqual(step_str, exp)

    def test_relation_id_bounds(self):
        # Build relation ids for a batch and verify all values are valid
        tgt_ids = torch.tensor([
            [tok2id["}"], tok2id["2"], tok2id["{"], tok2id["^"], tok2id["x"]],
            [tok2id["}"], tok2id["b"], tok2id["{"], tok2id["}"], tok2id["a"]]
        ], dtype=torch.long)
        
        rel_ids = self.builder.build(tgt_ids)
        
        # Max relation ID is num_buckets * (type_size * type_size) = 8 * 36 = 288 in full mode.
        self.assertTrue(torch.all(rel_ids >= 0))
        self.assertTrue(torch.all(rel_ids < self.builder.num_relations))
        
        # Verify causal masking: j > i should be zero
        B, L = tgt_ids.shape
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    if j > i:
                        self.assertEqual(rel_ids[b, i, j].item(), 0)

    def test_decoder_fallback(self):
        # Test decoder builds correct relation IDs when rel_ids is None in bidirectional mode
        decoder = Decoder(
            d_model=16,
            nhead=2,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.1,
            dc=4,
            cross_coverage=False,
            self_coverage=False,
            vocab_info=self.vocab_info,
            use_tree_bias=True,
            use_bidirectional=True,
            tree_bias_num_buckets=8,
            tree_bias_mode="full",
        )
        decoder.eval()
        
        src = torch.randn(2, 2, 2, 16)
        src_mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        
        # Batch of 4 targets (2 L2R + 2 R2L)
        tgt = torch.tensor([
            [tok2id["<sos>"], tok2id["x"], tok2id["^"], tok2id["2"]],
            [tok2id["<sos>"], tok2id["y"], tok2id["_"], tok2id["b"]],
            # R2L ones:
            [tok2id["<eos>"], tok2id["2"], tok2id["{"], tok2id["^"]],
            [tok2id["<eos>"], tok2id["b"], tok2id["{"], tok2id["_"]]
        ], dtype=torch.long)
        
        with torch.inference_mode():
            # Should not raise any error and build rel_ids internally
            out = decoder(src.repeat(2, 1, 1, 1), src_mask.repeat(2, 1, 1), tgt, rel_ids=None)
            
        self.assertEqual(out.shape, (4, 4, len(id2tok)))

    def test_state_cache_equivalence(self):
        expressions = [
            ["x", "^", "{", "2", "}"],
            ["x", "_", "{", "i", "}"],
            ["x", "^", "{", "a", "_", "{", "i", "}", "}"],
            ["\\frac", "{", "a", "}", "{", "b", "}"],
            ["\\frac", "{", "x", "^", "{", "2", "}", "}", "{", "y", "_", "{", "i", "}", "}"],
            ["x", "+", "\\{", "y", "\\}"],
            ["x", "<pad>", "^", "{", "2", "}"],
        ]

        # Check L2R first
        l2r_builder = Decoder(
            d_model=16,
            nhead=2,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            dc=4,
            cross_coverage=False,
            self_coverage=False,
            vocab_info=self.vocab_info,
            use_tree_bias=True,
            use_bidirectional=False,
            tree_bias_num_buckets=8,
            tree_bias_mode="full",
        )._tree_builder

        for expr in expressions:
            tokens = [tok2id[t] for t in expr]
            for length in range(1, len(tokens) + 1):
                prefix = tokens[:length]
                full_ids = l2r_builder.build(torch.tensor([prefix], dtype=torch.long))
                state = l2r_builder.init_state(len(tokens), prefix[0])
                for t in prefix[1:]:
                    l2r_builder.append_state(state, t)
                cached_ids = l2r_builder.materialize_state(state, length, torch.device("cpu"))
                self.assertTrue(torch.equal(full_ids[0], cached_ids), f"L2R mismatch for prefix {expr[:length]}: full={full_ids[0]}, cached={cached_ids}")

        # Check R2L
        r2l_builder = self.builder
        for expr in expressions:
            r2l_expr = list(reversed(expr))
            tokens = [tok2id[t] for t in r2l_expr]
            for length in range(1, len(tokens) + 1):
                prefix = tokens[:length]
                full_ids = r2l_builder.build(torch.tensor([prefix], dtype=torch.long))
                state = r2l_builder.init_state(len(tokens), prefix[0])
                for t in prefix[1:]:
                    r2l_builder.append_state(state, t)
                cached_ids = r2l_builder.materialize_state(state, length, torch.device("cpu"))
                self.assertTrue(torch.equal(full_ids[0], cached_ids), f"R2L mismatch for prefix {r2l_expr[:length]}: full={full_ids[0]}, cached={cached_ids}")

    def test_beam_reorder_no_shared_mutable_state(self):
        r2l_builder = self.builder
        parent = r2l_builder.init_state(10, tok2id["}"])
        r2l_builder.append_state(parent, tok2id["2"])
        
        beam1 = r2l_builder.clone_state(parent)
        beam2 = r2l_builder.clone_state(parent)
        
        r2l_builder.append_state(beam1, tok2id["{"])
        r2l_builder.append_state(beam1, tok2id["^"])
        r2l_builder.append_state(beam1, tok2id["x"])
        
        r2l_builder.append_state(beam2, tok2id["{"])
        r2l_builder.append_state(beam2, tok2id["_"])
        r2l_builder.append_state(beam2, tok2id["y"])
        
        rel1 = r2l_builder.materialize_state(beam1, 5, torch.device("cpu"))
        rel2 = r2l_builder.materialize_state(beam2, 5, torch.device("cpu"))
        
        expected_seq1 = torch.tensor([[tok2id["}"], tok2id["2"], tok2id["{"], tok2id["^"], tok2id["x"]]], dtype=torch.long)
        expected_seq2 = torch.tensor([[tok2id["}"], tok2id["2"], tok2id["{"], tok2id["_"], tok2id["y"]]], dtype=torch.long)
        
        exp1 = r2l_builder.build(expected_seq1)[0]
        exp2 = r2l_builder.build(expected_seq2)[0]
        
        self.assertTrue(torch.equal(rel1, exp1))
        self.assertTrue(torch.equal(rel2, exp2))
        self.assertFalse(torch.equal(rel1, rel2))

        # Collect CtxNode ids to verify no sharing of CtxNodes
        def get_all_node_ids(state):
            node_ids = set()
            for fr in state.frames:
                if fr.node is not None:
                    node_ids.add(id(fr.node))
                for op in fr.operands:
                    if op.node is not None:
                        node_ids.add(id(op.node))
                    for p in op.parent_nodes:
                        if p is not None:
                            node_ids.add(id(p))
            for ref in state.path_refs:
                for n in ref:
                    if n is not None:
                        node_ids.add(id(n))
            return node_ids

        parent_nodes = get_all_node_ids(parent)
        beam1_nodes = get_all_node_ids(beam1)
        beam2_nodes = get_all_node_ids(beam2)
        self.assertTrue(parent_nodes.isdisjoint(beam1_nodes))
        self.assertTrue(parent_nodes.isdisjoint(beam2_nodes))
        self.assertTrue(beam1_nodes.isdisjoint(beam2_nodes))

    def test_decoder_equivalence(self):
        decoder = Decoder(
            d_model=16,
            nhead=2,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            dc=4,
            cross_coverage=False,
            self_coverage=False,
            vocab_info=self.vocab_info,
            use_tree_bias=True,
            use_bidirectional=True,
            tree_bias_num_buckets=8,
            tree_bias_mode="full",
        )
        decoder.eval()
        
        src = torch.randn(2, 2, 2, 16)
        src_mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        
        tgt = torch.tensor([
            [tok2id["<sos>"], tok2id["x"], tok2id["^"], tok2id["2"]],
            [tok2id["<sos>"], tok2id["y"], tok2id["_"], tok2id["b"]],
            [tok2id["<eos>"], tok2id["}"], tok2id["2"], tok2id["{"]],
            [tok2id["<eos>"], tok2id["}"], tok2id["b"], tok2id["{"]]
        ], dtype=torch.long)
        
        rel_ids_list = []
        for i in range(4):
            if i < 2:
                state = decoder._tree_builder.init_state(4, tgt[i, 0].item())
                for t in tgt[i, 1:]:
                    decoder._tree_builder.append_state(state, t.item())
                rel_ids_list.append(decoder._tree_builder.materialize_state(state, 4, torch.device("cpu")))
            else:
                state = decoder._tree_builder_r2l.init_state(4, tgt[i, 0].item())
                for t in tgt[i, 1:]:
                    decoder._tree_builder_r2l.append_state(state, t.item())
                rel_ids_list.append(decoder._tree_builder_r2l.materialize_state(state, 4, torch.device("cpu")))
        rel_ids_cached = torch.stack(rel_ids_list, dim=0)
        
        with torch.inference_mode():
            out_none = decoder(src.repeat(2, 1, 1, 1), src_mask.repeat(2, 1, 1), tgt, rel_ids=None)
            out_cached = decoder(src.repeat(2, 1, 1, 1), src_mask.repeat(2, 1, 1), tgt, rel_ids=rel_ids_cached)
            
        self.assertTrue(torch.allclose(out_none, out_cached, atol=1e-5))

    def test_beam_search_equivalence(self):
        decoder = Decoder(
            d_model=16,
            nhead=2,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            dc=4,
            cross_coverage=False,
            self_coverage=False,
            vocab_info=self.vocab_info,
            use_tree_bias=True,
            use_bidirectional=True,
            tree_bias_num_buckets=8,
            tree_bias_mode="full",
        )
        decoder.eval()
        
        src = [torch.randn(2, 2, 2, 16)]
        src_mask = [torch.zeros(2, 2, 2, dtype=torch.bool)]
        
        with torch.inference_mode():
            hyps_cached = decoder.beam_search(
                src=src,
                src_mask=src_mask,
                beam_size=4,
                max_len=10,
                alpha=1.0,
                early_stopping=False,
                temperature=1.0,
                use_cache=True,
            )
            hyps_nocache = decoder.beam_search(
                src=src,
                src_mask=src_mask,
                beam_size=4,
                max_len=10,
                alpha=1.0,
                early_stopping=False,
                temperature=1.0,
                use_cache=False,
            )
            
        self.assertEqual(len(hyps_cached), len(hyps_nocache))
        for h1, h2 in zip(hyps_cached, hyps_nocache):
            self.assertEqual(h1.seq, h2.seq)
            self.assertAlmostEqual(h1.score, h2.score, places=4)


if __name__ == "__main__":
    unittest.main()
