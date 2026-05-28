from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Set

import torch
import torch.nn as nn

# Relation type ids (keep stable!)
TYPE_ROOT  = 0
TYPE_SUP   = 1
TYPE_SUB   = 2
TYPE_NUM   = 3
TYPE_DEN   = 4
TYPE_UNK   = 5

TYPE_SIZE_L2R = 5
TYPE_SIZE_BIDIR = 6



def distance_bucket_tensor(d: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """
    Simple bucketing: bucket = min(d, num_buckets-1)

    d: (...,) int tensor >= 0
    """
    return torch.clamp(d, min=0, max=int(num_buckets) - 1)


@dataclass
class _CtxMark:
    ctx: int
    start_depth: int


@dataclass
class L2RState:
    tokens: List[int]
    ctx_stack: List[int]
    ctx_marks: List[_CtxMark]
    brace_depth: int
    pending_ctx: Optional[int]
    paths: List[Tuple[int, ...]]
    rel_list: List[int]
    max_len: int


class TreeRelationBuilder:
    """
    Build relation ids (B, L, L) from LaTeX token ids (B, L) using a simple
    stack-based parser that tracks nested contexts: SUP/SUB and FRAC (NUM/DEN).

    This is an *approximate structural tree* built from nesting contexts.
    """

    def __init__(
        self,
        id2tok: Union[Dict[int, str], Sequence[str]],
        pad_id: int = 0,
        num_buckets: int = 8,
        type_size: int = 5,
        mode: str = "full",
        rel_set: str = "full",
    ) -> None:
        self.pad_id = int(pad_id)
        self.num_buckets = int(num_buckets)
        self.type_size = int(type_size)
        self.mode = mode
        rel_set_alias = {
            "supsub": "script",
            "numden": "fraction",
            "frac": "fraction",
            "supsub_frac": "core",
            "script_frac": "core",
            "supsub_numden": "core",
            "full": "core",   
            "all": "core",    
        }
        self.rel_set = rel_set_alias.get(rel_set, rel_set)
        if self.rel_set not in {"script", "fraction", "core"}:
            raise ValueError(
                f"Unknown rel_set={rel_set!r}. "
                "Expected one of: script, fraction, core "
                "(aliases: supsub, numden, supsub_frac, full, all)."
            )

        # Normalize id2tok to a list-like sequence
        if isinstance(id2tok, dict):
            max_id = max(id2tok.keys()) if len(id2tok) else -1
            seq: List[str] = [""] * (max_id + 1)
            for k, v in id2tok.items():
                if 0 <= k <= max_id:
                    seq[k] = v
            self.id2tok = seq
        else:
            self.id2tok = list(id2tok)

        # Precompute special token ids once (FAST)
        self.sup_ids = self._find_all({"^", "^{"})
        self.sub_ids = self._find_all({"_", "_{"})
        self.lbrace_ids = self._find_all({"{"})
        self.rbrace_ids = self._find_all({"}"})

        self.frac_ids: Set[int] = self._find_all({"\\frac", "\\dfrac", "\\tfrac"})

        # Relation id space size
        if self.mode == "dist_only":
            self.num_relations = self.num_buckets
        elif self.mode == "type_only":
            self.num_relations = self.type_size * self.type_size
        else: # "full"
            self.num_relations = self.num_buckets * (self.type_size * self.type_size)

    def _pair_to_rel_id(self, pi: Tuple[int, ...], pj: Tuple[int, ...]) -> int:
        pi_clean = () if pi == (TYPE_ROOT,) else pi
        pj_clean = () if pj == (TYPE_ROOT,) else pj

        min_l = min(len(pi_clean), len(pj_clean))
        lcp = 0
        while lcp < min_l and pi_clean[lcp] == pj_clean[lcp]:
            lcp += 1

        d = len(pi_clean) + len(pj_clean) - 2 * lcp
        db = min(max(d, 0), self.num_buckets - 1)

        ti = pi_clean[lcp] if lcp < len(pi_clean) else TYPE_ROOT
        tj = pj_clean[lcp] if lcp < len(pj_clean) else TYPE_ROOT

        if self.rel_set == "script":
            keep_i = (ti == TYPE_ROOT) or (ti == TYPE_SUP) or (ti == TYPE_SUB) or (ti == TYPE_UNK)
            keep_j = (tj == TYPE_ROOT) or (tj == TYPE_SUP) or (tj == TYPE_SUB) or (tj == TYPE_UNK)
            if not keep_i:
                ti = TYPE_ROOT
            if not keep_j:
                tj = TYPE_ROOT
        elif self.rel_set == "fraction":
            keep_i = (ti == TYPE_ROOT) or (ti == TYPE_NUM) or (ti == TYPE_DEN) or (ti == TYPE_UNK)
            keep_j = (tj == TYPE_ROOT) or (tj == TYPE_NUM) or (tj == TYPE_DEN) or (tj == TYPE_UNK)
            if not keep_i:
                ti = TYPE_ROOT
            if not keep_j:
                tj = TYPE_ROOT

        if self.mode == "dist_only":
            return db
        elif self.mode == "type_only":
            return ti * self.type_size + tj
        else:
            return db * (self.type_size * self.type_size) + ti * self.type_size + tj

    def init_state(self, max_len: int, start_token: int) -> L2RState:
        state = L2RState(
            tokens=[],
            ctx_stack=[],
            ctx_marks=[],
            brace_depth=0,
            pending_ctx=None,
            paths=[],
            rel_list=[0] * (max_len * max_len),
            max_len=max_len,
        )
        self.append_state(state, start_token)
        return state

    def append_state(self, state: L2RState, token_id: int) -> None:
        tid = int(token_id)
        state.tokens.append(tid)
        pos = len(state.tokens) - 1

        if tid == self.pad_id:
            state.paths.append((TYPE_ROOT,))
            return

        has_braces = len(self.lbrace_ids) > 0 and len(self.rbrace_ids) > 0

        if tid in self.frac_ids:
            state.pending_ctx = TYPE_NUM
            state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
            self._fill_relation_row(state, pos)
            return

        if tid in self.sup_ids:
            state.pending_ctx = TYPE_SUP
            state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
            self._fill_relation_row(state, pos)
            return
        if tid in self.sub_ids:
            state.pending_ctx = TYPE_SUB
            state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
            self._fill_relation_row(state, pos)
            return

        if has_braces and tid in self.lbrace_ids:
            state.brace_depth += 1
            if state.pending_ctx is not None:
                state.ctx_stack.append(state.pending_ctx)
                state.ctx_marks.append(_CtxMark(ctx=state.pending_ctx, start_depth=state.brace_depth))
                state.pending_ctx = None
            state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
            self._fill_relation_row(state, pos)
            return

        if has_braces and tid in self.rbrace_ids:
            if state.ctx_marks and state.brace_depth == state.ctx_marks[-1].start_depth:
                closed = state.ctx_marks.pop().ctx
                if state.ctx_stack:
                    state.ctx_stack.pop()
                if closed == TYPE_NUM:
                    state.pending_ctx = TYPE_DEN
                elif closed == TYPE_DEN:
                    state.pending_ctx = None
            state.brace_depth = max(0, state.brace_depth - 1)
            state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
            self._fill_relation_row(state, pos)
            return

        if not has_braces:
            tok = self.id2tok[tid] if tid < len(self.id2tok) else ""
            if tok in ("{", "\\lbrace"):
                state.brace_depth += 1
                if state.pending_ctx is not None:
                    state.ctx_stack.append(state.pending_ctx)
                    state.ctx_marks.append(_CtxMark(ctx=state.pending_ctx, start_depth=state.brace_depth))
                    state.pending_ctx = None
                state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
                self._fill_relation_row(state, pos)
                return
            if tok in ("}", "\\rbrace"):
                if state.ctx_marks and state.brace_depth == state.ctx_marks[-1].start_depth:
                    closed = state.ctx_marks.pop().ctx
                    if state.ctx_stack:
                        state.ctx_stack.pop()
                    if closed == TYPE_NUM:
                        state.pending_ctx = TYPE_DEN
                    elif closed == TYPE_DEN:
                        state.pending_ctx = None
                state.brace_depth = max(0, state.brace_depth - 1)
                state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
                self._fill_relation_row(state, pos)
                return

        # content token:
        if state.pending_ctx is not None:
            state.ctx_stack.append(state.pending_ctx)
            state.paths.append(tuple(state.ctx_stack))
            state.ctx_stack.pop()

            if state.pending_ctx == TYPE_NUM:
                state.pending_ctx = TYPE_DEN
            elif state.pending_ctx == TYPE_DEN:
                state.pending_ctx = None
            else:
                state.pending_ctx = None
            self._fill_relation_row(state, pos)
            return

        state.paths.append(tuple(state.ctx_stack) if state.ctx_stack else (TYPE_ROOT,))
        self._fill_relation_row(state, pos)

    def _fill_relation_row(self, state: L2RState, pos: int) -> None:
        pi = state.paths[pos]
        row_offset = pos * state.max_len
        for j in range(pos + 1):
            if state.tokens[j] == self.pad_id:
                continue
            pj = state.paths[j]
            state.rel_list[row_offset + j] = self._pair_to_rel_id(pi, pj)
            state.rel_list[j * state.max_len + pos] = self._pair_to_rel_id(pj, pi)

    def materialize_state(self, state: L2RState, cur_len: int, device: torch.device) -> torch.Tensor:
        rel = torch.tensor(state.rel_list, dtype=torch.long).view(state.max_len, state.max_len)
        rel_sliced = rel[:cur_len, :cur_len]
        return rel_sliced.to(device, non_blocking=True)

    def clone_state(self, state: L2RState) -> L2RState:
        return L2RState(
            tokens=state.tokens.copy(),
            ctx_stack=state.ctx_stack.copy(),
            ctx_marks=[_CtxMark(ctx=m.ctx, start_depth=m.start_depth) for m in state.ctx_marks],
            brace_depth=state.brace_depth,
            pending_ctx=state.pending_ctx,
            paths=state.paths.copy(),
            rel_list=state.rel_list.copy(),
            max_len=state.max_len,
        )


    def _find_first(self, candidates: Set[str]) -> Optional[int]:
        for i, tok in enumerate(self.id2tok):
            if tok in candidates:
                return i
        return None

    def _find_all(self, candidates: Set[str]) -> Set[int]:
        out = set()
        for i, tok in enumerate(self.id2tok):
            if tok in candidates:
                out.add(i)
        return out

    def _paths_for_seq_ids(self, seq_ids: torch.Tensor) -> List[Tuple[int, ...]]:
        """
        Convert one token-id sequence -> list of context paths (tuple of TYPE_*).
        seq_ids: (L,) CPU tensor
        """
        L = int(seq_ids.numel())
        paths: List[Tuple[int, ...]] = [(TYPE_ROOT,)] * L  # placeholder

        ctx_stack: List[int] = []
        ctx_marks: List[_CtxMark] = []

        brace_depth = 0
        pending_ctx: Optional[int] = None

        # frac_mode: 0 none, 1 expecting numerator, 2 expecting denominator
        frac_mode = 0

        # If braces are missing, we fall back to applying pending_ctx to the *next* content token only.
        has_braces = len(self.lbrace_ids) > 0 and len(self.rbrace_ids) > 0

        for pos in range(L):
            tid = int(seq_ids[pos].item())

            # pad -> keep root/empty
            if tid == self.pad_id:
                paths[pos] = (TYPE_ROOT,)
                continue

            # detect \frac variants
            if tid in self.frac_ids:
                # next group is numerator
                frac_mode = 1
                pending_ctx = TYPE_NUM
                paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                continue

            # detect ^ / _
            if tid in self.sup_ids:
                pending_ctx = TYPE_SUP
                paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                continue
            if tid in self.sub_ids:
                pending_ctx = TYPE_SUB
                paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                continue

            # braces handling
            if has_braces and tid in self.lbrace_ids:
                brace_depth += 1
                if pending_ctx is not None:
                    ctx_stack.append(pending_ctx)
                    ctx_marks.append(_CtxMark(ctx=pending_ctx, start_depth=brace_depth))
                    pending_ctx = None
                paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                continue

            if has_braces and tid in self.rbrace_ids:
                # close group if it matches the latest ctx start
                if ctx_marks and brace_depth == ctx_marks[-1].start_depth:
                    closed = ctx_marks.pop().ctx
                    if ctx_stack:
                        ctx_stack.pop()

                    # frac: after closing numerator -> start denominator
                    if closed == TYPE_NUM:
                        frac_mode = 2
                        pending_ctx = TYPE_DEN
                    elif closed == TYPE_DEN:
                        frac_mode = 0
                        pending_ctx = None
                brace_depth = max(0, brace_depth - 1)
                paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                continue

            # Fallback: if braces were not detected in vocab (has_braces=False),
            # try matching literal brace tokens by string so contexts still work for patterns like ^ { ... }.
            if not has_braces:
                tok = self.id2tok[tid] if tid < len(self.id2tok) else ""
                if tok in ("{", "\\lbrace"):
                    brace_depth += 1
                    if pending_ctx is not None:
                        ctx_stack.append(pending_ctx)
                        ctx_marks.append(_CtxMark(ctx=pending_ctx, start_depth=brace_depth))
                        pending_ctx = None
                    paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                    continue
                if tok in ("}", "\\rbrace"):
                    if ctx_marks and brace_depth == ctx_marks[-1].start_depth:
                        closed = ctx_marks.pop().ctx
                        if ctx_stack:
                            ctx_stack.pop()
                        if closed == TYPE_NUM:
                            frac_mode = 2
                            pending_ctx = TYPE_DEN
                        elif closed == TYPE_DEN:
                            frac_mode = 0
                            pending_ctx = None
                    brace_depth = max(0, brace_depth - 1)
                    paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)
                    continue

            # content token:
            # If a pending context is not followed by a real grouping brace, apply it to
            # this single content token. This handles valid forms like x ^ 2 or x _ i.
            if pending_ctx is not None:
                ctx_stack.append(pending_ctx)
                paths[pos] = tuple(ctx_stack)
                ctx_stack.pop()

                # For an unbraced fraction, treat the next token after \frac as numerator
                # and the following token as denominator. Braced fractions are handled above.
                if pending_ctx == TYPE_NUM:
                    pending_ctx = TYPE_DEN
                    frac_mode = 2
                elif pending_ctx == TYPE_DEN:
                    pending_ctx = None
                    frac_mode = 0
                else:
                    pending_ctx = None
                continue

            # normal: just record current context
            paths[pos] = tuple(ctx_stack) if ctx_stack else (TYPE_ROOT,)

        return paths

    def build(self, tgt_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Build relation ids for decoder self-attention.

        tgt_ids: (B, L) LongTensor on any device
        rel_ids: (B, L, L) LongTensor on tgt_ids.device
        """
        if tgt_ids.dim() != 2:
            raise ValueError(f"tgt_ids must be (B, L), got {tuple(tgt_ids.shape)}")

        device = tgt_ids.device
        B, L = tgt_ids.shape

        # Move ids to CPU once for sequential parsing (tiny copy: B*L ints).
        tgt_cpu = tgt_ids.detach().to("cpu")

        # Build per-token paths (python, O(B*L))
        all_paths: List[List[Tuple[int, ...]]] = []
        max_depth = 1
        for b in range(B):
            paths = self._paths_for_seq_ids(tgt_cpu[b])
            all_paths.append(paths)
            for p in paths:
                if len(p) > max_depth:
                    max_depth = len(p)

        D = max_depth  # depth dimension for padded paths

        # Build padded path tensors on CPU first. This avoids thousands of tiny
        # GPU allocations/copies from inside Python loops.
        P_cpu = torch.full((B, L, D), fill_value=TYPE_ROOT, dtype=torch.long)
        A_cpu = torch.zeros((B, L, D), dtype=torch.bool)

        for b in range(B):
            for i in range(L):
                tid = int(tgt_cpu[b, i].item())
                if tid == self.pad_id:
                    continue
                path = all_paths[b][i]
                # If path is just (ROOT,), treat as empty context
                if len(path) == 1 and path[0] == TYPE_ROOT:
                    continue
                li = len(path)
                P_cpu[b, i, :li] = torch.as_tensor(path, dtype=torch.long)
                A_cpu[b, i, :li] = True

        if device.type == "cpu":
            P = P_cpu
            A = A_cpu
        else:
            P = P_cpu.to(device, non_blocking=True)
            A = A_cpu.to(device, non_blocking=True)

        # lengths (B, L)
        lens = A.sum(dim=-1).to(torch.long)

        # Compute LCP length for every pair (B, L, L) using vectorized ops
        Pi = P.unsqueeze(2)   # (B, L, 1, D)
        Pj = P.unsqueeze(1)   # (B, 1, L, D)
        Ai = A.unsqueeze(2)   # (B, L, 1, D)
        Aj = A.unsqueeze(1)   # (B, 1, L, D)

        eq = (Pi == Pj) & Ai & Aj                     # (B, L, L, D) bool
        eqi = eq.to(torch.int16)                      # small int
        prefix = torch.cumprod(eqi, dim=-1)           # (B, L, L, D)
        lcp = prefix.sum(dim=-1).to(torch.long)       # (B, L, L)

        # distance on the "context tree"
        d = lens.unsqueeze(2) + lens.unsqueeze(1) - 2 * lcp               # (B, L, L)
        db = distance_bucket_tensor(d, self.num_buckets)                  # (B, L, L)

        # Determine the first differing ctx type after the LCP
        P_masked = torch.where(A, P, torch.full_like(P, TYPE_ROOT))
        root_col = torch.full((B, L, 1), TYPE_ROOT, dtype=torch.long, device=device)
        P_ext = torch.cat([P_masked, root_col], dim=-1)                   # (B, L, D+1)

        lcp_idx = torch.clamp(lcp, max=D).unsqueeze(-1)                   # (B, L, L, 1)

        ti = torch.gather(
            P_ext.unsqueeze(2).expand(B, L, L, D + 1),
            dim=3,
            index=lcp_idx
        ).squeeze(-1)                                                     # (B, L, L)

        tj = torch.gather(
            P_ext.unsqueeze(1).expand(B, L, L, D + 1),
            dim=3,
            index=lcp_idx
        ).squeeze(-1)                                                     # (B, L, L)

        # Rel set remap
        # Rel set remap for ablation:
        # - script:   keep ROOT/SUP/SUB only
        # - fraction: keep ROOT/NUM/DEN only
        # - core:     keep ROOT/SUP/SUB/NUM/DEN
        root = torch.tensor(TYPE_ROOT, device=device)

        if self.rel_set == "script":
            keep_i = (ti == TYPE_ROOT) | (ti == TYPE_SUP) | (ti == TYPE_SUB)
            keep_j = (tj == TYPE_ROOT) | (tj == TYPE_SUP) | (tj == TYPE_SUB)
            ti = torch.where(keep_i, ti, root)
            tj = torch.where(keep_j, tj, root)

        elif self.rel_set == "fraction":
            keep_i = (ti == TYPE_ROOT) | (ti == TYPE_NUM) | (ti == TYPE_DEN)
            keep_j = (tj == TYPE_ROOT) | (tj == TYPE_NUM) | (tj == TYPE_DEN)
            ti = torch.where(keep_i, ti, root)
            tj = torch.where(keep_j, tj, root)

        elif self.rel_set == "core":
            pass

        else:
            raise ValueError(f"Unknown rel_set after normalization: {self.rel_set}")

        # relation id
        if self.mode == "dist_only":
            rid = db
        elif self.mode == "type_only":
            rid = ti * self.type_size + tj
        else:
            rid = db * (self.type_size * self.type_size) + ti * self.type_size + tj

        # pad mask: zero out pairs with PAD on either side
        is_pad = (tgt_ids == self.pad_id)                                 # (B, L)
        valid = (~is_pad).unsqueeze(2) & (~is_pad).unsqueeze(1)           # (B, L, L)
        rid = rid.masked_fill(~valid, 0).to(torch.long)

        return rid


class TreeRelativeBias(nn.Module):
    """
    Convert discrete relation ids (B, L, L) into per-head bias (B, H, L, L)
    using a learnable embedding table.
    """

    def __init__(self, num_heads: int, num_relations: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_relations = int(num_relations)

        self.emb = nn.Embedding(self.num_relations, self.num_heads)
        nn.init.zeros_(self.emb.weight)

    def forward(self, rel_ids: torch.LongTensor, flatten: bool = False) -> torch.Tensor:
        if rel_ids.dim() != 3:
            raise ValueError(f"rel_ids must be (B, L, L), got {tuple(rel_ids.shape)}")
        B, L, S = rel_ids.shape
        if L != S:
            raise ValueError(f"rel_ids must be square (B, L, L), got {tuple(rel_ids.shape)}")

        # (B, L, L, H) -> (B, H, L, L)
        bias = self.emb(rel_ids).permute(0, 3, 1, 2).contiguous()

        if flatten:
            # (B*H, L, L) - matches attn_output_weights shape
            return bias.view(B * self.num_heads, L, L)

        return bias


@dataclass
class CtxNode:
    type: int


@dataclass
class Operand:
    kind: str                 # "atom" or "group"
    token_indices: List[int]  # for atom
    node: Optional[CtxNode]   # for group
    parent_nodes: List[CtxNode]


@dataclass
class Frame:
    node: Optional[CtxNode]
    operands: List[Operand]
    token_indices: List[int]


@dataclass
class R2LState:
    tokens: List[int]
    frames: List[Frame]
    path_refs: List[List[CtxNode]]
    paths: List[Tuple[int, ...]]
    rel_list: List[int]
    max_len: int


class CausalR2LTreeRelationBuilder(TreeRelationBuilder):
    """
    Build causal relation ids (B, L, L) from LaTeX token ids (B, L)
    for the R2L decoder direction. The relation ids are computed using a one-pass
    incremental causal parser with mutable context nodes and cached path retrieval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_ids = self._find_all({"<sos>", "<eos>"})

    def init_state(self, max_len: int, start_token: int) -> R2LState:
        state = R2LState(
            tokens=[],
            frames=[Frame(node=None, operands=[], token_indices=[])],
            path_refs=[],
            paths=[],
            rel_list=[0] * (max_len * max_len),
            max_len=max_len,
        )
        self.append_state(state, start_token)
        return state

    def append_state(self, state: R2LState, token_id: int) -> None:
        tid = int(token_id)
        state.tokens.append(tid)
        i = len(state.tokens) - 1

        state.path_refs.append([])
        state.paths.append((TYPE_ROOT,))

        if tid == self.pad_id:
            return

        def current_path_nodes():
            return [fr.node for fr in state.frames if fr.node is not None]

        def materialize(nodes):
            types = [n.type for n in nodes]
            return (TYPE_ROOT,) if len(types) == 0 else tuple(types)

        def resolve_operand(operand: Operand, target_type: int):
            if operand.kind == "group":
                if operand.node is not None:
                    operand.node.type = target_type
                    for idx in operand.token_indices:
                        state.paths[idx] = materialize(state.path_refs[idx])
            elif operand.kind == "atom":
                node = CtxNode(target_type)
                for idx in operand.token_indices:
                    state.path_refs[idx].append(node)
                    state.paths[idx] = materialize(state.path_refs[idx])

        if tid in self.boundary_ids:
            state.path_refs[i] = current_path_nodes()

        elif tid in self.rbrace_ids:
            state.path_refs[i] = current_path_nodes()
            node = CtxNode(TYPE_UNK)
            state.frames.append(Frame(node=node, operands=[], token_indices=[]))

        elif tid in self.lbrace_ids:
            state.path_refs[i] = current_path_nodes()
            if len(state.frames) > 1:
                closed = state.frames.pop()
                closed.token_indices.append(i)
                state.frames[-1].operands.append(
                    Operand(
                        kind="group",
                        token_indices=closed.token_indices,
                        node=closed.node,
                        parent_nodes=current_path_nodes(),
                    )
                )

        elif tid in self.sup_ids or tid in self.sub_ids:
            state.path_refs[i] = current_path_nodes()
            op_type = TYPE_SUP if tid in self.sup_ids else TYPE_SUB
            if len(state.frames[-1].operands) > 0:
                resolve_operand(state.frames[-1].operands[-1], op_type)

        elif tid in self.frac_ids:
            state.path_refs[i] = current_path_nodes()
            if len(state.frames[-1].operands) >= 2:
                resolve_operand(state.frames[-1].operands[-1], TYPE_NUM)
                resolve_operand(state.frames[-1].operands[-2], TYPE_DEN)
            elif len(state.frames[-1].operands) == 1:
                resolve_operand(state.frames[-1].operands[-1], TYPE_NUM)

        else:
            state.path_refs[i] = current_path_nodes()
            state.frames[-1].operands.append(
                Operand(
                    kind="atom",
                    token_indices=[i],
                    node=None,
                    parent_nodes=current_path_nodes(),
                )
            )

        state.paths[i] = materialize(state.path_refs[i])
        for fr in state.frames[1:]:
            fr.token_indices.append(i)

        pi = state.paths[i]
        row_offset = i * state.max_len
        for j in range(i + 1):
            if state.tokens[j] == self.pad_id:
                continue
            pj = state.paths[j]
            state.rel_list[row_offset + j] = self._pair_to_rel_id(pi, pj)



    def materialize_state(self, state: R2LState, cur_len: int, device: torch.device) -> torch.Tensor:
        rel = torch.tensor(state.rel_list, dtype=torch.long).view(state.max_len, state.max_len)
        rel_sliced = rel[:cur_len, :cur_len]
        return rel_sliced.to(device, non_blocking=True)

    def clone_state(self, state: R2LState) -> R2LState:
        node_map = {}
        def get_copied_node(node):
            if node is None:
                return None
            nid = id(node)
            if nid not in node_map:
                node_map[nid] = CtxNode(node.type)
            return node_map[nid]

        new_frames = []
        for fr in state.frames:
            new_node = get_copied_node(fr.node)
            new_operands = []
            for op in fr.operands:
                new_op = Operand(
                    kind=op.kind,
                    token_indices=op.token_indices.copy(),
                    node=get_copied_node(op.node),
                    parent_nodes=[get_copied_node(n) for n in op.parent_nodes]
                )
                new_operands.append(new_op)
            new_frames.append(Frame(
                node=new_node,
                operands=new_operands,
                token_indices=fr.token_indices.copy()
            ))

        new_path_refs = []
        for ref in state.path_refs:
            new_path_refs.append([get_copied_node(n) for n in ref])

        return R2LState(
            tokens=state.tokens.copy(),
            frames=new_frames,
            path_refs=new_path_refs,
            paths=state.paths.copy(),
            rel_list=state.rel_list.copy(),
            max_len=state.max_len,
        )

    def _paths_for_seq_ids(self, seq_ids: torch.Tensor) -> List[List[Tuple[int, ...]]]:
        """
        Convert one token-id sequence -> paths for each step.
        Returns:
            List of lists of tuples, where out[i][j] is the path of token j at step i.
        """
        L = int(seq_ids.numel())
        tokens = seq_ids.tolist()

        frames = [Frame(node=None, operands=[], token_indices=[])]
        path_refs = [[] for _ in range(L)]
        paths = [None] * L

        def current_path_nodes():
            return [fr.node for fr in frames if fr.node is not None]

        def materialize(nodes):
            types = [n.type for n in nodes]
            return (TYPE_ROOT,) if len(types) == 0 else tuple(types)

        def resolve_operand(operand: Operand, target_type: int):
            if operand.kind == "group":
                if operand.node is not None:
                    operand.node.type = target_type
                    for idx in operand.token_indices:
                        paths[idx] = materialize(path_refs[idx])
            elif operand.kind == "atom":
                node = CtxNode(target_type)
                for idx in operand.token_indices:
                    path_refs[idx].append(node)
                    paths[idx] = materialize(path_refs[idx])

        all_step_paths = []

        for i in range(L):
            tid = tokens[i]
            if tid == self.pad_id:
                path_refs[i] = []
                paths[i] = (TYPE_ROOT,)
                step_paths = []
                for j in range(i + 1):
                    step_paths.append(paths[j])
                all_step_paths.append(step_paths)
                continue

            if tid in self.boundary_ids:
                path_refs[i] = current_path_nodes()

            elif tid in self.rbrace_ids:
                path_refs[i] = current_path_nodes()
                node = CtxNode(TYPE_UNK)
                frames.append(Frame(node=node, operands=[], token_indices=[]))

            elif tid in self.lbrace_ids:
                path_refs[i] = current_path_nodes()
                if len(frames) > 1:
                    closed = frames.pop()
                    closed.token_indices.append(i)
                    frames[-1].operands.append(
                        Operand(
                            kind="group",
                            token_indices=closed.token_indices,
                            node=closed.node,
                            parent_nodes=current_path_nodes(),
                        )
                    )

            elif tid in self.sup_ids or tid in self.sub_ids:
                path_refs[i] = current_path_nodes()
                op_type = TYPE_SUP if tid in self.sup_ids else TYPE_SUB
                if len(frames[-1].operands) > 0:
                    resolve_operand(frames[-1].operands[-1], op_type)

            elif tid in self.frac_ids:
                path_refs[i] = current_path_nodes()
                if len(frames[-1].operands) >= 2:
                    resolve_operand(frames[-1].operands[-1], TYPE_NUM)
                    resolve_operand(frames[-1].operands[-2], TYPE_DEN)
                elif len(frames[-1].operands) == 1:
                    resolve_operand(frames[-1].operands[-1], TYPE_NUM)

            else:
                path_refs[i] = current_path_nodes()
                frames[-1].operands.append(
                    Operand(
                        kind="atom",
                        token_indices=[i],
                        node=None,
                        parent_nodes=current_path_nodes(),
                    )
                )

            paths[i] = materialize(path_refs[i])
            for fr in frames[1:]:
                fr.token_indices.append(i)

            step_paths = []
            for j in range(i + 1):
                step_paths.append(paths[j])
            all_step_paths.append(step_paths)

        return all_step_paths

    def _build_one(self, seq_ids: torch.Tensor) -> torch.Tensor:
        L = int(seq_ids.numel())
        tokens = seq_ids.tolist()

        frames = [Frame(node=None, operands=[], token_indices=[])]
        path_refs = [[] for _ in range(L)]
        paths = [None] * L
        rel_list = [0] * (L * L)

        def current_path_nodes():
            return [fr.node for fr in frames if fr.node is not None]

        def materialize(nodes):
            types = [n.type for n in nodes]
            return (TYPE_ROOT,) if len(types) == 0 else tuple(types)

        def resolve_operand(operand: Operand, target_type: int):
            if operand.kind == "group":
                if operand.node is not None:
                    operand.node.type = target_type
                    for idx in operand.token_indices:
                        paths[idx] = materialize(path_refs[idx])
            elif operand.kind == "atom":
                node = CtxNode(target_type)
                for idx in operand.token_indices:
                    path_refs[idx].append(node)
                    paths[idx] = materialize(path_refs[idx])

        rel_cache = {}

        def pair_to_rel_id(pi, pj):
            pi_clean = () if pi == (TYPE_ROOT,) else pi
            pj_clean = () if pj == (TYPE_ROOT,) else pj

            min_l = min(len(pi_clean), len(pj_clean))
            lcp = 0
            while lcp < min_l and pi_clean[lcp] == pj_clean[lcp]:
                lcp += 1

            d = len(pi_clean) + len(pj_clean) - 2 * lcp
            db = min(max(d, 0), self.num_buckets - 1)

            ti = pi_clean[lcp] if lcp < len(pi_clean) else TYPE_ROOT
            tj = pj_clean[lcp] if lcp < len(pj_clean) else TYPE_ROOT

            if self.rel_set == "script":
                keep_i = (ti == TYPE_ROOT) or (ti == TYPE_SUP) or (ti == TYPE_SUB) or (ti == TYPE_UNK)
                keep_j = (tj == TYPE_ROOT) or (tj == TYPE_SUP) or (tj == TYPE_SUB) or (tj == TYPE_UNK)
                if not keep_i:
                    ti = TYPE_ROOT
                if not keep_j:
                    tj = TYPE_ROOT
            elif self.rel_set == "fraction":
                keep_i = (ti == TYPE_ROOT) or (ti == TYPE_NUM) or (ti == TYPE_DEN) or (ti == TYPE_UNK)
                keep_j = (tj == TYPE_ROOT) or (tj == TYPE_NUM) or (tj == TYPE_DEN) or (tj == TYPE_UNK)
                if not keep_i:
                    ti = TYPE_ROOT
                if not keep_j:
                    tj = TYPE_ROOT
            elif self.rel_set == "core":
                pass

            if self.mode == "dist_only":
                return db
            elif self.mode == "type_only":
                return ti * self.type_size + tj
            else:
                return db * (self.type_size * self.type_size) + ti * self.type_size + tj

        for i in range(L):
            tid = tokens[i]
            if tid == self.pad_id:
                path_refs[i] = []
                paths[i] = (TYPE_ROOT,)
                continue

            if tid in self.boundary_ids:
                path_refs[i] = current_path_nodes()

            elif tid in self.rbrace_ids:
                path_refs[i] = current_path_nodes()
                node = CtxNode(TYPE_UNK)
                frames.append(Frame(node=node, operands=[], token_indices=[]))

            elif tid in self.lbrace_ids:
                path_refs[i] = current_path_nodes()
                if len(frames) > 1:
                    closed = frames.pop()
                    closed.token_indices.append(i)
                    frames[-1].operands.append(
                        Operand(
                            kind="group",
                            token_indices=closed.token_indices,
                            node=closed.node,
                            parent_nodes=current_path_nodes(),
                        )
                    )

            elif tid in self.sup_ids or tid in self.sub_ids:
                path_refs[i] = current_path_nodes()
                op_type = TYPE_SUP if tid in self.sup_ids else TYPE_SUB
                if len(frames[-1].operands) > 0:
                    resolve_operand(frames[-1].operands[-1], op_type)

            elif tid in self.frac_ids:
                path_refs[i] = current_path_nodes()
                if len(frames[-1].operands) >= 2:
                    resolve_operand(frames[-1].operands[-1], TYPE_NUM)
                    resolve_operand(frames[-1].operands[-2], TYPE_DEN)
                elif len(frames[-1].operands) == 1:
                    resolve_operand(frames[-1].operands[-1], TYPE_NUM)

            else:
                path_refs[i] = current_path_nodes()
                frames[-1].operands.append(
                    Operand(
                        kind="atom",
                        token_indices=[i],
                        node=None,
                        parent_nodes=current_path_nodes(),
                    )
                )

            paths[i] = materialize(path_refs[i])
            for fr in frames[1:]:
                fr.token_indices.append(i)

            pi = paths[i]
            row_offset = i * L
            for j in range(i + 1):
                if tokens[j] == self.pad_id:
                    continue
                pj = paths[j]
                key = (pi, pj)
                rid = rel_cache.get(key)
                if rid is None:
                    rid = pair_to_rel_id(pi, pj)
                    rel_cache[key] = rid
                rel_list[row_offset + j] = rid

        return torch.tensor(rel_list, dtype=torch.long).view(L, L)

    def build(self, tgt_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Build relation ids for decoder self-attention in R2L direction.

        tgt_ids: (B, L) LongTensor on any device
        rel_ids: (B, L, L) LongTensor on tgt_ids.device
        """
        if tgt_ids.dim() != 2:
            raise ValueError(f"tgt_ids must be (B, L), got {tuple(tgt_ids.shape)}")

        device = tgt_ids.device
        B, L = tgt_ids.shape

        tgt_cpu = tgt_ids.detach().to("cpu")
        out = torch.zeros((B, L, L), dtype=torch.long)

        for b in range(B):
            out[b] = self._build_one(tgt_cpu[b])

        if device.type == "cpu":
            return out
        else:
            return out.to(device, non_blocking=True)


