import torch

from models.comer import CoMER
from utils.utils import ce_loss, to_bi_tgt_out
from utils.vocab_info import VocabInfo

ID2TOK = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "x", 4: "^", 5: "{", 6: "2", 7: "}"}
VOCAB_SIZE = len(ID2TOK)
SOS, EOS, PAD = 1, 2, 0


class _FakeWords:
    def __init__(self, idx2word):
        self.idx2word = idx2word


def _vocab_info() -> VocabInfo:
    return VocabInfo(
        vocab_size=VOCAB_SIZE, sos_id=SOS, eos_id=EOS, pad_id=PAD, words=_FakeWords(ID2TOK)
    )


def _small_config():
    return {
        "model": {
            "d_model": 8,
            "growth_rate": 4,
            "num_layers": 1,
            "reduction": 0.5,
            "bottleneck": True,
            "use_dropout": False,
            "encoder_dropout": 0.0,
            "nhead": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 16,
            "decoder_dropout": 0.0,
            "dc": 4,
            "cross_coverage": False,
            "self_coverage": False,
            "use_tree_bias": True,
            "tree_bias_num_buckets": 8,
            "tree_bias_mode": "full",
            "tree_bias_layers": "all",
            "tree_bias_rel_set": "full",
        }
    }


def _make_batch(real_batch=2, img_size=64, seq_len=4, seed=0):
    torch.manual_seed(seed)
    imgs = torch.randn(real_batch, 1, img_size, img_size)
    mask = torch.zeros(real_batch, img_size, img_size, dtype=torch.bool)
    # Content tokens only (avoid pad/sos/eos ids), varying "sentence" per row.
    tokens = [
        [int(t) for t in torch.randint(3, VOCAB_SIZE, (seq_len,))] for _ in range(real_batch)
    ]
    tgt, out = to_bi_tgt_out(tokens, imgs.device, SOS, EOS, PAD)
    return imgs, mask, tgt, out


def test_training_step_smoke_loss_finite_and_bias_embedding_learns():
    torch.manual_seed(0)
    model = CoMER(_small_config(), vocab_info=_vocab_info())
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    bias_emb = model.decoder._tree_rel_bias.emb
    initial_weight = bias_emb.weight.detach().clone()

    imgs, mask, tgt, out = _make_batch()

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        out_hat = model(imgs, mask, tgt)
        loss = ce_loss(out_hat, out, ignore_idx=PAD)
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # The bias embedding must have actually moved away from its zero-init
    # after real gradient-descent steps through the full encoder+decoder+
    # loss pipeline -- not just in an isolated unit test.
    assert not torch.equal(bias_emb.weight.detach(), initial_weight)
    assert all(torch.isfinite(torch.tensor(losses)))


def test_training_step_smoke_matches_between_tree_bias_on_and_off_at_init():
    """Sanity/regression guard at the full-model level: with tree bias at
    its zero-init (untrained) state, the very first forward's loss must be
    identical whether tree bias is wired in or not."""
    imgs, mask, tgt, out = _make_batch(seed=1)

    torch.manual_seed(5)
    cfg_off = _small_config()
    cfg_off["model"]["use_tree_bias"] = False
    model_off = CoMER(cfg_off, vocab_info=_vocab_info())
    model_off.eval()
    with torch.no_grad():
        loss_off = ce_loss(model_off(imgs, mask, tgt), out, ignore_idx=PAD)

    torch.manual_seed(5)
    cfg_on = _small_config()
    model_on = CoMER(cfg_on, vocab_info=_vocab_info())
    model_on.eval()
    with torch.no_grad():
        loss_on = ce_loss(model_on(imgs, mask, tgt), out, ignore_idx=PAD)

    assert torch.allclose(loss_off, loss_on, atol=1e-6)
