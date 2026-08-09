from pathlib import Path

import yaml

from lit_comer import LitCoMER
from utils.vocab_info import VocabInfo

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "crohme_config.yaml"

ID2TOK = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "x", 4: "^", 5: "{", 6: "2", 7: "}"}


class _FakeWords:
    def __init__(self, idx2word):
        self.idx2word = idx2word


def _fake_vocab_info() -> VocabInfo:
    return VocabInfo(
        vocab_size=len(ID2TOK),
        sos_id=1,
        eos_id=2,
        pad_id=0,
        words=_FakeWords(ID2TOK),
    )


def test_config_has_tree_bias_block():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mcfg = config["model"]
    assert mcfg["use_tree_bias"] is True
    assert mcfg["tree_bias_num_buckets"] == 16
    assert mcfg["tree_bias_mode"] == "full"
    assert mcfg["tree_bias_layers"] == "all"
    assert mcfg["tree_bias_rel_set"] == "core"


def test_lit_comer_instantiates_end_to_end_from_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = LitCoMER(config=config, vocab_info=_fake_vocab_info())

    decoder = model.comer_model.decoder
    assert decoder.use_tree_bias is True
    assert decoder._tree_builder is not None
    assert decoder._tree_rel_bias is not None
    assert decoder._tree_rel_bias.num_heads == config["model"]["nhead"]


def test_lit_comer_instantiates_with_tree_bias_disabled_via_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["model"]["use_tree_bias"] = False

    model = LitCoMER(config=config, vocab_info=_fake_vocab_info())

    decoder = model.comer_model.decoder
    assert decoder.use_tree_bias is False
    assert decoder._tree_builder is None
    assert decoder._tree_rel_bias is None
