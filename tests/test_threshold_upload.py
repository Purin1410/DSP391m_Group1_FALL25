from pathlib import Path
from types import SimpleNamespace

import pytest

import train
from lit_comer import LitCoMER
from utils.remote_sync import analysis_epoch_from_name


class _Metric:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def numel(self):
        return 1

    def cpu(self):
        return self

    def item(self):
        return self.value


class _Module:
    def __init__(self, gate_open=False, activation_epoch=None):
        self._analysis_gate_open = gate_open
        self._analysis_gate_activation_epoch = activation_epoch
        self.analysis_logging_cfg = {
            "run_id": "run",
            "seeds": "7",
        }
        self.config = {"seed_everything": 7}


class _Trainer:
    def __init__(self, epoch=0, metric=None, sanity_checking=False):
        self.current_epoch = epoch
        self.callback_metrics = {"val_ExpRate": metric}
        self.sanity_checking = sanity_checking
        self.check_val_every_n_epoch = 2
        self.is_global_zero = True
        self.logger = SimpleNamespace(experiment=None)


def _callback(tmp_path):
    return train.RcloneUploadCallback(
        checkpoint_dir=tmp_path / "checkpoints",
        analysis_dir=tmp_path / "analysis_logs" / "run",
        remote_run_dir="remote:run",
        rclone_cfg={
            "enabled": True,
            "upload_checkpoints": True,
            "upload_analysis_logs": True,
            "every_n_epochs": 1,
        },
        wandb_cfg={"upload_artifacts": False},
        monitor="val_ExpRate",
        threshold=0.57,
    )


def test_more_validation_uses_inclusive_threshold_and_skips_sanity():
    callback = train.MoreValidationCallback(threshold=0.57)
    trainer = _Trainer(metric=_Metric(0.57))

    callback.on_validation_epoch_end(trainer, _Module())
    assert trainer.check_val_every_n_epoch == 1

    trainer = _Trainer(metric=_Metric(0.8), sanity_checking=True)
    callback.on_validation_epoch_end(trainer, _Module())
    assert trainer.check_val_every_n_epoch == 2


def test_analysis_gate_is_monotonic_and_checkpointed():
    module = LitCoMER.__new__(LitCoMER)
    module._analysis_gate_open = False
    module._analysis_gate_threshold = 0.57
    module._analysis_gate_activation_epoch = None
    trainer = SimpleNamespace(current_epoch=4, sanity_checking=False)
    module._analysis_trainer_or_none = lambda: trainer

    assert module.update_analysis_gate(0.5699) is False
    assert module._analysis_gate_open is False
    assert module.update_analysis_gate(0.57) is True
    assert module._analysis_gate_open is True
    assert module._analysis_gate_activation_epoch == 4
    assert module.update_analysis_gate(0.1) is False
    assert module._analysis_gate_open is True

    module._target_lrs = None
    module._warmup_total_steps = None
    module._warmup_finished = True
    module._plateau_scheduler = None
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    restored = LitCoMER.__new__(LitCoMER)
    restored._target_lrs = None
    restored._warmup_total_steps = None
    restored._warmup_finished = False
    restored._analysis_gate_open = False
    restored._analysis_gate_activation_epoch = None
    restored.on_load_checkpoint(checkpoint)
    assert restored._analysis_gate_open is True
    assert restored._analysis_gate_activation_epoch == 4


def test_upload_callback_does_nothing_before_gate(monkeypatch, tmp_path):
    callback = _callback(tmp_path)
    trainer = _Trainer(epoch=3, metric=_Metric(0.8))
    module = _Module(gate_open=False)
    called = []
    monkeypatch.setattr(callback, "_upload_on_all_ranks", lambda *args: called.append(True))

    callback.on_validation_end(trainer, module)
    callback.on_train_end(trainer, module)
    assert called == []


def test_upload_callback_runs_on_crossing_epoch(monkeypatch, tmp_path):
    callback = _callback(tmp_path)
    trainer = _Trainer(epoch=6, metric=_Metric(0.57))
    module = _Module(gate_open=True, activation_epoch=6)
    called = []
    monkeypatch.setattr(callback, "_upload_on_all_ranks", lambda *args: called.append(True))

    callback.on_validation_end(trainer, module)
    assert called == [True]
    assert callback._activation_epoch == 6
    assert callback._last_upload_epoch == 6


def test_upload_callback_state_round_trip(tmp_path):
    callback = _callback(tmp_path)
    callback._activation_epoch = 6
    callback._last_upload_epoch = 8
    state = callback.on_save_checkpoint(None, None, {})

    restored = _callback(tmp_path)
    restored.on_load_checkpoint(None, None, state)
    assert restored._activation_epoch == 6
    assert restored._last_upload_epoch == 8


def test_eligible_files_exclude_prethreshold_artifacts(tmp_path):
    callback = _callback(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    analysis_dir = tmp_path / "analysis_logs" / "run"
    checkpoint_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    for path in (
        checkpoint_dir / "run_4-val_ExpRate=0.5600.ckpt",
        checkpoint_dir / "run_6-val_ExpRate=0.5700.ckpt",
        analysis_dir / "run_7_0005.csv",
        analysis_dir / "run_7_0007.csv",
    ):
        path.write_text("data", encoding="utf-8")

    callback._activation_epoch = 6
    eligible = {path.name for path in callback._eligible_files("run")}
    assert eligible == {
        "run_6-val_ExpRate=0.5700.ckpt",
        "run_7_0007.csv",
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("run_7_0007.csv", 7),
        ("run_seed7_0042.jsonl", 42),
        ("run_7_0007_rank_0.csv", None),
        ("other_7_0007.csv", None),
    ],
)
def test_analysis_epoch_from_name(name, expected):
    assert analysis_epoch_from_name(name, "run") == expected
