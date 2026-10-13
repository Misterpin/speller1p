"""Independent waveform decoding, channel mapping and timing failure cases."""
from dataclasses import replace
import time
import numpy as np
import pytest

from eeg_speller.core.config import load_config
from eeg_speller.core.distributions import Repertoire
from eeg_speller.physiology.base import EpochStart, StimulusTiming, StimulusEvent
from eeg_speller.physiology.edf_demo import generate_edf, calibrate
from eeg_speller.physiology.interactive import P300Physiology
from eeg_speller.physiology.sources import EDFSource
from eeg_speller.physiology.stimuli import make_plan
from eeg_speller.web.synchronization import BrowserClock


@pytest.fixture
def recording(tmp_path):
    cfg = load_config("configs/gui_edf.yaml")
    labels = generate_edf(cfg, "ажмтчя", tmp_path / "calibration.edf", signal_seed=521)
    calibrate(cfg, tmp_path / "calibration.edf", labels, tmp_path / "model.json")
    cfg.physiology["model_path"] = str(tmp_path / "model.json")
    cfg.physiology["source"]["path"] = str(tmp_path / "test.edf")
    return cfg, tmp_path


def start_epoch(module, cfg, eid):
    rep = Repertoire(tuple(cfg.repertoire["symbols"]), (6, 6))
    timing = StimulusTiming(**{k: cfg.stimulus[k] for k in ("flash_ms", "isi_ms", "pause_ms")},
                           n_repetitions=cfg.epoch["n_repetitions"])
    module.start_epoch(EpochStart(eid, eid, rep, cfg.stimulus["paradigm"], timing, tuple(make_plan(cfg, eid))))


def feed_recorded_frames(module, cfg, eid):
    start_epoch(module, cfg, eid)
    plan = module.presentation_plan()
    # Past monotonic frames make the complete EEG window available immediately.
    anchor = time.monotonic_ns() - round((plan[-1]["onset_ms"] + 2000) * 1e6)
    for flash in plan:
        module.on_stimulus(StimulusEvent(eid, flash["id"], anchor + round(flash["onset_ms"] * 1e6), 0.2))


def test_arbitrary_text_decoded_from_independent_edf_waveform(recording):
    cfg, directory = recording
    rng = np.random.default_rng(8402)
    text = "".join(rng.choice(cfg.repertoire["symbols"][:-1], 5))
    generate_edf(cfg, text, directory / "test.edf", signal_seed=9238)
    module = P300Physiology(cfg.physiology, directory)
    decoded = []
    for eid in range(len(text)):
        feed_recorded_frames(module, cfg, eid)
        obs = module.get_observation()
        decoded.append(cfg.repertoire["symbols"][int(np.argmax(obs.log_p))])
        assert obs.semantics == "likelihood" and obs.module_metrics["usable"]
        assert obs.module_metrics["synchronization"]["max_replay_drift_ms"] < 0.001
    assert "".join(decoded) == text
    assert not module.has_epoch(len(text))


def test_removing_p300_changes_decoding_to_uniform(recording):
    cfg, directory = recording
    generate_edf(cfg, "п", directory / "test.edf", signal_seed=18, amplitude_uv=0)
    module = P300Physiology(cfg.physiology, directory)
    feed_recorded_frames(module, cfg, 0)
    obs = module.get_observation()
    assert np.max(np.exp(obs.log_p)) < 0.1
    assert not obs.module_metrics["usable"] and not obs.module_metrics["target_response_present"]


def test_missing_duplicate_and_desynchronized_markers_are_rejected(recording):
    cfg, directory = recording
    generate_edf(cfg, "ю", directory / "test.edf", signal_seed=49)
    module = P300Physiology(cfg.physiology, directory)
    start_epoch(module, cfg, 0)
    with pytest.raises(ValueError, match="missing stimulus"):
        module.get_observation()
    anchor = time.monotonic_ns() - 30_000_000_000
    event = StimulusEvent(0, 0, anchor, 0.1)
    module.on_stimulus(event)
    with pytest.raises(ValueError, match="once, in order"):
        module.on_stimulus(event)
    for flash in module.plan[1:]:
        module.on_stimulus(replace(event, flash_id=flash["id"],
                                  onset_ns=anchor + round((flash["onset_ms"] + 200) * 1e6)))
    with pytest.raises(ValueError, match="timing drift"):
        module.get_observation()


def test_device_channel_aliases_and_unknown_channels(recording):
    cfg, directory = recording
    generate_edf(cfg, "а", directory / "test.edf", signal_seed=2)
    source = EDFSource({"path": directory / "test.edf", "channel_aliases": {"posterior": "Pz"}}, ["posterior"])
    assert source.data.shape[0] == 1 and source.fs == 256
    with pytest.raises(ValueError, match="channels missing"):
        EDFSource({"path": directory / "test.edf"}, ["missing"])


def test_clock_mapping_rejects_stale_and_nonfinite_markers():
    now_ms = time.monotonic_ns() / 1e6
    clock = BrowserClock.create(now_ms - 400, 0.5)
    assert abs(clock.map(400) / 1e6 - now_ms) < 1
    with pytest.raises(ValueError, match="stale"):
        clock.map(-3000)
    with pytest.raises(ValueError, match="finite"):
        BrowserClock.create(float("nan"), 0.2)


def test_model_requires_matching_preprocessing(recording):
    cfg, directory = recording
    generate_edf(cfg, "а", directory / "test.edf", signal_seed=2)
    cfg.physiology["response_s"] = [0.4, 0.6]
    with pytest.raises(ValueError, match="recalibrate"):
        P300Physiology(cfg.physiology, directory)


def test_contract_rejects_wrong_epoch_shape_and_embedded_prior():
    from eeg_speller.physiology.base import PhysiologyObservation, validate_observation
    valid = PhysiologyObservation(3, np.full(36, -np.log(36)), "likelihood")
    validate_observation(valid, 3, 36)
    for invalid in (replace(valid, epoch_id=2), replace(valid, log_p=np.zeros(6)),
                    replace(valid, semantics="posterior_with_prior"), replace(valid, log_p=np.zeros(36))):
        with pytest.raises(ValueError):
            validate_observation(invalid, 3, 36)


def test_language_prior_cannot_type_without_physiological_response(recording, monkeypatch):
    from eeg_speller.core.distributions import SymbolDistribution
    from eeg_speller.epoch.fusion import Fusion
    from eeg_speller.web.engine import InteractiveSession
    cfg, directory = recording
    generate_edf(cfg, "п", directory / "test.edf", signal_seed=18, amplitude_uv=0)
    monkeypatch.setattr("eeg_speller.web.engine.create_interactive",
                        lambda unused_cfg, project: P300Physiology(cfg.physiology, project))
    session = InteractiveSession("gui_edf.yaml", "", directory / "sessions",
                                 overrides={"detector.mode": "label"})
    # Give the actual fusion/detector a 99.9% prior. EEG must still authorize input.
    probs = np.full(36, 0.001 / 35)
    probs[0] = 0.999
    session.pending["prior"] = SymbolDistribution(session.rep, np.log(probs), "test_prior", 0)
    session.fusion = Fusion(session.cfg.fusion, "on")
    feed_recorded_frames(session.physiology, cfg, 0)
    state = session.submit(epoch_id=0)
    assert state["text"] == "" and state["last"]["symbol"] is None
    assert state["last"]["confidence"] > 0.99
