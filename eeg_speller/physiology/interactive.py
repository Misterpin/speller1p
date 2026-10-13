"""One P300 decoder, interchangeable EEG transports, no access to task text."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import shutil
import time
import numpy as np

from eeg_speller.core.distributions import normalize_logits
from eeg_speller.physiology.base import PhysiologyObservation
from eeg_speller.physiology.keyboard import KeyboardPhysiology
from eeg_speller.physiology.p300 import P300Features, CalibratedP300
from eeg_speller.physiology.sources import EDFSource, LSLSource


class P300Physiology:
    backend = "p300"
    input_mode = "eeg"

    def __init__(self, cfg, project):
        self.cfg = cfg
        self.features = P300Features(cfg)
        model_path = Path(cfg["model_path"])
        if not model_path.is_absolute():
            model_path = project / model_path
        self.model = CalibratedP300(model_path, self.features)
        source_cfg = dict(cfg["source"])
        if source_cfg["kind"] == "edf_replay":
            path = Path(source_cfg["path"])
            source_cfg["path"] = path if path.is_absolute() else project / path
            self.source = EDFSource(source_cfg, self.features.channels)
        elif source_cfg["kind"] == "lsl":
            self.source = LSLSource(source_cfg, self.features.channels)
        else:
            raise ValueError("EEG source.kind must be edf_replay or lsl")
        self.backend = "p300_" + self.source.kind
        self.model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.events = {}
        self.scores = {}
        self.rejected = set()
        self.sync_residuals = []
        self.sync_error = None
        self.acquisition_error = None
        self.archive_directory = None

    def has_epoch(self, epoch_id):
        return self.source.kind != "edf_replay" or epoch_id in self.source.markers

    def start_epoch(self, msg):
        self.msg = msg
        self.plan = [dict(x) for x in msg.stimuli]
        self.events, self.scores, self.rejected = {}, {}, set()
        self.sync_residuals, self.sync_error = [], None
        if self.source.kind == "edf_replay":
            markers = self.source.markers.get(msg.epoch_id)
            if markers is None or len(markers) != len(self.plan):
                raise ValueError("EDF epoch/flash count differs from the selected profile")
            start = markers[0]["source_time_s"]
            for flash, marker in zip(self.plan, markers):
                for key in ("id", "kind", "index", "repetition"):
                    if flash[key] != marker[key]:
                        raise ValueError("EDF stimulus order differs from the selected profile")
                recorded_ms = (marker["source_time_s"] - start) * 1000
                if abs(recorded_ms - flash["onset_ms"]) > 1:
                    raise ValueError("EDF stimulus timing differs from the selected profile")
                flash.update(source_time_s=marker["source_time_s"], onset_ms=recorded_ms)

    def presentation_plan(self):
        return self.plan

    def on_stimulus(self, event):
        if event.epoch_id != self.msg.epoch_id or event.flash_id != len(self.events):
            raise ValueError("EEG markers must arrive once, in order, for the current epoch")
        if event.uncertainty_ms > self.cfg.get("max_clock_uncertainty_ms", 20):
            raise ValueError("browser/server clock uncertainty exceeds the configured limit")
        if self.events and event.onset_ns <= self.events[event.flash_id - 1].onset_ns:
            raise ValueError("stimulus timestamps must increase")
        self.events[event.flash_id] = event
        if self.source.kind == "lsl":
            self.plan[event.flash_id]["source_time_s"] = self.source.event_time(event.onset_ns)
            self.source.publish_marker(event, self.plan[event.flash_id])
        if self.source.kind == "edf_replay":
            elapsed_ms = (event.onset_ns - self.events[0].onset_ns) / 1e6
            residual = elapsed_ms - self.plan[event.flash_id]["onset_ms"]
            self.sync_residuals.append(residual)
            if abs(residual) > self.cfg.get("max_replay_drift_ms", 75):
                self.sync_error = f"EDF/display timing drift {residual:.1f} ms exceeds limit"

    def on_keyboard_response(self, flash_id):
        raise ValueError("keyboard responses are disabled in EEG mode")

    def finish_presentation(self):
        pass

    def _read_available(self):
        for fid, event in self.events.items():
            if fid in self.scores or fid in self.rejected:
                continue
            onset = (self.plan[fid]["source_time_s"] if self.source.kind == "edf_replay"
                     else self.source.event_time(event.onset_ns))
            # Replay data becomes available when its post-stimulus interval has elapsed.
            if self.source.kind == "edf_replay" and time.monotonic_ns() < event.onset_ns + self.features.window[1] * 1e9:
                continue
            samples = self.source.window(onset, self.features.window)
            if samples is None:
                continue
            feature = self.features.extract(samples, self.source.fs)
            if feature is None:
                self.rejected.add(fid)
            else:
                self.scores[fid] = self.model.llr(feature)

    def _observation(self):
        logits = np.zeros(len(self.msg.repertoire.symbols))
        for fid, llr in self.scores.items():
            logits[self.plan[fid]["indices"]] += llr
        # Every row/column must have evidence; otherwise a missing channel/window
        # can spuriously win by avoiding negative evidence.
        required = {(x["kind"], x["index"]) for x in self.plan}
        covered = {(self.plan[i]["kind"], self.plan[i]["index"]) for i in self.scores}
        positive_kinds = {self.plan[i]["kind"] for i, llr in self.scores.items() if llr > 0}
        target_response_present = {x["kind"] for x in self.plan}.issubset(positive_kinds)
        metrics = {"usable": required == covered and target_response_present and not self.sync_error,
                   "target_response_present": target_response_present,
                   "evidence_flashes": len(self.scores), "rejected_flashes": len(self.rejected),
                   "positive_flashes": [i for i, s in self.scores.items() if s > 0],
                   "flash_log_likelihood_ratios": [{"flash_id": i, "llr": s} for i, s in sorted(self.scores.items())],
                   "synchronization": self.status()["synchronization"]}
        return PhysiologyObservation(self.msg.epoch_id, normalize_logits(logits), "likelihood",
                                    stimulus_events=[{**self.plan[i], **asdict(e)} for i, e in self.events.items()],
                                    module_metrics=metrics, t_module_ns=time.monotonic_ns())

    def preview_observation(self):
        try:
            self._read_available()
        except ValueError as exc:
            self.acquisition_error = str(exc)
            return None
        return self._observation() if self.scores else None

    def get_observation(self):
        if len(self.events) != len(self.plan):
            raise ValueError("EEG epoch is missing stimulus markers")
        if self.sync_error:
            raise ValueError(self.sync_error)
        if self.acquisition_error:
            raise ValueError(self.acquisition_error)
        deadline = time.monotonic() + self.cfg.get("data_timeout_s", 5)
        while time.monotonic() < deadline:
            self._read_available()
            if len(self.scores) + len(self.rejected) == len(self.plan):
                return self._observation()
            time.sleep(0.01)
        raise ValueError("timed out waiting for EEG after the last stimulus")

    def status(self):
        sync = {"method": "edf_annotation_to_browser_frame" if self.source.kind == "edf_replay" else "browser_to_monotonic_to_lsl",
                "onset_kind": "browser_frame_estimate", "markers_received": len(self.events),
                "max_clock_uncertainty_ms": max((e.uncertainty_ms for e in self.events.values()), default=0),
                "max_replay_drift_ms": max(map(abs, self.sync_residuals), default=0),
                "error": self.sync_error or self.acquisition_error}
        if self.events and self.source.kind == "edf_replay":
            sync["anchor"] = {"server_monotonic_ns": self.events[0].onset_ns,
                              "edf_time_s": self.plan[0]["source_time_s"]}
        return {**self.source.status(), "backend": self.backend, "input_mode": self.input_mode,
                "evidence_flashes": len(self.scores), "rejected_flashes": len(self.rejected),
                "model_sha256": self.model_hash, "synchronization": sync}

    def archive(self, directory):
        self.archive_directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.model.path, directory / "p300_model.json")
        if self.source.kind == "edf_replay":
            shutil.copy2(self.source.path, directory / "source.edf")

    def receive_feedback(self, fb):
        pass

    def receive_update(self, batch):
        pass

    def close(self):
        self.source.close()
        if self.source.kind == "lsl" and self.archive_directory:
            self.source.save(self.archive_directory / "lsl_eeg.npz")


def create_interactive(cfg, project):
    plugin = cfg.physiology.get("plugin", "keyboard_space")
    if plugin in {"keyboard_space", "simulated"}:
        return KeyboardPhysiology(cfg.gui)
    if plugin == "p300":
        return P300Physiology(cfg.physiology, project)
    raise ValueError(f"unsupported interactive physiology plugin: {plugin}")
