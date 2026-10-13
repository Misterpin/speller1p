"""Shared waveform processing and calibrated flash likelihoods for every EEG source."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import butter, sosfiltfilt


class P300Features:
    def __init__(self, cfg):
        self.channels = list(cfg["channels"])
        self.baseline = tuple(cfg.get("baseline_s", [-0.2, 0.0]))
        self.response = tuple(cfg.get("response_s", [0.25, 0.5]))
        self.window = tuple(cfg.get("window_s", [-0.2, 0.8]))
        self.band = tuple(cfg.get("bandpass_hz", [0.5, 15.0]))
        self.artifact_uv = float(cfg.get("artifact_uv", 150.0))
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("physiology.channels must be a nonempty list of distinct names")
        if not self.window[0] <= self.baseline[0] < self.baseline[1] <= 0:
            raise ValueError("baseline must lie inside the pre-stimulus window")
        if not 0 <= self.response[0] < self.response[1] <= self.window[1]:
            raise ValueError("response must lie inside the post-stimulus window")

    def signature(self):
        return {"channels": self.channels, "baseline_s": self.baseline, "response_s": self.response,
                "window_s": self.window, "bandpass_hz": self.band, "artifact_uv": self.artifact_uv}

    def extract(self, samples_v, fs):
        x = np.asarray(samples_v, dtype=float) * 1e6
        if x.ndim != 2 or x.shape[0] != len(self.channels) or not np.all(np.isfinite(x)):
            raise ValueError("invalid EEG window")
        if np.max(np.ptp(x, axis=1)) > self.artifact_uv:
            return None
        if not 0 < self.band[0] < self.band[1] < fs / 2:
            raise ValueError("bandpass must be inside the EEG Nyquist frequency")
        # The identical epoch filter is used for calibration, EDF and online LSL.
        # Online decoding consequently waits for the complete post-stimulus window.
        filtered = sosfiltfilt(butter(2, self.band, btype="bandpass", fs=fs, output="sos"), x, axis=1)
        def mean_between(interval):
            a = round((interval[0] - self.window[0]) * fs)
            b = round((interval[1] - self.window[0]) * fs)
            return filtered[:, a:b].mean(axis=1)
        return mean_between(self.response) - mean_between(self.baseline)


class CalibratedP300:
    """Diagonal Gaussian LDA; output is log P(EEG|target)/P(EEG|non-target).

    Equal class priors are used; no task text or language prior is accepted.
    This intentionally small baseline needs calibration on each actual setup.
    """
    def __init__(self, path, features):
        self.path = Path(path)
        model = json.loads(self.path.read_text(encoding="utf-8"))
        expected = json.loads(json.dumps(features.signature()))
        if model.get("schema") != "speller-p300@1" or model.get("features") != expected:
            raise ValueError("P300 model does not match channels/preprocessing; recalibrate")
        self.weights = np.asarray(model["weights"], dtype=float)
        self.bias = float(model["bias"])
        if self.weights.shape != (len(features.channels),) or not np.all(np.isfinite(self.weights)):
            raise ValueError("invalid P300 model weights")

    def llr(self, x):
        return float(np.clip(x @ self.weights + self.bias, -20, 20))

    @staticmethod
    def fit(x, labels, features, output):
        x, labels = np.asarray(x), np.asarray(labels, dtype=bool)
        if min(labels.sum(), (~labels).sum()) < 4 or not np.all(np.isfinite(x)):
            raise ValueError("calibration requires at least four usable flashes of each class")
        m1, m0 = x[labels].mean(axis=0), x[~labels].mean(axis=0)
        residual = x - np.where(labels[:, None], m1, m0)
        variance = np.maximum(np.mean(residual ** 2, axis=0), 0.1)
        weights = (m1 - m0) / variance
        bias = float(-0.5 * np.sum((m1 ** 2 - m0 ** 2) / variance))
        model = {"schema": "speller-p300@1", "features": features.signature(),
                 "weights": weights.tolist(), "bias": bias,
                 "calibration": {"target_flashes": int(labels.sum()),
                                 "non_target_flashes": int((~labels).sum())}}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(model, indent=2), encoding="utf-8")
        return model
