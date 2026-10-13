"""Contract offered to the physiology team."""
from dataclasses import dataclass
from typing import Literal, Protocol
import numpy as np
from scipy.special import logsumexp
from eeg_speller.core.distributions import Repertoire


@dataclass(frozen=True)
class StimulusTiming:
    flash_ms: int
    isi_ms: int
    pause_ms: int
    n_repetitions: int


@dataclass(frozen=True)
class EpochStart:
    epoch_id: int
    position: int
    repertoire: Repertoire
    paradigm: str
    timing: StimulusTiming
    stimuli: tuple[dict, ...] = ()


@dataclass(frozen=True)
class StimulusEvent:
    """Actual frame onset in the server's monotonic clock domain.

    No target symbol, task text or language prior crosses this boundary.
    Browser frame time is an estimate of display onset, not a hardware trigger.
    """
    epoch_id: int
    flash_id: int
    onset_ns: int
    uncertainty_ms: float = 0.0
    timing_method: str = "browser_animation_frame"


@dataclass(frozen=True)
class PhysiologyObservation:
    epoch_id: int
    log_p: np.ndarray
    semantics: Literal["likelihood", "posterior_with_prior"]
    per_repetition: list[np.ndarray] | None = None
    stimulus_events: list[dict] | None = None
    module_metrics: dict | None = None
    t_module_ns: int | None = None
    sim_flash_scores: list[list[float]] | None = None


class PhysiologyModule(Protocol):
    """Physiology Module; NEED-F-02/03/04; MA-03/04/07; Q-01/02/06."""
    def start_epoch(self, msg: EpochStart) -> None: ...
    def get_observation(self) -> PhysiologyObservation: ...
    def receive_feedback(self, fb) -> None: ...
    def receive_update(self, batch) -> None: ...


class InteractivePhysiologyModule(PhysiologyModule, Protocol):
    """Acquisition/decoding boundary used by the browser controller.

    Implementations own channels, sample clocks, preprocessing and classification.
    Only the keyboard implementation consumes keyboard responses.
    """
    input_mode: str
    backend: str
    def presentation_plan(self) -> list[dict]: ...
    def on_stimulus(self, event: StimulusEvent) -> None: ...
    def on_keyboard_response(self, flash_id: int) -> None: ...
    def finish_presentation(self) -> None: ...
    def preview_observation(self) -> PhysiologyObservation | None: ...
    def status(self) -> dict: ...
    def has_epoch(self, epoch_id: int) -> bool: ...
    def close(self) -> None: ...


def validate_observation(obs: PhysiologyObservation, epoch_id: int, n_symbols: int) -> None:
    """Reject malformed evidence or an embedded prior before fusion."""
    values = np.asarray(obs.log_p)
    if obs.epoch_id != epoch_id or values.shape != (n_symbols,) or not np.all(np.isfinite(values)):
        raise ValueError("physiology observation has a stale epoch or invalid symbol vector")
    if obs.semantics != "likelihood":
        raise ValueError("interactive physiology must return likelihood without language prior")
    if not np.isclose(logsumexp(values), 0.0, atol=1e-6):
        raise ValueError("physiology log likelihoods must be normalized")
