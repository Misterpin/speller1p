"""Browser performance clock -> local server monotonic clock, with measured uncertainty."""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class BrowserClock:
    offset_ms: float
    uncertainty_ms: float

    @classmethod
    def create(cls, offset_ms, uncertainty_ms):
        values = (offset_ms, uncertainty_ms)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError("finite clock offset and uncertainty required")
        if not 0 <= uncertainty_ms <= 1000:
            raise ValueError("invalid clock uncertainty")
        return cls(float(offset_ms), float(uncertainty_ms))

    def map(self, client_ms):
        if isinstance(client_ms, bool) or not isinstance(client_ms, (int, float)) or not math.isfinite(client_ms):
            raise ValueError("finite browser frame timestamp required")
        onset = round((client_ms + self.offset_ms) * 1e6)
        arrival = time.monotonic_ns()
        if onset > arrival + round(self.uncertainty_ms * 1e6) + 1_000_000 or arrival - onset > 2_000_000_000:
            raise ValueError("browser marker is stale or its clock mapping is invalid")
        return onset
