"""Keyboard demonstration behind the same physiology boundary as EEG."""
import numpy as np
from eeg_speller.core.distributions import normalize_logits
from eeg_speller.physiology.base import PhysiologyObservation


class KeyboardPhysiology:
    backend = "keyboard_space"
    input_mode = "keyboard"

    def __init__(self, cfg):
        self.cfg = cfg
        self.msg = None

    def start_epoch(self, msg):
        self.msg = msg
        self.plan = [dict(x) for x in msg.stimuli]
        self.hits = [False] * len(self.plan)
        self.closed = -1
        self.events = {}

    def presentation_plan(self):
        return self.plan

    def on_stimulus(self, event):
        self.events[event.flash_id] = event
        self.closed = max(self.closed, event.flash_id - 1)

    def on_keyboard_response(self, flash_id):
        self.hits[flash_id] = True

    def finish_presentation(self):
        self.closed = len(self.plan) - 1

    def _observation(self, partial):
        scores = np.zeros(len(self.msg.repertoire.symbols))
        used = []
        for flash, hit in zip(self.plan, self.hits):
            if partial and flash["id"] > self.closed and not hit:
                continue
            p = np.full(len(scores), self.cfg["response_false_alarm"])
            p[flash["indices"]] = self.cfg["response_hit"]
            scores += np.log(p if hit else 1 - p)
            used.append(flash["id"])
        return PhysiologyObservation(self.msg.epoch_id, normalize_logits(scores), "likelihood",
                                    stimulus_events=self.plan,
                                    module_metrics={"usable": any(self.hits), "keyboard_hits": sum(self.hits),
                                                    "evidence_flashes": len(used),
                                                    "positive_flashes": [i for i, h in enumerate(self.hits) if h]})

    def preview_observation(self):
        return self._observation(True)

    def get_observation(self):
        return self._observation(False)

    def status(self):
        return {"backend": self.backend, "input_mode": self.input_mode, "connected": True,
                "label": "Пробел вместо ЭЭГ"}

    def has_epoch(self, epoch_id):
        return True

    def receive_feedback(self, fb):
        pass

    def receive_update(self, batch):
        pass

    def close(self):
        pass
