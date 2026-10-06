"""Interactive stimulus sessions using the existing prior, fusion and detector blocks."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import threading
import time
import uuid

import numpy as np

from eeg_speller.core.clock import Clock
from eeg_speller.core.config import Config, load_config
from eeg_speller.core.distributions import Repertoire, SymbolDistribution, log_probs, normalize_logits
from eeg_speller.decoding.detector import SymbolDetector
from eeg_speller.epoch.fusion import Fusion
from eeg_speller.llm.fast import create_fast
from eeg_speller.llm.slow import CorrectionScheduler, create_slow, prompt_hash, validate_correction
from eeg_speller.physiology.base import PhysiologyObservation
from eeg_speller.recording.recorder import SessionRecorder
from eeg_speller.recording.storage import SessionStorage
from eeg_speller.text.primary import PrimaryText
from eeg_speller.text.secondary import SecondaryText


PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT / "configs"
CORPUS = PROJECT / "data" / "corpus_ru_fallback.txt"


def available_configs() -> list[dict]:
    result = []
    for path in sorted(CONFIGS.glob("gui_*.yaml")):
        try:
            cfg = load_config(path)
        except Exception:
            continue
        result.append({"name": path.name, "flash_ms": cfg.stimulus["flash_ms"],
                       "isi_ms": cfg.stimulus["isi_ms"], "repetitions": cfg.epoch["n_repetitions"],
                       "paradigm": cfg.stimulus["paradigm"],
                       "model": cfg.fast_llm["backend"] if cfg.mode["llm"] == "on" else "off"})
    return result


def gui_symbols() -> list[str]:
    return list(load_config(CONFIGS / "gui_slow.yaml").repertoire["symbols"])


class InteractiveSession:
    """Human space presses replace binary EEG responses to each visual flash."""

    def __init__(self, config_name: str, target: str, output_root: Path):
        names = {item["name"] for item in available_configs()}
        if config_name not in names:
            raise ValueError("choose a listed GUI config")
        original = load_config(CONFIGS / config_name)
        self.rep = Repertoire(tuple(original.repertoire["symbols"]), tuple(original.repertoire["layout"]))
        if len(target) > 120 or any(ch not in self.rep.symbols or ch == "⌫" for ch in target):
            raise ValueError("target must contain at most 120 repertoire characters")
        data = original.model_dump()
        data["participant_id"] = "GUI-" + uuid.uuid4().hex[:8]
        data["task_id"] = "GUI"
        data["run_mode"] = "live"
        self.cfg = Config.model_validate(data)
        self.target = target
        self.primary = PrimaryText()
        self.secondary = SecondaryText()
        self.detector = SymbolDetector(self.cfg.detector, self.rep)
        self.fusion = Fusion(self.cfg.fusion, self.cfg.mode["llm"])
        self.fast = create_fast(self.cfg.fast_llm, CORPUS) if self.cfg.mode["llm"] == "on" else None
        self.slow = create_slow(self.cfg.slow_llm, False, CORPUS) if self.cfg.mode["llm"] == "on" else None
        self.scheduler = CorrectionScheduler(self.cfg.slow_llm) if self.slow else None
        models = {"fast_llm": {"backend": self.cfg.fast_llm["backend"] if self.fast else "uniform",
                               "model_id": self.fast.model_id if self.fast else "uniform",
                               "version": self.fast.version if self.fast else "1",
                               "hash": self.fast.weights_hash if self.fast else "none"},
                  "slow_llm": {"backend": self.cfg.slow_llm["backend"] if self.slow else "disabled",
                               "model_id": self.slow.model_id if self.slow else "disabled",
                               "version": self.slow.version if self.slow else "1",
                               "hash": self.slow.weights_hash if self.slow else "none"},
                  "physiology": {"backend": "keyboard_space", "model_id": "keyboard_space", "version": "1",
                                  "hash": "none"}}
        self.storage = SessionStorage(output_root, self.cfg, self.rep, models)
        self.clock = Clock(live=True)
        self.rec = SessionRecorder(self.storage.directory, self.storage.session_id,
                                   self.cfg.participant_id, self.clock)
        self.rec.emit("Session Recorder", "session_started",
                      {"run_mode": "live", "llm_mode": self.cfg.mode["llm"],
                       "paradigm": self.cfg.stimulus["paradigm"], "seed": self.cfg.seed,
                       "config_hash": self.cfg.digest()})
        self.rec.emit("Session Recorder", "task_assigned", {"task_id": "GUI", "target_text": target})
        self.epoch_id = 0
        self.target_index = 0
        self.retry = 0
        self.finished = False
        self.last = None
        self.plan = self._make_plan()
        self.lock = threading.RLock()

    def _make_plan(self) -> list[dict]:
        rng = random.Random(self.cfg.child_seed("gui_flashes") + self.epoch_id)
        groups = []
        if self.cfg.stimulus["paradigm"] == "row_column":
            groups.extend({"kind": "row", "index": row, "indices": list(range(row * 6, row * 6 + 6))}
                          for row in range(6))
            groups.extend({"kind": "column", "index": col, "indices": list(range(col, 36, 6))}
                          for col in range(6))
        else:
            groups.extend({"kind": "symbol", "index": i, "indices": [i]} for i in range(36))
        plan = []
        for repetition in range(int(self.cfg.epoch["n_repetitions"])):
            order = groups[:]
            rng.shuffle(order)
            for group in order:
                plan.append({"id": len(plan), "repetition": repetition, **group})
        return plan

    def snapshot(self) -> dict:
        return {"session_id": self.storage.session_id, "target": self.target,
                "text": self.primary.value, "secondary": self.secondary.value,
                "epoch_id": self.epoch_id, "retry": self.retry,
                "target_index": self.target_index,
                "plan": self.plan if not self.finished else [], "finished": self.finished,
                "last": self.last, "symbols": list(self.rep.symbols),
                "timing": {"flash_ms": self.cfg.stimulus["flash_ms"],
                           "isi_ms": self.cfg.stimulus["isi_ms"],
                           "pause_ms": self.cfg.stimulus["pause_ms"]},
                "config": {"paradigm": self.cfg.stimulus["paradigm"],
                           "repetitions": self.cfg.epoch["n_repetitions"],
                           "model": self.fast.model_id if self.fast else "off"}}

    def _prior(self):
        if self.fast is None:
            return SymbolDistribution(self.rep, log_probs(np.ones(36)), "uniform", len(self.primary.value))
        return self.fast.predict(self.primary.value, self.rep)

    def _correct(self, epoch_id: int) -> None:
        req = self.scheduler.on_primary_update(self.primary.value) if self.scheduler else None
        if req is None or not req.words:
            return
        self.rec.emit("Post-factum Text Correction", "correction_requested",
                      {"correction_id": req.correction_id, "trigger": req.trigger,
                       "window_words": req.words, "window_start_word": req.start_word,
                       "left_context_len": len(req.left_context), "model_id": self.slow.model_id,
                       "model_version": self.slow.version, "model_hash": self.slow.weights_hash,
                       "prompt_hash": prompt_hash(PROJECT / self.cfg.slow_llm["prompt_file"])}, epoch_id)
        start = time.perf_counter_ns()
        try:
            result = self.slow.correct(req.left_context, req.words)
            accepted, reason = validate_correction(req.words, result, self.cfg.slow_llm["strict_word_count"])
        except Exception as exc:
            result, accepted, reason = req.words, False, type(exc).__name__
        self.rec.emit("Post-factum Text Correction", "correction_completed",
                      {"correction_id": req.correction_id, "words_in": req.words,
                       "words_out": result, "accepted": accepted, "reject_reason": reason,
                       "model_id": self.slow.model_id, "model_version": self.slow.version,
                       "model_hash": self.slow.weights_hash,
                       "wall_latency_ms": (time.perf_counter_ns() - start) / 1e6}, epoch_id)
        if accepted:
            self.secondary.replace_window(result, req.start_word)
            self.rec.emit("Secondary (Corrected) Text", "secondary_text_updated",
                          {"correction_id": req.correction_id, "text": self.secondary.value,
                           "frozen_upto_word": self.secondary.frozen_upto_word}, epoch_id)

    def submit(self, hits: list[bool]) -> dict:
        with self.lock:
            if self.finished:
                raise ValueError("session is finished")
            if len(hits) != len(self.plan) or any(type(hit) is not bool for hit in hits):
                raise ValueError("one boolean response per flash is required")
            eid = self.epoch_id
            position = len(self.primary.value)
            self.rec.emit("One Epoch", "epoch_started",
                          {"position": position,
                           "repertoire_hash": hashlib.sha256("".join(self.rep.symbols).encode()).hexdigest(),
                           "paradigm": self.cfg.stimulus["paradigm"],
                           "flash_ms": self.cfg.stimulus["flash_ms"],
                           "isi_ms": self.cfg.stimulus["isi_ms"],
                           "pause_ms": self.cfg.stimulus["pause_ms"],
                           "n_repetitions": self.cfg.epoch["n_repetitions"],
                           "retry_idx": self.retry}, eid)
            prior = self._prior()
            self.rec.emit("Source Model (Fast LLM)", "prior_computed",
                          {"model_id": self.fast.model_id if self.fast else "uniform",
                           "context_len": len(self.primary.value), "log_p": prior.log_p,
                           "wall_latency_ms": 0.0}, eid)
            p_hit = float(self.cfg.gui["response_hit"])
            p_false = float(self.cfg.gui["response_false_alarm"])
            if not 0 < p_false < p_hit < 1:
                raise ValueError("invalid keyboard response probabilities")
            scores = np.zeros(36)
            for flash, hit in zip(self.plan, hits):
                self.rec.emit("Presentation Method", "stimulus_event",
                              {"flash_id": flash["id"], "repetition": flash["repetition"],
                               "kind": flash["kind"], "index": flash["index"],
                               "indices": flash["indices"], "keyboard_hit": hit}, eid)
                for i in range(36):
                    p = p_hit if i in flash["indices"] else p_false
                    scores[i] += math.log(p if hit else 1 - p)
            obs = PhysiologyObservation(eid, normalize_logits(scores), "likelihood",
                                        module_metrics={"keyboard_hits": sum(hits)},
                                        stimulus_events=self.plan)
            self.rec.emit("EEG / EOG source", "physiology_observation",
                          {"log_p": obs.log_p, "semantics": "likelihood",
                           "source": "keyboard_space", "module_metrics": obs.module_metrics}, eid)
            post = self.fusion.combine(prior, obs)
            self.rec.emit("Statistical Inference", "posterior_computed",
                          {"fusion": post.fusion, "params": post.params, "log_p": post.log_p,
                           "wall_latency_ms": 0.0}, eid)
            decision = self.detector.decide(post, position, self.retry)
            uncertain = not any(hits) or decision.uncertain
            symbol = None if uncertain else decision.symbol
            top = np.argsort(-post.log_p)[:3]
            suggestions = [{"symbol": self.rep.symbols[int(i)],
                            "probability": round(float(np.exp(post.log_p[i])), 3)} for i in top]
            self.rec.emit("Primary Symbol Detector", "symbol_decided",
                          {"symbol": symbol, "confidence": decision.confidence,
                           "margin": decision.margin, "uncertain": uncertain,
                           "cold_start": decision.cold_start, "abstained": symbol is None}, eid)
            if symbol is not None:
                self.primary.apply(symbol)
                self.secondary.synchronize(self.primary.value)
                if self.target:
                    self.target_index = (max(0, self.target_index - 1) if symbol == "⌫"
                                         else min(len(self.target), self.target_index + 1))
                self.rec.emit("Primary Text", "primary_text_updated",
                              {"op": "delete" if symbol == "⌫" else "append",
                               "symbol": symbol, "position": position,
                               "text": self.primary.value}, eid)
                self._correct(eid)
                self.retry = 0
            else:
                self.retry += 1
            self.rec.emit("ext:ui", "ui_rendered",
                          {"primary_text": self.primary.value, "secondary_text": self.secondary.value,
                           "candidates_shown": suggestions,
                           "status": "selected" if symbol else "retry"}, eid)
            self.last = {"symbol": symbol, "confidence": round(decision.confidence, 3),
                         "suggestions": suggestions, "hits": sum(hits), "uncertain": uncertain}
            self.epoch_id += 1
            if self.target and self.target_index >= len(self.target):
                self.stop("target_finished")
            else:
                self.plan = self._make_plan()
            return self.snapshot()

    def stop(self, reason: str = "user_stopped") -> dict:
        with self.lock:
            if not self.finished:
                self.finished = True
                self.plan = []
                self.rec.emit("Session Recorder", "session_ended",
                              {"status": "completed", "reason": reason,
                               "n_epochs": self.epoch_id, "n_events": self.rec.seq + 1})
                count = self.rec.seq
                self.rec.close()
                self.storage.finish(self.target, self.primary.value, self.secondary.value,
                                    self.epoch_id, count)
            return self.snapshot()


class SessionManager:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.sessions: dict[str, InteractiveSession] = {}
        self.lock = threading.Lock()

    def start(self, config_name: str, target: str) -> dict:
        session = InteractiveSession(config_name, target, self.output_root)
        with self.lock:
            self.sessions[session.storage.session_id] = session
        return session.snapshot()

    def get(self, session_id: str) -> InteractiveSession:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown session")
        return session
