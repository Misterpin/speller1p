"""Optional actual liblsl loopback test; no EEG hardware is needed."""
import json
import threading
import time
import uuid
import numpy as np
import pytest


def test_real_lsl_samples_channel_aliases_units_and_marker_clock(tmp_path):
    pylsl = pytest.importorskip("pylsl")
    from eeg_speller.physiology.base import StimulusEvent
    from eeg_speller.physiology.sources import LSLSource
    stream_id = "speller-test-" + uuid.uuid4().hex
    info = pylsl.StreamInfo("SyntheticTestEEG", "EEG", 4, 256, pylsl.cf_float32, stream_id)
    channels = info.desc().append_child("channels")
    for name in ("device_Cz", "Pz", "P3", "P4"):
        channels.append_child("channel").append_child_value("label", name)
    outlet = pylsl.StreamOutlet(info)
    stop = threading.Event()

    def publish():
        anchor = pylsl.local_clock()
        i = 0
        while not stop.is_set():
            last = anchor + (i + 15) / 256
            if stop.wait(max(0, last - pylsl.local_clock())):
                break
            outlet.push_chunk([[0.01, 0.02, 0.03, 0.04]] * 16, timestamp=last)
            i += 16

    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    source = None
    try:
        source = LSLSource({"stream_value": stream_id, "units": "mV",
                            "channel_aliases": {"Cz": "device_Cz"}}, ["Cz", "Pz", "P3", "P4"])
        deadline = time.monotonic() + 6
        window = None
        while time.monotonic() < deadline:
            if source.last_sample:
                window = source.window(source.last_sample - 0.3, (-0.1, 0.1))
                if window is not None:
                    break
            time.sleep(0.03)
        assert window is not None and window.shape == (4, 51)
        assert np.allclose(window[:, 0], [1e-5, 2e-5, 3e-5, 4e-5])
        assert source.status()["connected"]
        streams = pylsl.resolve_byprop("source_id", source.marker_id, timeout=3)
        markers = pylsl.StreamInlet(streams[0])
        markers.open_stream(timeout=3)
        event = StimulusEvent(0, 0, time.monotonic_ns(), 0.3)
        source.publish_marker(event, {"kind": "row", "index": 2, "indices": list(range(12, 18)), "repetition": 0})
        sample, timestamp = markers.pull_sample(timeout=3)
        payload = json.loads(sample[0])
        assert payload["indices"] == list(range(12, 18)) and "target" not in payload
        assert abs(timestamp - source.event_time(event.onset_ns)) < 0.001
        source.save(tmp_path / "lsl.npz")
        assert np.load(tmp_path / "lsl.npz")["eeg_v"].shape[0] == 4
        markers.close_stream()
    finally:
        if source:
            source.close()
        stop.set()
        publisher.join(timeout=2)
