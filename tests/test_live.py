from courtsider.asr.streaming import Word
from courtsider.live.feeds import Snapshot, diff, roster_vocabulary
from courtsider.live.pipeline import Segmenter
from courtsider.live.report import summarize


def test_feed_diff_emits_goals_and_red_cards():
    a = Snapshot("1", "H", "A", (0, 0), "10'", "in")
    b = Snapshot("1", "H", "A", (1, 1), "12'", "in", red_cards=(0, 1))
    ev = diff(a, b, wall=100.0)
    assert [(e["event"], e["team"]) for e in ev] == [("goal", "home"), ("goal", "away"), ("red_card", "away")]
    assert diff(None, b) == [] and diff(b, b) == []


def test_roster_vocabulary_surnames_first_and_deduplicated():
    v = roster_vocabulary({"home": ["Lamine Yamal", "Pedri"], "away": ["Pedri"]})
    assert v == ["Yamal", "Lamine Yamal", "Pedri"]


def test_incremental_segmenter_closes_on_pause_and_punctuation():
    seg = Segmenter(pause=0.6)
    out = seg.add([Word("He", 0, 0.2), Word("shoots", 0.3, 0.6), Word("Goal!", 2.0, 2.4), Word("What", 2.5, 2.7)])
    assert [" ".join(w.text for w in s) for s in out] == ["He shoots", "Goal!"]
    assert [w.text for w in seg.cur] == ["What"]


def test_lead_time_report_matches_detections_before_the_feed():
    events = [
        dict(kind="session", home="H", away="A"),
        dict(kind="detection", source="crowd", event="roar", wall=100.0),
        dict(kind="detection", source="text", event="goal", wall=104.0),
        dict(kind="feed", event="goal", team="home", clock="23'", wall=109.5),
        dict(kind="detection", source="text", event="goal", wall=500.0),  # false alarm, no feed goal follows
        dict(kind="status", audio_seconds=600),
    ]
    s = summarize(events)
    assert s["goals"] == [dict(clock="23'", team="home", text_lead_s=5.5, crowd_lead_s=9.5)]
    assert s["false_alarms"] == {"text": 1, "crowd": 0} and s["audio_minutes"] == 10.0


def test_fast_text_detections_count_as_text():
    events = [
        dict(kind="session", home="H", away="A"),
        dict(kind="detection", source="text-fast", event="goal", wall=60.8),
        dict(kind="feed", event="goal", team="home", clock="27'", wall=76.5),
    ]
    assert summarize(events)["goals"][0]["text_lead_s"] == 15.7


def test_recorder_reconnects_when_the_stream_drops(tmp_path, monkeypatch):
    import subprocess
    import threading
    import time as _time

    from courtsider.live import recorder

    wav = tmp_path / "s.wav"  # a 2 s "stream" that ends, so the recorder has to reconnect
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=2", "-ar", "16000", str(wav)], check=True
    )
    monkeypatch.setitem(recorder.STATIONS, "test", str(wav))
    monkeypatch.setattr(recorder, "LIVE_DIR", tmp_path)

    class FakeFeed:
        def snapshot(self, mid):
            return Snapshot(mid, "H", "A", (0, 0), "1'", "in", {"home": ["Ana Ruiz"], "away": ["Bo Lee"]})

    class FakePipe:
        seconds = 0.0

        def __init__(self, *a, **k):
            pass

        def feed(self, chunk):
            FakePipe.seconds += len(chunk) / 16000

        def tick(self):
            return [dict(kind="status", wall=_time.time(), audio_seconds=FakePipe.seconds)]

    monkeypatch.setattr(recorder, "EspnFeed", FakeFeed)
    monkeypatch.setattr(recorder, "LivePipeline", FakePipe)
    s = recorder.Session("test", [dict(id="1")])
    stop = threading.Event()
    threading.Timer(7.0, stop.set).start()
    s.run(stop)
    log = (next(tmp_path.glob("*_radio-test.jsonl"))).read_text()
    assert log.count("stream_restart") >= 2 and FakePipe.seconds >= 4.0
