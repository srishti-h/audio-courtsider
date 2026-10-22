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
    assert s["false_alarms"] == {"text": 1, "audio": 0, "crowd": 0} and s["audio_minutes"] == 10.0


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


def test_recorder_restarts_a_stalled_stream(tmp_path, monkeypatch):
    import subprocess
    import threading

    from courtsider.live import recorder

    class FakeFeed:
        def snapshot(self, mid):
            return Snapshot(mid, "H", "A", (0, 0), "1'", "in", {})

    class FakePipe:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(recorder, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(recorder, "EspnFeed", FakeFeed)
    monkeypatch.setattr(recorder, "LivePipeline", FakePipe)
    monkeypatch.setattr(recorder, "STALL_S", 1.0)
    # a "stream" that connects but never sends a byte, like ffmpeg retrying a dead CDN mount
    monkeypatch.setattr(
        recorder.Session, "_ffmpeg", lambda self: subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)
    )
    s = recorder.Session("ser", [dict(id="1")])
    stop = threading.Event()
    threading.Timer(4.0, stop.set).start()
    s.run(stop)
    assert next(tmp_path.glob("*_radio-ser.jsonl")).read_text().count("stream_restart") >= 1


def test_mel_log_tiles_the_audio_clock(tmp_path, monkeypatch):
    import json

    import numpy as np

    from courtsider.live import recorder

    monkeypatch.setattr(recorder, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(recorder.MelLog, "BLOCK", 2 * 16000)
    log = recorder.MelLog("s")
    rng = np.random.default_rng(0)
    events = []
    for _ in range(10):  # 5 s of audio in 0.5 s chunks -> two complete 2 s files
        events += log.feed(rng.normal(size=8000).astype(np.float32))
    assert [e["audio_start"] for e in events] == [0.0, 2.0]
    assert all(e["frames"] == 100 for e in events)
    assert np.load(tmp_path / "s_mel" / "0001.npy").shape == (100, 64)
    json.dumps(events)


def test_radio_delay_estimate_lines_up_goal_calls():
    import numpy as np
    import pandas as pd

    from courtsider.data import radio

    goals = np.array([100.0, 500.0, 900.0])
    calls = goals + 30  # the radio is 30 s behind ESPN's clock
    segs = pd.DataFrame(dict(audio_start=np.r_[calls, 300.0], text=["GOOOAL!", "goal by Pedri", "he scores", "a pass"]))
    hits = radio.delay_hits(segs, lambda d: goals + d)
    assert abs(radio.best_delay(hits, len(goals)) - 30) <= 2
    assert radio.best_delay(hits, 1) == radio.DEFAULT_DELAY  # too few goals to trust


def test_espn_names_map_to_football_data_teams():
    from courtsider.pricing.current import _match

    teams = ["Ath Madrid", "Ath Bilbao", "Real Madrid", "Man United", "Man City", "Nott'm Forest", "Sociedad", "Betis"]
    assert _match("Atlético Madrid", teams) == "Ath Madrid"
    assert _match("Athletic Club", teams) == "Ath Bilbao"
    assert _match("Manchester United", teams) == "Man United"
    assert _match("Nottingham Forest", teams) == "Nott'm Forest"
    assert _match("Real Betis", teams) == "Betis"
    assert _match("Galatasaray", teams) is None


def test_lead_report_anchors_on_the_real_goal_time(monkeypatch):
    from courtsider.live import report

    monkeypatch.setattr(report, "goal_walls", lambda mid: [1000.0])
    events = [
        dict(kind="session", home="H", away="A"),
        dict(kind="detection", source="text", event="goal", wall=950.0, text="goal elsewhere"),  # other match
        dict(kind="detection", source="text", event="goal", wall=1004.0, text="GOOOL de Lamine Yamal"),
        dict(kind="feed", event="goal", wall=1060.0, clock="12'", team="home", match_id="1", home="H", away="A"),
    ]
    g = report.summarize(events)["goals"][0]
    assert g["text_lead_s"] == 56.0 and g["text_after_goal_s"] == 4.0 and g["match"] == "H vs A"


def test_podcast_alignment_finds_where_kickoff_falls():
    import numpy as np
    import pandas as pd

    from courtsider.data import podcasts

    ev = pd.DataFrame(dict(event=["goal", "goal", "goal"], t=[600.0, 1500.0, 2400.0]))  # match-clock seconds
    segs = pd.DataFrame(
        dict(audio_start=np.r_[ev.t + 312, 100.0], text=["GOOOL!", "goal by Raphinha", "he scores", "a pass"])
    )
    off, matched, n = podcasts.align(segs, ev)  # the episode started 312 s before kick-off
    assert abs(off - 312) <= 3 and matched == 3 and n == 3
    assert podcasts.align(segs.iloc[3:], ev)[0] is None  # no calls: not confident, skipped


def test_best_delay_never_lands_between_two_plateaus():
    import numpy as np

    from courtsider.data import radio

    delays = np.arange(0.0, 100.0)
    hits = np.zeros(100, int)
    hits[10:30] = 2  # wide plateau
    hits[80:85] = 2  # narrow one far away; the old median of both fell at ~27, the new answer is 20
    assert radio.best_delay(hits, 3, delays) == 20.0
