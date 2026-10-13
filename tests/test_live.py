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
