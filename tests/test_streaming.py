import numpy as np

from courtsider.asr.segmenter import segments_from_words
from courtsider.asr.streaming import SR, LocalAgreement, Word, drop_repeats


def _fake(script):
    """Transcriber returning a scripted hypothesis per call (words given as (text, start, end))."""
    calls = iter(script)

    def transcribe(audio, offset, repo, prompt):
        return [Word(t, s, e) for t, s, e in next(calls)]

    return transcribe


def test_local_agreement_commits_only_agreed_prefix_and_flushes_at_end():
    script = [
        [("and", 0.1, 0.3), ("he", 0.3, 0.4)],
        [("and", 0.1, 0.3), ("he", 0.3, 0.4), ("shoots", 0.5, 0.9)],
        [("and", 0.1, 0.3), ("he", 0.3, 0.4), ("scores", 0.5, 0.9), ("goal", 1.0, 1.4)],
        [("and", 0.1, 0.3), ("he", 0.3, 0.4), ("scores", 0.5, 0.9), ("goal", 1.0, 1.4)],
    ]
    la = LocalAgreement(transcribe=_fake(script))
    la.append(np.zeros(SR))
    committed = [w.text for w in la.process()[0]]
    assert committed == []  # nothing to agree with yet
    committed += [w.text for w in la.process()[0]]
    assert committed == ["and", "he"]  # "shoots" not confirmed
    committed += [w.text for w in la.process()[0]]
    assert committed == ["and", "he"]  # "scores" disagrees with "shoots"
    committed += [w.text for w in la.process(final=True)[0]]
    assert committed == ["and", "he", "scores", "goal"]


def test_buffer_is_trimmed_so_live_memory_stays_bounded():
    def transcribe(audio, offset, repo, prompt):  # one word per second of audio, like steady speech
        end = offset + len(audio) / SR
        return [Word(f"w{int(t)}", t, t + 0.3) for t in np.arange(np.ceil(offset), end - 0.5)]

    la = LocalAgreement(max_buffer=2.0, transcribe=transcribe)
    for _ in range(10):
        la.append(np.zeros(SR))
        la.process()
    assert len(la.audio) < 4 * SR


def test_drop_repeats_and_segmenter():
    words = [
        Word(w, i * 0.3, i * 0.3 + 0.2, emit=5.0) for i, w in enumerate("the secret the secret the secret goal".split())
    ]
    assert [w.text for w in drop_repeats(words)] == ["the", "secret", "the", "secret", "goal"]
    seg = segments_from_words(
        [Word("Goal!", 0, 0.5, 1.0), Word("What", 2.0, 2.2, 3.0), Word("a", 2.2, 2.3, 3.0)], "g", 1
    )
    assert seg.text.tolist() == ["Goal!", "What a"] and seg.available.tolist() == [1.0, 3.0]
