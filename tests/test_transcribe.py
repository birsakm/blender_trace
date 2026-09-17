from pathlib import Path

from blender_trace.transcribe import parse_srt

ROLLING_CAPTION_SRT = """\
1
00:00:00,000 --> 00:00:01,430

What's up guys? In today's video, I want

2
00:00:01,430 --> 00:00:01,440
What's up guys? In today's video, I want


3
00:00:01,440 --> 00:00:04,310
What's up guys? In today's video, I want
to show you how to make this shape right

4
00:00:04,310 --> 00:00:04,320
to show you how to make this shape right

"""


def test_parse_srt_dedupes_rolling_caption_window(tmp_path):
    srt_path = tmp_path / "video.en.srt"
    srt_path.write_text(ROLLING_CAPTION_SRT)

    segments = parse_srt(srt_path)

    texts = [s["text"] for s in segments]
    assert texts == [
        "What's up guys? In today's video, I want",
        "to show you how to make this shape right",
    ]


def test_parse_srt_keeps_independent_non_rolling_cues(tmp_path):
    srt_path = tmp_path / "video.en.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nfirst sentence\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nsecond unrelated sentence\n"
    )

    segments = parse_srt(srt_path)

    assert [s["text"] for s in segments] == ["first sentence", "second unrelated sentence"]
