from blender_trace.segment import mentions_action_cue, segment_transcript


def test_mentions_action_cue_matches_hotkeys_and_tool_names():
    assert mentions_action_cue("press F to fill that in")
    assert mentions_action_cue("shift R to repeat the command")
    assert mentions_action_cue("let's add a bevel here")
    assert not mentions_action_cue("what's up guys, welcome back")


def test_segment_transcript_splits_on_action_cues_with_min_gap():
    transcript = [
        {"t_start": 0.0, "t_end": 2.0, "text": "what's up guys, welcome back"},
        {"t_start": 2.0, "t_end": 2.8, "text": "let's add a cube to start"},
        # "bevel" is only 0.8s after the split at t=2.0 -- too close to
        # split on again, so it should merge into the same segment.
        {"t_start": 2.8, "t_end": 4.0, "text": "and bevel this edge a bit"},
        {"t_start": 8.0, "t_end": 9.5, "text": "now let's extrude this face up"},
    ]

    segments = segment_transcript(transcript, min_segment_s=1.5)

    starts = [round(s["t_start"], 1) for s in segments]
    ends = [round(s["t_end"], 1) for s in segments]
    assert starts == [0.0, 2.0, 8.0]
    assert ends == [2.0, 8.0, 9.5]
    assert segments[1]["narration"] == (
        "let's add a cube to start and bevel this edge a bit"
    )


def test_segment_transcript_empty_input():
    assert segment_transcript([]) == []
