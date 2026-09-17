from blender_trace.manifest import build_manifest, overlapping_narration


def test_overlapping_narration_joins_covering_segments():
    transcript = [
        {"t_start": 0.0, "t_end": 3.0, "text": "first I'll add a cube"},
        {"t_start": 3.0, "t_end": 6.0, "text": "then bevel this edge"},
        {"t_start": 20.0, "t_end": 23.0, "text": "unrelated later narration"},
    ]
    assert overlapping_narration(transcript, 1.0, 5.0) == "first I'll add a cube then bevel this edge"


def test_build_manifest_pairs_consecutive_keyframes():
    transcript = [{"t_start": 0.0, "t_end": 5.0, "text": "add a bevel"}]
    keyframes = [
        {"t": 0.0, "frame_path": "frames/frame_0.00.png", "panel_path": "panels/panel_0.00.png"},
        {"t": 5.0, "frame_path": "frames/frame_5.00.png", "panel_path": "panels/panel_5.00.png"},
        {"t": 10.0, "frame_path": "frames/frame_10.00.png", "panel_path": "panels/panel_10.00.png"},
    ]

    manifest = build_manifest(transcript, keyframes)

    assert len(manifest) == 2
    assert manifest[0]["narration"] == "add a bevel"
    assert manifest[0]["frame_before"] == "frames/frame_0.00.png"
    assert manifest[0]["frame_after"] == "frames/frame_5.00.png"
    assert manifest[1]["narration"] == ""
