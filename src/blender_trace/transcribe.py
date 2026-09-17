"""
Produce a normalized, timestamped transcript for a downloaded video.

If an .srt file from download.py is present, parses it directly (fast, free,
and reflects what the creator/YouTube already produced). Otherwise falls back
to local Whisper (faster-whisper) on the extracted audio track.

Output: <video_dir>/transcript.json
    [{"t_start": 12.4, "t_end": 15.1, "text": "now I'll add a bevel..."}, ...]
"""
import json
import re
import subprocess
from pathlib import Path


def parse_srt(srt_path: Path) -> list[dict]:
    raw = srt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", raw.strip())
    segments = []
    time_re = re.compile(
        r"(\d\d):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d\d):(\d\d):(\d\d)[,.](\d+)"
    )

    def to_sec(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    for block in blocks:
        lines = block.strip().splitlines()
        for line in lines:
            m = time_re.search(line)
            if m:
                t_start = to_sec(*m.groups()[0:4])
                t_end = to_sec(*m.groups()[4:8])
                text_lines = [
                    l for l in lines[lines.index(line) + 1:] if l.strip()
                ]
                text = " ".join(text_lines).strip()
                # collapse duplicate cue text some auto-caption formats produce
                if text:
                    segments.append({"t_start": t_start, "t_end": t_end, "text": text})
                break
    return segments


def whisper_fallback(video_dir: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    video_path = next(video_dir.glob("video.*"))
    audio_path = video_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(audio_path)],
        check=True,
        capture_output=True,
    )

    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), vad_filter=True)
    return [
        {"t_start": s.start, "t_end": s.end, "text": s.text.strip()}
        for s in segments
    ]


def main(video_dir: Path):
    srt_candidates = sorted(video_dir.glob("*.srt"))
    if srt_candidates:
        print(f"Using existing captions: {srt_candidates[0].name}")
        segments = parse_srt(srt_candidates[0])
    else:
        print("No captions found, running local Whisper fallback (slower)...")
        segments = whisper_fallback(video_dir)

    out_path = video_dir / "transcript.json"
    out_path.write_text(json.dumps(segments, indent=2))
    print(f"Wrote {len(segments)} segments to {out_path}")
