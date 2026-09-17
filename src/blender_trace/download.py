"""
Download a single Blender tutorial video plus any available subtitles/captions.

Notes:
- Capped at 1080p by default: you need enough resolution to read the F9/header
  panel text, but not more than that -- keep files small.
- Tries manual captions first, falls back to auto-captions. If neither exists,
  transcribe.py will run local Whisper on the extracted audio instead.
- Only downloads one video at a time on purpose -- review each one before
  committing to a batch run over a whole playlist. Run discover.is_live(url)
  first if the URL wasn't just produced by a live discover call.
"""
import json
import subprocess
from pathlib import Path


def download(url: str, out_dir: Path, max_height: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # First, grab metadata only, so we know the video id / title up front and
    # can lay out a per-video folder before pulling the (large) video file.
    info_cmd = ["yt-dlp", "--dump-json", "--no-warnings", url]
    info = json.loads(subprocess.check_output(info_cmd, text=True))
    video_id = info["id"]
    video_dir = out_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    (video_dir / "info.json").write_text(json.dumps({
        "id": video_id,
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "url": url,
        "duration_s": info.get("duration"),
    }, indent=2))

    fmt = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-sub", "--write-auto-sub", "--sub-lang", "en.*",
        "--convert-subs", "srt",
        "-o", str(video_dir / "video.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)
    print(f"Downloaded to {video_dir}")
    return video_dir
