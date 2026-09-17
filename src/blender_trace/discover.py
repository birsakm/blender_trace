"""
Live discovery of candidate tutorial videos via yt-dlp.

Hand-picked video links (found through search or chat) go stale constantly --
taken down, made private, region-locked. yt-dlp talks to YouTube directly, so
querying a channel's /videos page or running a live search tells you
immediately what currently exists, instead of trusting a stale link. Use this
module (or `blender-trace discover-*`) to build candidate lists; never commit
to a single hand-picked URL without running `is_live` on it first.
"""
import json
import subprocess


def _yt_dlp_flat_playlist(url: str, limit: int | None = None) -> list[dict]:
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", "--no-warnings"]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {url}: {result.stderr.strip()}")

    videos = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        info = json.loads(line)
        video_id = info.get("id")
        videos.append({
            "id": video_id,
            "title": info.get("title"),
            "url": info.get("url") or f"https://www.youtube.com/watch?v={video_id}",
            "channel": info.get("playlist_channel") or info.get("channel") or info.get("uploader"),
            "duration_s": info.get("duration"),
        })
    return videos


def list_channel_videos(channel_url: str, limit: int | None = 50) -> list[dict]:
    """channel_url e.g. https://www.youtube.com/@JoshGambrell/videos"""
    return _yt_dlp_flat_playlist(channel_url, limit)


def search(query: str, limit: int = 10) -> list[dict]:
    return _yt_dlp_flat_playlist(f"ytsearch{limit}:{query}")


def is_live(url: str) -> bool:
    """Cheap liveness check -- simulates the download without fetching media."""
    result = subprocess.run(
        ["yt-dlp", "--simulate", "--no-warnings", url],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def filter_by_keywords(videos: list[dict], keywords: list[str]) -> list[dict]:
    keywords = [k.lower() for k in keywords]
    return [v for v in videos if any(k in (v["title"] or "").lower() for k in keywords)]
