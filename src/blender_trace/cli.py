"""
Unified CLI for the BlenderTrace mining pipeline:

    blender-trace discover-channel <channel_url> [--keyword ...] --out cands.json
    blender-trace discover-search "<query>" --out cands.json
    blender-trace discover-check <url>
    blender-trace download <url> [--out data] [--max-height 1080]
    blender-trace transcribe <video_dir>
    blender-trace manifest <video_dir> [--method narration|visual] [--panel-box x0 y0 x1 y1]
    blender-trace keyframes <video_dir> [--panel-box x0 y0 x1 y1]   # only for --method visual
    blender-trace pipeline <url>   # download -> transcribe -> manifest (narration by default)
    blender-trace render <script.py> --out <dir>   # headless-render a bpy/bmesh script
    blender-trace auto-verify <video_dir>   # GPT-powered reconstruct+render+judge, chained
"""
import argparse
import json
import sys
from pathlib import Path

from . import agent as agent_mod
from . import discover as discover_mod
from . import download as download_mod
from . import keyframes as keyframes_mod
from . import manifest as manifest_mod
from . import render as render_mod
from . import transcribe as transcribe_mod
from . import verify as verify_mod

DEFAULT_PANEL_BOX = [0.55, 0.0, 1.0, 0.06]
# 12.0/1.5 (the original defaults) missed a 70+ second run of ~10 distinct
# mesh edits on a real video because the detector compared each sample only
# to the previous sample, not to the last saved keyframe -- slow cumulative
# drift never tripped a single-step threshold. Fixed to diff against the
# last keyframe (see keyframes.py); 6.0/1.0 is what closed most of that gap
# in practice, but still doesn't distinguish camera orbit/zoom from an
# actual mesh edit -- expect to tune per channel, and treat this as a
# starting point, not a solved problem.
DEFAULT_DIFF_THRESHOLD = 6.0
DEFAULT_MIN_GAP_S = 1.0
DEFAULT_MIN_SEGMENT_S = 1.5


def cmd_discover_channel(args):
    videos = discover_mod.list_channel_videos(args.channel_url, limit=args.limit)
    if args.keyword:
        videos = discover_mod.filter_by_keywords(videos, args.keyword)
    Path(args.out).write_text(json.dumps(videos, indent=2))
    print(f"Found {len(videos)} videos, wrote {args.out}")


def cmd_discover_search(args):
    videos = discover_mod.search(args.query, limit=args.limit)
    Path(args.out).write_text(json.dumps(videos, indent=2))
    print(f"Found {len(videos)} videos, wrote {args.out}")


def cmd_discover_check(args):
    ok = discover_mod.is_live(args.url)
    print("LIVE" if ok else "DEAD/UNAVAILABLE")
    sys.exit(0 if ok else 1)


def cmd_download(args):
    download_mod.download(args.url, Path(args.out), args.max_height)


def cmd_transcribe(args):
    transcribe_mod.main(args.video_dir)


def cmd_keyframes(args):
    keyframes_mod.extract(
        args.video_dir, tuple(args.panel_box), args.diff_threshold, args.min_gap_s
    )


def cmd_render(args):
    result = render_mod.render_script(args.script, args.out)
    for name, path in result["render_paths"].items():
        print(f"Rendered {name} -> {path}")
    print(f"Stats: {result['stats']}")


def cmd_verify(args):
    result = verify_mod.verify_segment(args.video_dir, args.segment_index, args.script)
    for name, path in result["render_paths"].items():
        print(f"Rendered {name} -> {path}")
    print(f"Stats: {result['stats']}")
    if args.match is not None:
        verdict = {"match": args.match, "reasoning": args.reasoning, "judge": args.judge}
        path = verify_mod.save_verdict(args.video_dir, args.segment_index, verdict)
        print(f"Saved verdict to {path}")


def cmd_auto_verify(args):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import os
    try:
        from openai import OpenAI
    except ImportError:
        print("Requires the 'openai' package: pip install -e '.[automate]'", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Set it as an env var, or put it in a "
            ".env file at the repo root (OPENAI_API_KEY=sk-...).",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest = json.loads((args.video_dir / "manifest.json").read_text())
    end = args.end if args.end is not None else len(manifest) - 1
    client = OpenAI()

    prior_state = None
    if args.start > 0:
        candidate = verify_mod.segment_dir(args.video_dir, args.start - 1) / "state.blend"
        if candidate.exists():
            prior_state = candidate
        else:
            print(
                f"Warning: no prior state.blend at {candidate}; starting segment "
                f"{args.start} from an empty scene instead of the real chained state.",
                file=sys.stderr,
            )

    tally = {"match": 0, "partial": 0, "mismatch": 0}
    for i in range(args.start, end + 1):
        seg = manifest[i]
        print(f"[{i}] {seg['narration'][:80]}")
        result = agent_mod.run_segment(
            client, args.model, args.video_dir, i, seg, prior_state,
            max_retries=args.max_retries,
        )
        verdict = result.verdict.get("verdict", "mismatch")
        tally[verdict if verdict in tally else "mismatch"] += 1
        print(f"    -> {verdict} ({result.attempts} attempt(s)): {result.verdict.get('reasoning', '')}")
        prior_state = result.state_blend

    print(f"\nDone. {tally}")


def cmd_manifest(args):
    manifest_mod.main(
        args.video_dir,
        method=args.method,
        panel_box=tuple(args.panel_box),
        min_segment_s=args.min_segment_s,
    )


def cmd_pipeline(args):
    video_dir = download_mod.download(args.url, Path(args.out), max_height=1080)
    transcribe_mod.main(video_dir)
    if args.method == "visual":
        keyframes_mod.extract(
            video_dir, tuple(args.panel_box), args.diff_threshold, args.min_gap_s
        )
    manifest_mod.main(
        video_dir,
        method=args.method,
        panel_box=tuple(args.panel_box),
        min_segment_s=args.min_segment_s,
    )
    print(
        f"\nDone. Inspect {video_dir}/manifest.json, and open a couple of "
        f"{video_dir}/panels/*.png to check whether --panel-box needs adjusting "
        f"for this channel."
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="blender-trace")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "discover-channel", help="List a channel's current videos live via yt-dlp"
    )
    p.add_argument("channel_url", help="e.g. https://www.youtube.com/@JoshGambrell/videos")
    p.add_argument("--keyword", action="append", default=[])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default="discovered.json")
    p.set_defaults(func=cmd_discover_channel)

    p = sub.add_parser("discover-search", help="Live YouTube search via yt-dlp")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out", default="discovered.json")
    p.set_defaults(func=cmd_discover_search)

    p = sub.add_parser(
        "discover-check", help="Check whether a URL is currently downloadable"
    )
    p.add_argument("url")
    p.set_defaults(func=cmd_discover_check)

    p = sub.add_parser("download", help="Download one video + captions")
    p.add_argument("url")
    p.add_argument("--out", default="data")
    p.add_argument("--max-height", type=int, default=1080)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("transcribe", help="Build transcript.json for a downloaded video")
    p.add_argument("video_dir", type=Path)
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("keyframes", help="Extract edit-boundary keyframes + panel crops")
    p.add_argument("video_dir", type=Path)
    p.add_argument("--panel-box", type=float, nargs=4, default=DEFAULT_PANEL_BOX,
                    metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--diff-threshold", type=float, default=DEFAULT_DIFF_THRESHOLD)
    p.add_argument("--min-gap-s", type=float, default=DEFAULT_MIN_GAP_S)
    p.set_defaults(func=cmd_keyframes)

    p = sub.add_parser("manifest", help="Build manifest.json from transcript (+ keyframes)")
    p.add_argument("video_dir", type=Path)
    p.add_argument("--method", choices=["narration", "visual"], default="narration",
                   help="narration (default): segment on transcript content, sample frames "
                        "directly. visual: use keyframes.json from a prior `keyframes` run.")
    p.add_argument("--panel-box", type=float, nargs=4, default=DEFAULT_PANEL_BOX,
                    metavar=("X0", "Y0", "X1", "Y1"), help="only used by --method narration")
    p.add_argument("--min-segment-s", type=float, default=DEFAULT_MIN_SEGMENT_S,
                   help="only used by --method narration")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser(
        "pipeline", help="Run download->transcribe->manifest for one video"
    )
    p.add_argument("url")
    p.add_argument("--out", default="data")
    p.add_argument("--method", choices=["narration", "visual"], default="narration")
    p.add_argument("--panel-box", type=float, nargs=4, default=DEFAULT_PANEL_BOX,
                    metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--min-segment-s", type=float, default=DEFAULT_MIN_SEGMENT_S,
                   help="only used by --method narration")
    p.add_argument("--diff-threshold", type=float, default=DEFAULT_DIFF_THRESHOLD,
                   help="only used by --method visual")
    p.add_argument("--min-gap-s", type=float, default=DEFAULT_MIN_GAP_S,
                   help="only used by --method visual")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser(
        "render", help="Headless-render a bpy/bmesh reconstruction script (verification loop)"
    )
    p.add_argument("script", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser(
        "verify", help="Render a segment's reconstruction script + optionally record a verdict"
    )
    p.add_argument("video_dir", type=Path)
    p.add_argument("segment_index", type=int)
    p.add_argument("script", type=Path)
    match_group = p.add_mutually_exclusive_group()
    match_group.add_argument("--match", dest="match", action="store_true", default=None)
    match_group.add_argument("--no-match", dest="match", action="store_false")
    p.add_argument("--reasoning", default="")
    p.add_argument("--judge", default="claude-code-session")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "auto-verify",
        help="GPT-powered reconstruct+render+judge for a manifest range, chained via state.blend",
    )
    p.add_argument("video_dir", type=Path)
    p.add_argument("--start", type=int, default=0, help="first segment index (default 0)")
    p.add_argument("--end", type=int, default=None, help="last segment index, inclusive (default: last)")
    p.add_argument("--model", default=agent_mod.DEFAULT_MODEL)
    p.add_argument("--max-retries", type=int, default=2)
    p.set_defaults(func=cmd_auto_verify)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
