"""
Split a transcript into candidate edit segments using narration content
instead of viewport pixel-diffing.

The pilot run (see README "First experiment" / "Second and third videos")
found pixel-diff boundary detection unreliable in two different ways across
three real videos: it can't distinguish camera orbit/zoom from an actual
mesh edit, and it gets contaminated by talking-head overlays. Narration
never has either problem -- an artist naming a tool or hotkey is a much
more direct signal that a new discrete operation started.

This is a heuristic, not a parser: it looks for mentions of well-known
Blender hotkeys and tool/operator names and starts a new segment at the
first narration cue that mentions one, provided enough time has passed
since the last split (to avoid over-splitting on a single sentence that
happens to name two things in a row, e.g. "select one, shift G, co-planar").
Expect to extend ACTION_CUES as more videos get processed -- it will always
be incomplete for a domain with this many named operators.
"""
import re

# Each pattern is matched case-insensitively against a narration cue's text.
# Grouped roughly by what they cover; order doesn't matter for matching.
ACTION_CUES = [
    # hotkeys artists narrate literally
    r"\bpress f\b", r"\bshift r\b", r"\bshift g\b", r"\bshift d\b", r"\bshift a\b",
    r"\bctrl f\b", r"\bcontrol f\b", r"\bctrl r\b", r"\bcontrol r\b",
    r"\bctrl b\b", r"\bcontrol b\b", r"\bctrl 2\b", r"\bcontrol two\b", r"\bcontrol 2\b",
    r"\bctrl j\b", r"\bcontrol j\b",
    # named operators / tools
    r"\bbevel\b", r"\bextrude\b", r"\binset\b", r"\bsubdivide\b", r"\bsymmetrize\b",
    r"\bmirror(ed|ing)?\b", r"\bboolean\b", r"\bsolidify\b", r"\bshrinkwrap\b",
    r"\bshade smooth\b", r"\bshade flat\b", r"\bgrid fill\b", r"\bloop tools?\b",
    r"\blimited dissolve\b", r"\bdissolve\b", r"\bdelete\b", r"\bloop cut\b",
    r"\bknife\b", r"\bbisect\b", r"\barray\b", r"\bcurve modifier\b",
    r"\bsub[- ]?d\b", r"\bsubsurf\b", r"\bsubdivision surface\b",
    r"\badd (?:a|an|in a|in an)\b", r"\badd modifier\b",
    r"\bselect (?:all|one|similar|co-?planar)\b",
    r"\bmerge\b", r"\bjoin\b", r"\bseparate\b", r"\bapply\b",
    r"\bscale\b", r"\brotate\b", r"\bflip\b", r"\bchamfer\b",
    r"\bedit mode\b", r"\bobject mode\b",
    # UV/texturing
    r"\bunwrap\b", r"\bmark (?:a )?seam\b", r"\bsmart uv\b",
]
_ACTION_RE = re.compile("|".join(ACTION_CUES), re.IGNORECASE)


def mentions_action_cue(text: str) -> bool:
    return bool(_ACTION_RE.search(text))


def overlapping_narration(transcript: list[dict], t_start: float, t_end: float) -> str:
    texts = [
        seg["text"] for seg in transcript
        if seg["t_end"] > t_start and seg["t_start"] < t_end
    ]
    return " ".join(texts).strip()


def segment_transcript(transcript: list[dict], min_segment_s: float = 1.5) -> list[dict]:
    """Split narration into segments, starting a new one at each cue that
    mentions an action and is at least min_segment_s past the last split."""
    if not transcript:
        return []

    boundaries = [transcript[0]["t_start"]]
    for cue in transcript:
        if mentions_action_cue(cue["text"]) and cue["t_start"] - boundaries[-1] >= min_segment_s:
            boundaries.append(cue["t_start"])

    end_t = transcript[-1]["t_end"]
    if end_t - boundaries[-1] > 0:
        boundaries.append(end_t)

    segments = []
    for t_start, t_end in zip(boundaries[:-1], boundaries[1:]):
        segments.append({
            "t_start": t_start,
            "t_end": t_end,
            "narration": overlapping_narration(transcript, t_start, t_end),
        })
    return segments
