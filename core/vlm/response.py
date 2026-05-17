"""Normalize raw VLM outputs (strip reasoning blocks, markup, etc.)."""

import re

# <think>\n...\n</think>\n\n
_THINKING_BLOCK_RE = re.compile(
    r"<think>\s*[\s\S]*?</think>\s*",
    flags=re.IGNORECASE,
)
_THINKING_TAIL_RE = re.compile(r"<think>[\s\S]*\Z", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def strip_vlm_reasoning(text: str) -> str:
    """Remove model reasoning and return the final answer."""
    s = (text or "").strip()
    if not s:
        return ""

    s = _THINKING_BLOCK_RE.sub("", s)
    s = _THINKING_TAIL_RE.sub("", s).strip()

    if "<|begin_of_box|>" in s and "<|end_of_box|>" in s:
        try:
            s = s.split("<|begin_of_box|>", 1)[1].split("<|end_of_box|>", 1)[0]
        except (IndexError, ValueError):
            pass

    if "<answer" in s.lower():
        low = s.lower()
        i = low.find("<answer")
        if i != -1:
            j = low.find(">", i)
            if j != -1:
                s = s[j + 1 :]
            low2 = s.lower()
            k = low2.find("</answer>")
            if k != -1:
                s = s[:k]

    s = _THINKING_BLOCK_RE.sub(" ", s)
    s = re.sub(r"<\|.*?\|>", " ", s)
    s = re.sub(r"</?[^>]+>", " ", s)
    s = _WS_RE.sub(" ", s).strip()

    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return s
