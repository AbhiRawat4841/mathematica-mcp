"""Single-pass Wolfram Language scanner for string/comment stripping and brace counting.

Used by expression classifier (routing_memory), cache analysis (cache),
and response filtering (response_filter). Centralised to prevent drift
and handle nested Mathematica comments correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Scanner result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResult:
    """Result of scanning Wolfram code."""

    cleaned: str  # stripped code with spaces replacing removed regions
    ok: bool  # False if unbalanced strings/comments detected


# ---------------------------------------------------------------------------
# Single-pass scanner
# ---------------------------------------------------------------------------

# States
_NORMAL = 0
_STRING = 1
_COMMENT = 2


def scan_clean(code: str) -> ScanResult:
    """Strip string literals and nested Mathematica comments in one pass.

    Stripped regions are replaced with spaces to preserve token boundaries
    (so ``x(*c*)y`` becomes ``x     y``, not ``xy``).

    Returns ``ScanResult(ok=False)`` on malformed input (unbalanced strings
    or comments).  The ``cleaned`` field still contains a best-effort result.
    """
    out: list[str] = []
    state = _NORMAL
    comment_depth = 0
    i = 0
    n = len(code)
    ok = True

    while i < n:
        ch = code[i]

        if state == _NORMAL:
            # Check for comment open: (*
            if ch == "(" and i + 1 < n and code[i + 1] == "*":
                state = _COMMENT
                comment_depth = 1
                out.append("  ")  # replace (* with two spaces
                i += 2
                continue

            # Check for string open: "
            if ch == '"':
                state = _STRING
                out.append(" ")  # replace opening quote
                i += 1
                continue

            out.append(ch)
            i += 1

        elif state == _STRING:
            # Handle escaped characters inside strings
            if ch == "\\" and i + 1 < n:
                out.append("  ")  # replace escape sequence with spaces
                i += 2
                continue

            if ch == '"':
                # Closing quote
                state = _NORMAL
                out.append(" ")  # replace closing quote
                i += 1
                continue

            out.append(" ")  # replace string content with space
            i += 1

        elif state == _COMMENT:
            # Nested comment open
            if ch == "(" and i + 1 < n and code[i + 1] == "*":
                comment_depth += 1
                out.append("  ")
                i += 2
                continue

            # Comment close
            if ch == "*" and i + 1 < n and code[i + 1] == ")":
                comment_depth -= 1
                out.append("  ")
                i += 2
                if comment_depth == 0:
                    state = _NORMAL
                continue

            out.append(" ")  # replace comment content
            i += 1

    # Check for unbalanced state
    if state != _NORMAL:
        ok = False

    return ScanResult(cleaned="".join(out), ok=ok)


# ---------------------------------------------------------------------------
# Brace counting
# ---------------------------------------------------------------------------


def count_top_level_braces(code: str) -> int | None:
    """Count top-level comma-separated elements in ``{...}`` output.

    Ignores braces/commas inside strings and comments.
    Returns ``None`` if the output is not a well-formed brace-delimited list
    (e.g. unbalanced braces, trailing non-whitespace after ``}``).

    Tolerates leading whitespace before ``{``.
    """
    stripped = code.lstrip()
    if not stripped or stripped[0] != "{":
        return None

    # Find the matching closing brace using the scanner
    depth = 0
    count = 0
    found_close = -1
    state = _NORMAL
    comment_depth = 0
    i = 0
    n = len(stripped)
    has_content = False

    while i < n:
        ch = stripped[i]

        if state == _NORMAL:
            if ch == "(" and i + 1 < n and stripped[i + 1] == "*":
                state = _COMMENT
                comment_depth = 1
                i += 2
                continue
            if ch == '"':
                state = _STRING
                has_content = True
                i += 1
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    found_close = i
                    break
            elif ch == "," and depth == 1:
                count += 1
            elif depth == 1 and not ch.isspace():
                has_content = True

            i += 1

        elif state == _STRING:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                state = _NORMAL
            i += 1

        elif state == _COMMENT:
            if ch == "(" and i + 1 < n and stripped[i + 1] == "*":
                comment_depth += 1
                i += 2
                continue
            if ch == "*" and i + 1 < n and stripped[i + 1] == ")":
                comment_depth -= 1
                if comment_depth == 0:
                    state = _NORMAL
                i += 2
                continue
            i += 1

    if found_close < 0:
        return None  # unbalanced

    # Reject trailing non-whitespace after closing brace
    trailing = stripped[found_close + 1 :]
    if trailing.strip():
        return None

    # Empty list {}
    if not has_content and count == 0:
        return 0

    # Element count = commas + 1 (if there's any content)
    if has_content or count > 0:
        return count + 1

    return 0
