"""D1: retry_with honesty in the error analyzer.

retry_with must be a runnable, mechanically-corrected call sourced only from a
pattern's explicit "retry_with" field -- never a fallback to prose "example" /
"suggested_fix" strings. Prose-only matched patterns must return None.
"""

from __future__ import annotations

from mathematica_mcp.error_analyzer import ERROR_PATTERNS, analyze_messages


def test_retry_with_none_without_contextual_template():
    """No pattern ships a canned retry_with: a static template cannot substitute
    the user's actual failing input, so Divide::infy now returns None too."""
    result = analyze_messages([{"tag": "Divide::infy", "text": "Infinite expr", "type": "error"}])
    assert result["should_retry"] is True  # high-confidence match still flagged
    assert result["retry_with"] is None
    assert "retry_with" not in ERROR_PATTERNS["Divide::infy"]


def test_retry_with_none_for_prose_only_pattern():
    """A high-confidence match with no retry_with field returns None, not prose.

    Pre-fix this fell back to top.get("example") ("list[[idx, default]] ...")
    instead of None.
    """
    result = analyze_messages([{"tag": "Part::partw", "text": "out of range", "type": "error"}])
    assert result["should_retry"] is True  # high-confidence match
    assert "Part::partw" not in ERROR_PATTERNS or "retry_with" not in ERROR_PATTERNS["Part::partw"]
    assert result["retry_with"] is None


def test_should_retry_reflects_high_confidence_matches():
    """should_retry stays tied to high-confidence matches, independent of retry_with."""
    high = analyze_messages([{"tag": "Part::partd", "text": "too deep", "type": "error"}])
    assert high["should_retry"] is True
    assert high["retry_with"] is None  # matched but prose-only

    unknown = analyze_messages([{"tag": "Totally::unknownxyz", "text": "x", "type": "error"}])
    assert unknown["should_retry"] is False
    assert unknown["retry_with"] is None
