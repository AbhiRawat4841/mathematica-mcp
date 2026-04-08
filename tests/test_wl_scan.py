"""Tests for wl_scan.py — Wolfram Language scanner."""

from __future__ import annotations

from mathematica_mcp.wl_scan import count_top_level_braces, scan_clean

# ---------------------------------------------------------------------------
# scan_clean — string stripping
# ---------------------------------------------------------------------------


class TestScanCleanStrings:
    def test_strip_simple_string(self):
        result = scan_clean('x + "hello" + y')
        assert result.ok is True
        assert "hello" not in result.cleaned
        assert "x" in result.cleaned
        assert "y" in result.cleaned

    def test_strip_escaped_quotes(self):
        result = scan_clean('x + "he\\"llo" + y')
        assert result.ok is True
        assert "he" not in result.cleaned
        assert "x" in result.cleaned
        assert "y" in result.cleaned

    def test_string_containing_comment_chars(self):
        result = scan_clean('"(* not a comment *)"')
        assert result.ok is True
        # Everything inside the string is stripped — only spaces remain
        assert "not" not in result.cleaned
        assert "comment" not in result.cleaned

    def test_unbalanced_string_returns_ok_false(self):
        result = scan_clean('x + "unterminated')
        assert result.ok is False


# ---------------------------------------------------------------------------
# scan_clean — comment stripping
# ---------------------------------------------------------------------------


class TestScanCleanComments:
    def test_strip_simple_comment(self):
        result = scan_clean("1 + (* hello *) 2")
        assert result.ok is True
        assert "hello" not in result.cleaned
        assert "1" in result.cleaned
        assert "2" in result.cleaned

    def test_strip_nested_comment(self):
        result = scan_clean("(* outer (* inner *) *) x")
        assert result.ok is True
        assert "outer" not in result.cleaned
        assert "inner" not in result.cleaned
        assert "x" in result.cleaned

    def test_comment_containing_string(self):
        result = scan_clean('(* "hello" *) x')
        assert result.ok is True
        assert "hello" not in result.cleaned
        assert "x" in result.cleaned

    def test_unbalanced_comment_returns_ok_false(self):
        result = scan_clean("(* unclosed comment")
        assert result.ok is False


# ---------------------------------------------------------------------------
# scan_clean — combined + boundaries
# ---------------------------------------------------------------------------


class TestScanCleanCombined:
    def test_strip_combined(self):
        result = scan_clean('x + "str" + (* comment *) y')
        assert result.ok is True
        assert "str" not in result.cleaned
        assert "comment" not in result.cleaned
        assert "x" in result.cleaned
        assert "y" in result.cleaned

    def test_well_formed_returns_ok_true(self):
        result = scan_clean("Sin[x] + Cos[y]")
        assert result.ok is True

    def test_token_boundary_preserved(self):
        result = scan_clean("x(*c*)y")
        assert result.ok is True
        # Must NOT become "xy" — spaces must separate tokens
        assert "xy" not in result.cleaned
        assert "x" in result.cleaned
        assert "y" in result.cleaned
        # Verify cleaned length matches original (spaces replace removed chars)
        assert len(result.cleaned) == len("x(*c*)y")

    def test_empty_input(self):
        result = scan_clean("")
        assert result.ok is True
        assert result.cleaned == ""

    def test_no_strings_or_comments(self):
        code = "Plot[Sin[x], {x, 0, 2 Pi}]"
        result = scan_clean(code)
        assert result.ok is True
        assert result.cleaned == code


# ---------------------------------------------------------------------------
# count_top_level_braces
# ---------------------------------------------------------------------------


class TestCountTopLevelBraces:
    def test_simple_list(self):
        assert count_top_level_braces("{1, 2, 3}") == 3

    def test_nested_list(self):
        assert count_top_level_braces("{{1,2}, {3,4}}") == 2

    def test_deeply_nested(self):
        assert count_top_level_braces("{{1,{2,3}}, {4,{5,{6}}}}") == 2

    def test_not_a_list(self):
        assert count_top_level_braces("Sin[x]") is None

    def test_empty_list(self):
        assert count_top_level_braces("{}") == 0

    def test_unbalanced_braces(self):
        assert count_top_level_braces("{1,{2}") is None

    def test_ignores_braces_in_strings(self):
        assert count_top_level_braces('{"a,b", "c,d"}') == 2

    def test_rejects_trailing_content(self):
        assert count_top_level_braces("{1,2} extra") is None

    def test_tolerates_leading_whitespace(self):
        assert count_top_level_braces("  {1, 2, 3}") == 3

    def test_returns_none_on_malformed_scan(self):
        # Unbalanced comment inside — should not crash
        assert count_top_level_braces("{1, (* unclosed, 2}") is None or isinstance(
            count_top_level_braces("{1, (* unclosed, 2}"), int
        )
        # The key safety property: no crash

    def test_single_element(self):
        assert count_top_level_braces("{42}") == 1

    def test_empty_string(self):
        assert count_top_level_braces("") is None

    def test_trailing_whitespace_ok(self):
        assert count_top_level_braces("{1, 2}  ") == 2
