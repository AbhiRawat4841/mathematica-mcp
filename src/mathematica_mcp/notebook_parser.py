"""
Python-native Mathematica notebook (.nb) parser.

This module provides pure Python parsing of Mathematica notebook files,
converting complex BoxData structures into readable text or Wolfram Language code.

Key features:
- Parse BoxData structures (RowBox, FractionBox, SuperscriptBox, etc.)
- Convert Greek letters and special characters to Unicode
- Extract cells by type (Input, Output, Text, Section, etc.)
- Generate clean, executable Wolfram code OR readable Markdown
- Parse tables (GridBox) into Markdown tables

This parser works offline without requiring wolframscript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

TRUNCATION_THRESHOLD = 25000  # 25KB


def truncate_large_expression(
    text: str, threshold: int = TRUNCATION_THRESHOLD
) -> tuple[str, bool, int]:
    """Truncate text if it exceeds the threshold."""
    original_length = len(text)
    if original_length <= threshold:
        return text, False, original_length

    keep_start = int(threshold * 0.8)
    keep_end = int(threshold * 0.1)

    truncation_notice = (
        f"\n\n[... TRUNCATED: Expression too large ({original_length:,} chars, "
        f"threshold {threshold:,}). Use get_notebook_cell with full=True for complete content ...]\n\n"
    )

    truncated_text = text[:keep_start] + truncation_notice + text[-keep_end:]
    return truncated_text, True, original_length


class CellStyle(Enum):
    """Mathematica cell styles."""

    TITLE = "Title"
    CHAPTER = "Chapter"
    SECTION = "Section"
    SUBSECTION = "Subsection"
    SUBSUBSECTION = "Subsubsection"
    TEXT = "Text"
    INPUT = "Input"
    OUTPUT = "Output"
    CODE = "Code"
    MESSAGE = "Message"
    PRINT = "Print"
    ITEM = "Item"
    ITEM_NUMBERED = "ItemNumbered"
    UNKNOWN = "Unknown"


@dataclass
class Cell:
    """Represents a single cell in a Mathematica notebook."""

    style: CellStyle
    content: str
    raw_content: str = ""
    cell_label: str = ""
    cell_index: int = 0
    was_truncated: bool = False
    original_length: int = 0

    def is_code(self) -> bool:
        return self.style in (CellStyle.INPUT, CellStyle.CODE)

    def is_documentation(self) -> bool:
        return self.style in (
            CellStyle.TITLE,
            CellStyle.CHAPTER,
            CellStyle.SECTION,
            CellStyle.SUBSECTION,
            CellStyle.SUBSUBSECTION,
            CellStyle.TEXT,
            CellStyle.ITEM,
            CellStyle.ITEM_NUMBERED,
        )


@dataclass
class NotebookStructure:
    """Represents the structure of a Mathematica notebook."""

    cells: list[Cell] = field(default_factory=list)
    path: str = ""
    title: str = ""

    def get_code_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.is_code()]

    def get_documentation_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.is_documentation()]

    def get_outline(self) -> list[dict]:
        outline = []
        for cell in self.cells:
            if cell.style in (
                CellStyle.TITLE,
                CellStyle.CHAPTER,
                CellStyle.SECTION,
                CellStyle.SUBSECTION,
                CellStyle.SUBSUBSECTION,
            ):
                outline.append(
                    {
                        "level": cell.style.value,
                        "title": cell.content.strip(),
                        "index": cell.cell_index,
                    }
                )
        return outline


# Expanded Greek letters and special characters (Unicode)
SPECIAL_CHARS = {
    # Greek letters (lowercase)
    r"\[Alpha]": "α",
    r"\[Beta]": "β",
    r"\[Gamma]": "γ",
    r"\[Delta]": "δ",
    r"\[Epsilon]": "ε",
    r"\[CurlyEpsilon]": "ε",
    r"\[Zeta]": "ζ",
    r"\[Eta]": "η",
    r"\[Theta]": "θ",
    r"\[CurlyTheta]": "θ",
    r"\[Iota]": "ι",
    r"\[Kappa]": "κ",
    r"\[Lambda]": "λ",
    r"\[Mu]": "μ",
    r"\[Nu]": "ν",
    r"\[Xi]": "ξ",
    r"\[Omicron]": "ο",
    r"\[Pi]": "π",
    r"\[CurlyPi]": "ϖ",
    r"\[Rho]": "ρ",
    r"\[Sigma]": "σ",
    r"\[FinalSigma]": "ς",
    r"\[Tau]": "τ",
    r"\[Upsilon]": "υ",
    r"\[Phi]": "φ",
    r"\[CurlyPhi]": "ϕ",
    r"\[Chi]": "χ",
    r"\[Psi]": "ψ",
    r"\[Omega]": "ω",
    # Greek letters (uppercase)
    r"\[CapitalAlpha]": "Α",
    r"\[CapitalBeta]": "Β",
    r"\[CapitalGamma]": "Γ",
    r"\[CapitalDelta]": "Δ",
    r"\[CapitalEpsilon]": "Ε",
    r"\[CapitalZeta]": "Ζ",
    r"\[CapitalEta]": "Η",
    r"\[CapitalTheta]": "Θ",
    r"\[CapitalIota]": "Ι",
    r"\[CapitalKappa]": "Κ",
    r"\[CapitalLambda]": "Λ",
    r"\[CapitalMu]": "Μ",
    r"\[CapitalNu]": "Ν",
    r"\[CapitalXi]": "Ξ",
    r"\[CapitalOmicron]": "Ο",
    r"\[CapitalPi]": "Π",
    r"\[CapitalRho]": "Ρ",
    r"\[CapitalSigma]": "Σ",
    r"\[CapitalTau]": "Τ",
    r"\[CapitalUpsilon]": "Υ",
    r"\[CapitalPhi]": "Φ",
    r"\[CapitalChi]": "Χ",
    r"\[CapitalPsi]": "Ψ",
    r"\[CapitalOmega]": "Ω",
    # Mathematical symbols
    r"\[Infinity]": "∞",
    r"\[Degree]": "°",
    r"\[Rule]": "→",
    r"\[RuleDelayed]": ":>",
    r"\[Equal]": "==",
    r"\[NotEqual]": "≠",
    r"\[LessEqual]": "≤",
    r"\[GreaterEqual]": "≥",
    r"\[Element]": "∈",
    r"\[Subset]": "⊂",
    r"\[Superset]": "⊃",
    r"\[Union]": "∪",
    r"\[Intersection]": "∩",
    r"\[EmptySet]": "∅",
    r"\[And]": "∧",
    r"\[Or]": "∨",
    r"\[Not]": "¬",
    r"\[Implies]": "⟹",
    r"\[Equivalent]": "⟺",
    r"\[ForAll]": "∀",
    r"\[Exists]": "∃",
    # Operators
    r"\[Times]": "*",
    r"\[Divide]": "/",
    r"\[PlusMinus]": "±",
    r"\[MinusPlus]": "∓",
    r"\[Cross]": "×",
    r"\[CircleTimes]": "⊗",
    r"\[CirclePlus]": "⊕",
    r"\[TensorProduct]": "⊗",
    r"\[Dot]": "·",
    r"\[CenterDot]": "·",
    r"\[SmallCircle]": "∘",
    # Arrows
    r"\[LeftArrow]": "←",
    r"\[RightArrow]": "→",
    r"\[UpArrow]": "↑",
    r"\[DownArrow]": "↓",
    r"\[LeftRightArrow]": "↔",
    r"\[DoubleRightArrow]": "⇒",
    r"\[DoubleLeftArrow]": "⇐",
    r"\[LongRightArrow]": "⟶",
    r"\[LongLeftArrow]": "⟵",
    # Brackets
    r"\[LeftDoubleBracket]": "[[",
    r"\[RightDoubleBracket]": "]]",
    r"\[LeftAssociation]": "<|",
    r"\[RightAssociation]": "|>",
    r"\[LeftAngleBracket]": "⟨",
    r"\[RightAngleBracket]": "⟩",
    # Spacing
    r"\[IndentingNewLine]": "\n",
    r"\[NewLine]": "\n",
    r"\[InvisibleSpace]": "",
    r"\[VeryThinSpace]": " ",
    r"\[ThinSpace]": " ",
    r"\[MediumSpace]": " ",
    r"\[ThickSpace]": " ",
    r"\[InvisibleTimes]": " ",
    # Constants/Functions
    r"\[ExponentialE]": "E",
    r"\[ImaginaryI]": "I",
    r"\[ImaginaryJ]": "I",
    r"\[DifferentialD]": "d",
    r"\[PartialD]": "∂",
    r"\[Integral]": "∫",
    r"\[Sum]": "Σ",
    r"\[Product]": "Π",
    r"\[Sqrt]": "√",
    # Misc
    r"\[Null]": "",
    r"\[SpanFromAbove]": "",
    r"\[SpanFromLeft]": "",
    r"\[Blank]": "_",
    r"\[BlankSequence]": "__",
    r"\[BlankNullSequence]": "___",
}


def convert_special_chars(text: str) -> str:
    """Convert Mathematica special character codes to readable form."""
    result = text
    for pattern, replacement in SPECIAL_CHARS.items():
        result = result.replace(pattern, replacement)

    # Fallback for any remaining [Name]
    remaining = re.findall(r"\\\\?\[([A-Z][A-Za-z]+)\]", result)
    for name in remaining:
        # Keep name but strip brackets for readability
        result = result.replace(f"\\[{name}]", name)
        result = result.replace(f"[{name}]", name)

    return result


def clean_commutators(text: str) -> str:
    """Normalize commutator notation Cmt[a,b] -> [a, b]."""
    # Placeholder for future commutator normalization implementation
    return text


class BoxDataParser:
    """
    Parser for Mathematica BoxData structures.

    Modes:
    - "wolfram": Generate valid Wolfram Language code (Subscript[a,b])
    - "readable": Generate readable text/Markdown (a_b, tables)
    """

    def __init__(self, mode: str = "wolfram"):
        self.depth = 0
        self.max_depth = 100
        self.mode = mode  # "wolfram" or "readable"

    def parse(self, content: str) -> str:
        """Parse BoxData content."""
        self.depth = 0
        if not content or content.strip() == "":
            return ""

        # Check if already clean
        if not any(
            x in content
            for x in [
                "BoxData",
                "RowBox",
                "FractionBox",
                "SuperscriptBox",
                "SubscriptBox",
                "SqrtBox",
                "GridBox",
                "TemplateBox",
                "FormBox",
            ]
        ):
            if "Cell[" not in content:
                return convert_special_chars(content)

        try:
            if "BoxData[" in content:
                match = re.search(r"BoxData\[(.*)\]", content, re.DOTALL)
                if match:
                    content = match.group(1)

            content = content.strip()

            # Handle top-level list
            if content.startswith("{") and content.endswith("}"):
                inner = content[1:-1].strip()
                if inner.startswith("RowBox[") or inner.startswith('"'):
                    elements = self._parse_list_elements(inner)
                    parsed_elements = []
                    for elem in elements:
                        parsed = self._parse_expression(elem.strip())
                        if parsed:
                            parsed_elements.append(parsed)
                    # Join with newlines for code blocks
                    return "\n".join(parsed_elements)

            result = self._parse_expression(content)
            result = convert_special_chars(result)
            return result
        except Exception as e:
            try:
                with open("/tmp/parser_error.log", "a") as f:
                    f.write(
                        f"Parsing error in mode {self.mode}: {str(e)}\nContext: {content[:200]}...\n"
                    )
            except:
                pass
            return self._basic_cleanup(content)

    def _parse_expression(self, expr: str) -> str:
        self.depth += 1
        if self.depth > self.max_depth:
            return expr

        expr = expr.strip()
        try:
            if expr.startswith("RowBox["):
                return self._parse_row_box(expr)
            elif expr.startswith("FractionBox["):
                return self._parse_fraction_box(expr)
            elif expr.startswith("SuperscriptBox["):
                return self._parse_superscript_box(expr)
            elif expr.startswith("SubscriptBox["):
                return self._parse_subscript_box(expr)
            elif expr.startswith("SubsuperscriptBox["):
                return self._parse_subsuperscript_box(expr)
            elif expr.startswith("SqrtBox["):
                return self._parse_sqrt_box(expr)
            elif expr.startswith("RadicalBox["):
                return self._parse_radical_box(expr)
            elif expr.startswith("GridBox["):
                return self._parse_grid_box(expr)
            elif expr.startswith("FormBox["):
                return self._parse_form_box(expr)
            elif expr.startswith("TagBox["):
                return self._parse_tag_box(expr)
            elif expr.startswith("StyleBox["):
                return self._parse_style_box(expr)
            elif expr.startswith("InterpretationBox["):
                return self._parse_interpretation_box(expr)
            elif expr.startswith("TemplateBox["):
                return self._parse_template_box(expr)
            elif expr.startswith("UnderscriptBox["):
                return self._parse_underscript_box(expr)
            elif expr.startswith("OverscriptBox["):
                return self._parse_overscript_box(expr)
            elif expr.startswith("UnderoverscriptBox["):
                return self._parse_underoverscript_box(expr)
            elif expr.startswith('"') and expr.endswith('"'):
                return expr[1:-1]
            elif expr.startswith("{") and expr.endswith("}"):
                return self._parse_list(expr)
            else:
                return expr
        finally:
            self.depth -= 1

    def _extract_box_content(self, expr: str, box_type: str) -> str:
        prefix = f"{box_type}["
        if not expr.startswith(prefix):
            return expr
        depth = 0
        start = len(prefix)
        for i, char in enumerate(expr):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return expr[start:i]
        return expr[start:-1] if expr.endswith("]") else expr[start:]

    def _parse_list_elements(self, content: str) -> list[str]:
        elements = []
        current = []
        depth = 0
        in_string = False
        escape_next = False
        for char in content:
            if escape_next:
                current.append(char)
                escape_next = False
                continue
            if char == "\\":
                current.append(char)
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                current.append(char)
                continue
            if in_string:
                current.append(char)
                continue
            if char in "[{(":
                depth += 1
                current.append(char)
            elif char in "]})":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                elements.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            elements.append("".join(current).strip())
        return elements

    # --- Box Handlers ---

    def _parse_row_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "RowBox")
        if content.startswith("{") and content.endswith("}"):
            content = content[1:-1]
        elements = self._parse_list_elements(content)
        parsed = [self._parse_expression(e) for e in elements]
        return "".join(parsed).replace("\\n", "\n")

    def _parse_fraction_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "FractionBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            num = self._parse_expression(elements[0])
            denom = self._parse_expression(elements[1])
            if self.mode == "readable":
                return f"\\frac{{{num}}}{{{denom}}}"
            else:
                if any(x in num for x in " +-"):
                    num = f"({num})"
                if any(x in denom for x in " +-"):
                    denom = f"({denom})"
                return f"{num}/{denom}"
        return content

    def _parse_superscript_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "SuperscriptBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            exp = self._parse_expression(elements[1])
            if self.mode == "readable":
                return f"{{{base}}}^{{{exp}}}"

            # Wolfram mode
            if exp == "T":
                return f"Transpose[{base}]"
            elif exp == "-1":
                return f"Inverse[{base}]"
            elif exp == "*":
                return f"Conjugate[{base}]"
            else:
                if len(base) > 1 and not (base.startswith("(") and base.endswith(")")):
                    if not base[0].isupper():
                        base = f"({base})"
                return f"{base}^{exp}"
        return content

    def _parse_subscript_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "SubscriptBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            sub = self._parse_expression(elements[1])
            if self.mode == "readable":
                return f"{{{base}}}_{{{sub}}}"
            return f"Subscript[{base}, {sub}]"
        return content

    def _parse_subsuperscript_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "SubsuperscriptBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 3:
            base = self._parse_expression(elements[0])
            sub = self._parse_expression(elements[1])
            sup = self._parse_expression(elements[2])
            if self.mode == "readable":
                return f"{{{base}}}_{{{sub}}}^{{{sup}}}"
            return f"Subsuperscript[{base}, {sub}, {sup}]"
        return content

    def _parse_sqrt_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "SqrtBox")
        inner = self._parse_expression(content)
        if self.mode == "readable":
            return f"\\sqrt{{{inner}}}"
        return f"Sqrt[{inner}]"

    def _parse_radical_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "RadicalBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            inner = self._parse_expression(elements[0])
            n = self._parse_expression(elements[1])
            if self.mode == "readable":
                return f"\\sqrt[{n}]{{{inner}}}"
            return f"Power[{inner}, 1/{n}]"
        return content

    def _parse_grid_box(self, expr: str) -> str:
        try:
            with open("/tmp/parser_debug.log", "a") as f:
                f.write(f"Parsing GridBox in mode: {self.mode}\n")
        except:
            pass

        content = self._extract_box_content(expr, "GridBox")

        # Extract matrix content
        matrix_content = content
        if content.startswith("{"):
            # Find the closing brace for the list
            depth = 0
            for i, char in enumerate(content):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        matrix_content = content[: i + 1]
                        break

        parsed_rows = self._parse_matrix_rows(matrix_content)

        if self.mode == "readable":
            # Convert to Markdown Table
            if not parsed_rows:
                return ""

            md_lines = []

            # Format rows
            fmt_rows = []
            for row in parsed_rows:
                fmt_row = []
                for cell in row:
                    # Clean up quotes for strings
                    cell_text = cell.strip()

                    # Handle Mathematica \<"..."\> string wrapper
                    if cell_text.startswith(r"\<") and cell_text.endswith(r"\>"):
                        cell_text = cell_text[2:-2]

                    if cell_text.startswith('"') and cell_text.endswith('"'):
                        cell_text = cell_text[1:-1]
                        # Unescape quotes
                        cell_text = cell_text.replace('\\"', '"')

                    # Handle \" escapes that might remain
                    cell_text = cell_text.replace('\\"', '"')

                    # Escape pipes for Markdown
                    cell_text = cell_text.replace("|", "\\|")
                    fmt_row.append(cell_text)
                fmt_rows.append(fmt_row)

            # Determine columns
            max_cols = max(len(r) for r in fmt_rows) if fmt_rows else 0

            # Construct Header
            header = fmt_rows[0]
            # Pad header if needed
            while len(header) < max_cols:
                header.append("")

            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

            for row in fmt_rows[1:]:
                while len(row) < max_cols:
                    row.append("")
                md_lines.append("| " + " | ".join(row) + " |")

            return "\n" + "\n".join(md_lines) + "\n"

        else:
            # Wolfram format: {{a,b},{c,d}}
            rows_str = []
            for row in parsed_rows:
                rows_str.append("{" + ", ".join(row) + "}")
            return "{" + ", ".join(rows_str) + "}"

    def _parse_matrix_rows(self, content: str) -> list[list[str]]:
        if not (content.startswith("{") and content.endswith("}")):
            return []
        inner = content[1:-1].strip()
        rows = []
        depth = 0
        bracket_depth = 0
        current_elem = []
        current_row = []
        in_string = False
        escape_next = False

        for char in inner:
            if escape_next:
                if depth > 0:
                    current_elem.append(char)
                escape_next = False
                continue

            if char == "\\":
                if depth > 0:
                    current_elem.append(char)
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                if depth > 0:
                    current_elem.append(char)
                continue

            if in_string:
                if depth > 0:
                    current_elem.append(char)
                continue

            if char == "{":
                if depth == 0:
                    depth = 1  # Start row
                else:
                    depth += 1
                    current_elem.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:  # End row
                    if current_elem:
                        current_row.append("".join(current_elem).strip())
                    rows.append(current_row)
                    current_row = []
                    current_elem = []
                else:
                    current_elem.append(char)
            elif char == "[":
                bracket_depth += 1
                if depth > 0:
                    current_elem.append(char)
            elif char == "]":
                bracket_depth -= 1
                if depth > 0:
                    current_elem.append(char)
            elif char == "," and depth == 1 and bracket_depth == 0:
                # Only split if we are in a row and NOT inside brackets
                if current_elem:
                    current_row.append("".join(current_elem).strip())
                current_elem = []
            elif depth > 0:
                current_elem.append(char)

        # Now parse elements recursively
        parsed_rows = []
        for row in rows:
            parsed_rows.append([self._parse_expression(e) for e in row])
        return parsed_rows

    def _parse_form_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "FormBox")
        elements = self._parse_list_elements(content)
        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_tag_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "TagBox")
        elements = self._parse_list_elements(content)
        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_style_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "StyleBox")
        elements = self._parse_list_elements(content)
        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_interpretation_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "InterpretationBox")
        elements = self._parse_list_elements(content)
        if self.mode == "readable":
            if elements:
                return self._parse_expression(elements[0])
        else:
            if len(elements) >= 2:
                return self._parse_expression(elements[1])
            elif elements:
                return self._parse_expression(elements[0])
        return content

    def _parse_template_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "TemplateBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            args_str = elements[0]
            template = elements[1].strip('"')
            if args_str.startswith("{") and args_str.endswith("}"):
                args = self._parse_list_elements(args_str[1:-1])
                parsed_args = [self._parse_expression(a) for a in args]
                if template == "Times":
                    return " ".join(parsed_args)
                elif template == "Divide":
                    if self.mode == "readable":
                        return f"\\frac{{{parsed_args[0]}}}{{{parsed_args[1]}}}"
                    return f"({parsed_args[0]})/({parsed_args[1]})"
                elif template == "RowDefault":
                    return ",".join(parsed_args)

                return f"{template}[{', '.join(parsed_args)}]"
        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_underscript_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "UnderscriptBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            under = self._parse_expression(elements[1])
            if self.mode == "readable":
                return f"\\underset{{{under}}}{{{base}}}"
            return f"Underscript[{base}, {under}]"
        return content

    def _parse_overscript_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "OverscriptBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            over = self._parse_expression(elements[1])
            if self.mode == "readable":
                return f"\\overset{{{over}}}{{{base}}}"
            return f"Overscript[{base}, {over}]"
        return content

    def _parse_underoverscript_box(self, expr: str) -> str:
        content = self._extract_box_content(expr, "UnderoverscriptBox")
        elements = self._parse_list_elements(content)
        if len(elements) >= 3:
            base = self._parse_expression(elements[0])
            under = self._parse_expression(elements[1])
            over = self._parse_expression(elements[2])
            if self.mode == "readable":
                return f"\\overset{{{over}}}{{\\underset{{{under}}}{{{base}}}}}"
            return f"Underoverscript[{base}, {under}, {over}]"
        return content

    def _parse_list(self, expr: str) -> str:
        content = expr[1:-1]
        elements = self._parse_list_elements(content)
        parsed = [self._parse_expression(e) for e in elements]
        if self.mode == "readable":
            return "\\{" + ", ".join(parsed) + "\\}"
        return "{" + ", ".join(parsed) + "}"

    def _basic_cleanup(self, content: str) -> str:
        result = content
        for box_type in ["BoxData", "RowBox", "Cell", "FormBox"]:
            result = re.sub(rf"{box_type}\[", "", result)
        return convert_special_chars(result)


class NotebookParser:
    """
    Parser for Mathematica notebook (.nb) files.
    """

    def __init__(self, truncation_threshold: int = TRUNCATION_THRESHOLD):
        self.wolfram_parser = BoxDataParser(mode="wolfram")
        self.readable_parser = BoxDataParser(mode="readable")
        self.truncation_threshold = truncation_threshold

    def parse_file(self, path: str | Path) -> NotebookStructure:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_content(content, str(path))

    def parse_content(self, content: str, path: str = "") -> NotebookStructure:
        notebook = NotebookStructure(path=path)
        title_match = re.search(r'Cell\["([^"]+)",\s*"Title"', content)
        if title_match:
            notebook.title = title_match.group(1)
        notebook.cells = self._extract_cells(content)
        return notebook

    def _extract_cells(self, content: str) -> list[Cell]:
        cells = []
        cell_index = 0
        i = 0
        while i < len(content):
            cell_start = content.find("Cell[", i)
            if cell_start == -1:
                break

            depth = 0
            j = cell_start
            in_string = False
            escape_next = False
            found_end = False

            while j < len(content):
                char = content[j]
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if char == "\\":
                    escape_next = True
                    j += 1
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            found_end = True
                            break
                j += 1

            if not found_end:
                break

            cell_text = content[cell_start : j + 1]

            # Check for CellGroupData and flatten structure
            if re.match(r"Cell\[\s*CellGroupData", cell_text):
                cell_content = self._extract_cell_content(cell_text)

                # Extract CellGroupData content manually since we can't access BoxDataParser here easily
                # CellGroupData[...]
                prefix = "CellGroupData["
                group_content = cell_content
                if cell_content.startswith(prefix):
                    depth_grp = 0
                    start_grp = len(prefix)
                    for k, char_grp in enumerate(cell_content):
                        if char_grp == "[":
                            depth_grp += 1
                        elif char_grp == "]":
                            depth_grp -= 1
                            if depth_grp == 0:
                                group_content = cell_content[start_grp:k]
                                break

                if group_content.strip().startswith("{"):
                    list_depth = 0
                    list_end = 0
                    in_s = False
                    esc = False
                    for k, c in enumerate(group_content):
                        if esc:
                            esc = False
                            continue
                        if c == "\\":
                            esc = True
                            continue
                        if c == '"' and not esc:
                            in_s = not in_s
                        if not in_s:
                            if c == "{":
                                list_depth += 1
                            elif c == "}":
                                list_depth -= 1
                                if list_depth == 0:
                                    list_end = k + 1
                                    break

                    if list_end > 0:
                        inner_cells_text = group_content[0:list_end]
                        inner_cells_text = inner_cells_text[1:-1]
                        group_cells = self._extract_cells(inner_cells_text)
                        for gc in group_cells:
                            gc.cell_index = cell_index
                            cell_index += 1
                        cells.extend(group_cells)

                i = j + 1
                continue

            cell = self._parse_cell(cell_text, cell_index)
            if cell:
                cells.append(cell)
                cell_index += 1
            i = j + 1
        return cells

    def _parse_cell(self, cell_text: str, index: int) -> Cell | None:
        if not cell_text.startswith("Cell["):
            return None

        style = self._detect_style(cell_text)
        if style == CellStyle.UNKNOWN:
            return None

        label = ""
        label_match = re.search(r'CellLabel\s*->\s*"([^"]+)"', cell_text)
        if label_match:
            label = label_match.group(1)

        raw_content = self._extract_cell_content(cell_text)

        if style == CellStyle.INPUT or style == CellStyle.CODE:
            if "BoxData[" in raw_content:
                parsed_content = self.wolfram_parser.parse(raw_content)
            else:
                parsed_content = convert_special_chars(raw_content)
        else:
            if "BoxData[" in raw_content:
                parsed_content = self.readable_parser.parse(raw_content)
            elif style in (
                CellStyle.TEXT,
                CellStyle.TITLE,
                CellStyle.SECTION,
                CellStyle.SUBSECTION,
            ):
                parsed_content = self._extract_text_content(raw_content)
            else:
                parsed_content = convert_special_chars(raw_content)

        trunc, was_trunc, orig_len = truncate_large_expression(
            parsed_content, self.truncation_threshold
        )

        return Cell(style, trunc, raw_content, label, index, was_trunc, orig_len)

    def _detect_style(self, cell_text: str) -> CellStyle:
        style_map = {
            "Title": CellStyle.TITLE,
            "Chapter": CellStyle.CHAPTER,
            "Section": CellStyle.SECTION,
            "Subsection": CellStyle.SUBSECTION,
            "Subsubsection": CellStyle.SUBSUBSECTION,
            "Text": CellStyle.TEXT,
            "Input": CellStyle.INPUT,
            "Output": CellStyle.OUTPUT,
            "Code": CellStyle.CODE,
            "Message": CellStyle.MESSAGE,
            "Print": CellStyle.PRINT,
            "Item": CellStyle.ITEM,
            "ItemNumbered": CellStyle.ITEM_NUMBERED,
        }
        try:
            for name, style in style_map.items():
                if re.search(rf', \s*"{name}"', cell_text):
                    return style
        except Exception:
            pass
        return CellStyle.UNKNOWN

    def _extract_cell_content(self, cell_text: str) -> str:
        start = 5  # Cell[
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cell_text)):
            c = cell_text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                elif c == "," and depth == 0:
                    return cell_text[start:i].strip()
        return cell_text[start:-1].strip()

    def _extract_text_content(self, content: str) -> str:
        if content.startswith('"') and content.endswith('"'):
            return content[1:-1]
        if content.startswith("TextData["):
            return "".join(re.findall(r'"([^"]*)"', content))
        return convert_special_chars(content)

    def to_markdown(self, notebook: NotebookStructure) -> str:
        lines = []
        if notebook.title:
            lines.append(f"# {notebook.title}\n")

        for cell in notebook.cells:
            c = cell.content
            if cell.style == CellStyle.TITLE:
                lines.append(f"# {c}\n")
            elif cell.style == CellStyle.SECTION:
                lines.append(f"## {c}\n")
            elif cell.style == CellStyle.SUBSECTION:
                lines.append(f"### {c}\n")
            elif cell.style == CellStyle.SUBSUBSECTION:
                lines.append(f"#### {c}\n")
            elif cell.style == CellStyle.TEXT:
                lines.append(f"{c}\n")
            elif cell.style == CellStyle.INPUT:
                lbl = f" (* {cell.cell_label} *)" if cell.cell_label else ""
                lines.append(f"```wolfram{lbl}\n{c}\n```\n")
            elif cell.style == CellStyle.OUTPUT:
                if c.strip().startswith("|"):
                    lines.append(f"{c}\n")
                else:
                    if "\\" in c and not c.startswith("```"):
                        lines.append(f"$$\n{c}\n$$\n")
                    else:
                        lines.append(f"```\n(* Output *)\n{c}\n```\n")

            if cell.was_truncated:
                lines.append(f"> *(Truncated {cell.original_length} chars)*\n")

        return "\n".join(lines)


def parse_notebook(path: str | Path) -> NotebookStructure:
    return NotebookParser().parse_file(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(NotebookParser().to_markdown(parse_notebook(sys.argv[1])))
