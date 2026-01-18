"""
Python-native Mathematica notebook (.nb) parser.

This module provides pure Python parsing of Mathematica notebook files,
converting complex BoxData structures into readable Wolfram Language code.

Key features:
- Parse BoxData structures (RowBox, FractionBox, SuperscriptBox, etc.)
- Convert Greek letters and special characters
- Extract cells by type (Input, Output, Text, Section, etc.)
- Generate clean, executable Wolfram code

This parser works offline without requiring wolframscript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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

    def is_code(self) -> bool:
        """Check if this cell contains executable code."""
        return self.style in (CellStyle.INPUT, CellStyle.CODE)

    def is_documentation(self) -> bool:
        """Check if this cell is documentation/text."""
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
        """Get only executable code cells."""
        return [c for c in self.cells if c.is_code()]

    def get_documentation_cells(self) -> list[Cell]:
        """Get only documentation cells."""
        return [c for c in self.cells if c.is_documentation()]

    def get_outline(self) -> list[dict]:
        """Get hierarchical outline of sections."""
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


# Greek letters and special Mathematica characters
SPECIAL_CHARS = {
    # Greek letters (lowercase)
    r"\[Alpha]": "alpha",
    r"\[Beta]": "beta",
    r"\[Gamma]": "gamma",
    r"\[Delta]": "delta",
    r"\[Epsilon]": "epsilon",
    r"\[Zeta]": "zeta",
    r"\[Eta]": "eta",
    r"\[Theta]": "theta",
    r"\[Iota]": "iota",
    r"\[Kappa]": "kappa",
    r"\[Lambda]": "lambda",
    r"\[Mu]": "mu",
    r"\[Nu]": "nu",
    r"\[Xi]": "xi",
    r"\[Omicron]": "omicron",
    r"\[Pi]": "Pi",  # Keep capital for Mathematica
    r"\[Rho]": "rho",
    r"\[Sigma]": "sigma",
    r"\[Tau]": "tau",
    r"\[Upsilon]": "upsilon",
    r"\[Phi]": "phi",
    r"\[Chi]": "chi",
    r"\[Psi]": "psi",
    r"\[Omega]": "omega",
    # Greek letters (uppercase) - use backslash prefix
    r"\[CapitalAlpha]": "Alpha",
    r"\[CapitalBeta]": "Beta",
    r"\[CapitalGamma]": "Gamma",
    r"\[CapitalDelta]": "Delta",
    r"\[CapitalEpsilon]": "Epsilon",
    r"\[CapitalZeta]": "Zeta",
    r"\[CapitalEta]": "Eta",
    r"\[CapitalTheta]": "Theta",
    r"\[CapitalIota]": "Iota",
    r"\[CapitalKappa]": "Kappa",
    r"\[CapitalLambda]": "Lambda",
    r"\[CapitalMu]": "Mu",
    r"\[CapitalNu]": "Nu",
    r"\[CapitalXi]": "Xi",
    r"\[CapitalOmicron]": "Omicron",
    r"\[CapitalPi]": "PI",
    r"\[CapitalRho]": "Rho",
    r"\[CapitalSigma]": "Sigma",
    r"\[CapitalTau]": "Tau",
    r"\[CapitalUpsilon]": "Upsilon",
    r"\[CapitalPhi]": "Phi",
    r"\[CapitalChi]": "Chi",
    r"\[CapitalPsi]": "Psi",
    r"\[CapitalOmega]": "Omega",
    # Mathematical symbols
    r"\[Infinity]": "Infinity",
    r"\[Degree]": "Degree",
    r"\[Rule]": "->",
    r"\[RuleDelayed]": ":>",
    r"\[Equal]": "==",
    r"\[NotEqual]": "!=",
    r"\[LessEqual]": "<=",
    r"\[GreaterEqual]": ">=",
    r"\[Element]": "\\[Element]",
    r"\[And]": "&&",
    r"\[Or]": "||",
    r"\[Not]": "!",
    r"\[Implies]": "=>",
    r"\[Equivalent]": "<=>",
    r"\[ForAll]": "ForAll",
    r"\[Exists]": "Exists",
    # Operators
    r"\[Times]": "*",
    r"\[Divide]": "/",
    r"\[PlusMinus]": "+-",
    r"\[MinusPlus]": "-+",
    r"\[Cross]": "\\[Cross]",
    r"\[CircleTimes]": "\\[CircleTimes]",
    r"\[CirclePlus]": "\\[CirclePlus]",
    # Arrows
    r"\[LeftArrow]": "<-",
    r"\[RightArrow]": "->",
    r"\[UpArrow]": "^",
    r"\[DownArrow]": "v",
    r"\[DoubleLeftArrow]": "<=",
    r"\[DoubleRightArrow]": "=>",
    # Brackets and grouping
    r"\[LeftDoubleBracket]": "[[",
    r"\[RightDoubleBracket]": "]]",
    r"\[LeftAssociation]": "<|",
    r"\[RightAssociation]": "|>",
    # Spacing and formatting
    r"\[IndentingNewLine]": "\n",
    r"\[NewLine]": "\n",
    r"\[InvisibleSpace]": "",
    r"\[VeryThinSpace]": " ",
    r"\[ThinSpace]": " ",
    r"\[MediumSpace]": " ",
    r"\[ThickSpace]": " ",
    r"\[NegativeVeryThinSpace]": "",
    r"\[NegativeThinSpace]": "",
    r"\[NegativeMediumSpace]": "",
    r"\[NegativeThickSpace]": "",
    r"\[NonBreakingSpace]": " ",
    r"\[InvisibleComma]": ",",
    r"\[InvisibleApplication]": "",
    r"\[InvisibleTimes]": "*",
    # Common mathematical constants and functions
    r"\[ExponentialE]": "E",
    r"\[ImaginaryI]": "I",
    r"\[ImaginaryJ]": "I",
    r"\[DifferentialD]": "d",
    r"\[PartialD]": "D",
    r"\[Integral]": "Integrate",
    r"\[Sum]": "Sum",
    r"\[Product]": "Product",
    r"\[Sqrt]": "Sqrt",
    # Subscript/superscript indicators (handled specially in box parsing)
    r"\[UnderBrace]": "",
    r"\[OverBrace]": "",
    # Miscellaneous
    r"\[Null]": "",
    r"\[Continuation]": "",
    r"\[DoubleStruckCapitalC]": "Complexes",
    r"\[DoubleStruckCapitalN]": "Integers",
    r"\[DoubleStruckCapitalQ]": "Rationals",
    r"\[DoubleStruckCapitalR]": "Reals",
    r"\[DoubleStruckCapitalZ]": "Integers",
    # Pattern matching
    r"\[Blank]": "_",
    r"\[BlankSequence]": "__",
    r"\[BlankNullSequence]": "___",
}


def convert_special_chars(text: str) -> str:
    """Convert Mathematica special character codes to readable form."""
    result = text
    for pattern, replacement in SPECIAL_CHARS.items():
        result = result.replace(pattern, replacement)

    remaining = re.findall(r"\\\\?\[([A-Z][A-Za-z]+)\]", result)
    for name in remaining:
        result = result.replace(f"\\[{name}]", name)
        result = result.replace(f"[{name}]", name)

    return result


class BoxDataParser:
    """
    Parser for Mathematica BoxData structures.

    Converts nested box expressions like:
        RowBox[{RowBox[{"Clear", "[", "R", "]"}], ";"}]

    Into readable Wolfram code:
        Clear[R];
    """

    def __init__(self):
        self.depth = 0
        self.max_depth = 100  # Prevent infinite recursion

    def parse(self, content: str) -> str:
        """Parse BoxData content and return clean Wolfram code."""
        self.depth = 0

        # Handle empty content
        if not content or content.strip() == "":
            return ""

        # If it's already clean text (no BoxData), return as-is
        if not any(
            box in content
            for box in [
                "BoxData",
                "RowBox",
                "FractionBox",
                "SuperscriptBox",
                "SubscriptBox",
                "SqrtBox",
                "GridBox",
                "Cell[",
            ]
        ):
            return convert_special_chars(content)

        try:
            if "BoxData[" in content:
                match = re.search(r"BoxData\[(.*)\]", content, re.DOTALL)
                if match:
                    content = match.group(1)

            content = content.strip()
            if content.startswith("{") and content.endswith("}"):
                inner = content[1:-1].strip()
                if inner.startswith("RowBox[") or inner.startswith('"'):
                    elements = self._parse_list_elements(inner)
                    parsed_elements = []
                    for elem in elements:
                        parsed = self._parse_expression(elem.strip())
                        if parsed:
                            parsed_elements.append(parsed)
                    result = "\n".join(parsed_elements)
                    return convert_special_chars(result)

            result = self._parse_expression(content)
            return convert_special_chars(result)
        except Exception as e:
            # If parsing fails, try basic cleanup
            return self._basic_cleanup(content)

    def _parse_expression(self, expr: str) -> str:
        """Recursively parse a Mathematica expression."""
        self.depth += 1
        if self.depth > self.max_depth:
            return expr

        expr = expr.strip()

        try:
            # Handle different box types
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
                # String literal
                return expr[1:-1]
            elif expr.startswith("{") and expr.endswith("}"):
                # List
                return self._parse_list(expr)
            else:
                return expr
        finally:
            self.depth -= 1

    def _extract_box_content(self, expr: str, box_type: str) -> str:
        """Extract the content inside a box expression."""
        prefix = f"{box_type}["
        if not expr.startswith(prefix):
            return expr

        # Find matching bracket
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
        """Parse comma-separated elements from a list, respecting nesting."""
        elements = []
        current = []
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(content):
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

    def _parse_row_box(self, expr: str) -> str:
        """Parse RowBox[{elem1, elem2, ...}]."""
        content = self._extract_box_content(expr, "RowBox")

        if content.startswith("{") and content.endswith("}"):
            content = content[1:-1]

        elements = self._parse_list_elements(content)
        result_parts = []

        for elem in elements:
            elem = elem.strip()
            parsed = self._parse_expression(elem)
            result_parts.append(parsed)

        result = "".join(result_parts)

        result = result.replace("\\n", "\n")

        return result

    def _parse_fraction_box(self, expr: str) -> str:
        """Parse FractionBox[num, denom]."""
        content = self._extract_box_content(expr, "FractionBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            num = self._parse_expression(elements[0])
            denom = self._parse_expression(elements[1])
            # Add parentheses if needed
            if " " in num or "+" in num or "-" in num:
                num = f"({num})"
            if " " in denom or "+" in denom or "-" in denom:
                denom = f"({denom})"
            return f"{num}/{denom}"
        return content

    def _parse_superscript_box(self, expr: str) -> str:
        """Parse SuperscriptBox[base, exp]."""
        content = self._extract_box_content(expr, "SuperscriptBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            exp = self._parse_expression(elements[1])
            # Handle special cases
            if exp == "T":  # Transpose
                return f"Transpose[{base}]"
            elif exp == "-1":  # Inverse
                return f"Inverse[{base}]"
            elif exp == "*":  # Conjugate
                return f"Conjugate[{base}]"
            else:
                if len(base) > 1 and not (base.startswith("(") and base.endswith(")")):
                    if not base[0].isupper():  # Don't parenthesize function names
                        base = f"({base})"
                return f"{base}^{exp}"
        return content

    def _parse_subscript_box(self, expr: str) -> str:
        """Parse SubscriptBox[base, sub]."""
        content = self._extract_box_content(expr, "SubscriptBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            sub = self._parse_expression(elements[1])
            return f"Subscript[{base}, {sub}]"
        return content

    def _parse_subsuperscript_box(self, expr: str) -> str:
        """Parse SubsuperscriptBox[base, sub, super]."""
        content = self._extract_box_content(expr, "SubsuperscriptBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 3:
            base = self._parse_expression(elements[0])
            sub = self._parse_expression(elements[1])
            sup = self._parse_expression(elements[2])
            return f"Subsuperscript[{base}, {sub}, {sup}]"
        return content

    def _parse_sqrt_box(self, expr: str) -> str:
        """Parse SqrtBox[content]."""
        content = self._extract_box_content(expr, "SqrtBox")
        inner = self._parse_expression(content)
        return f"Sqrt[{inner}]"

    def _parse_radical_box(self, expr: str) -> str:
        """Parse RadicalBox[content, n] for n-th root."""
        content = self._extract_box_content(expr, "RadicalBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            inner = self._parse_expression(elements[0])
            n = self._parse_expression(elements[1])
            return f"Power[{inner}, 1/{n}]"
        elif len(elements) == 1:
            inner = self._parse_expression(elements[0])
            return f"Sqrt[{inner}]"
        return content

    def _parse_grid_box(self, expr: str) -> str:
        """Parse GridBox[{{...}, {...}}] as matrix."""
        content = self._extract_box_content(expr, "GridBox")

        # Find the first list argument (the matrix data)
        if content.startswith("{"):
            depth = 0
            end = 0
            for i, char in enumerate(content):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            matrix_content = content[:end]
            return self._parse_matrix(matrix_content)

        return content

    def _parse_matrix(self, content: str) -> str:
        """Parse a matrix {{a, b}, {c, d}}."""
        if not (content.startswith("{") and content.endswith("}")):
            return content

        # Remove outer braces
        inner = content[1:-1].strip()

        # Parse rows
        rows = []
        depth = 0
        current_row = []
        current_elem = []

        for char in inner:
            if char == "{":
                if depth == 0:
                    depth = 1
                else:
                    depth += 1
                    current_elem.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    if current_elem:
                        current_row.append("".join(current_elem).strip())
                    if current_row:
                        rows.append(current_row)
                    current_row = []
                    current_elem = []
                else:
                    current_elem.append(char)
            elif char == "," and depth == 1:
                if current_elem:
                    current_row.append("".join(current_elem).strip())
                current_elem = []
            elif depth > 0:
                current_elem.append(char)

        # Format as Mathematica list
        formatted_rows = []
        for row in rows:
            parsed_elems = [self._parse_expression(e) for e in row]
            formatted_rows.append("{" + ", ".join(parsed_elems) + "}")

        return "{" + ", ".join(formatted_rows) + "}"

    def _parse_form_box(self, expr: str) -> str:
        """Parse FormBox[content, form]."""
        content = self._extract_box_content(expr, "FormBox")
        elements = self._parse_list_elements(content)

        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_tag_box(self, expr: str) -> str:
        """Parse TagBox[content, tag]."""
        content = self._extract_box_content(expr, "TagBox")
        elements = self._parse_list_elements(content)

        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_style_box(self, expr: str) -> str:
        """Parse StyleBox[content, styles...]."""
        content = self._extract_box_content(expr, "StyleBox")
        elements = self._parse_list_elements(content)

        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_interpretation_box(self, expr: str) -> str:
        """Parse InterpretationBox[display, interpretation]."""
        content = self._extract_box_content(expr, "InterpretationBox")
        elements = self._parse_list_elements(content)

        # Prefer the interpretation (second element) over display
        if len(elements) >= 2:
            return self._parse_expression(elements[1])
        elif elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_template_box(self, expr: str) -> str:
        """Parse TemplateBox[{args...}, template_name]."""
        content = self._extract_box_content(expr, "TemplateBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            args_str = elements[0]
            template = elements[1].strip('"')

            # Parse arguments
            if args_str.startswith("{") and args_str.endswith("}"):
                args = self._parse_list_elements(args_str[1:-1])
                parsed_args = [self._parse_expression(a) for a in args]

                # Common templates
                if template == "Times":
                    return " * ".join(parsed_args)
                elif template == "Divide":
                    if len(parsed_args) >= 2:
                        return f"({parsed_args[0]})/({parsed_args[1]})"

                return ", ".join(parsed_args)

        if elements:
            return self._parse_expression(elements[0])
        return content

    def _parse_underscript_box(self, expr: str) -> str:
        """Parse UnderscriptBox[base, under]."""
        content = self._extract_box_content(expr, "UnderscriptBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            under = self._parse_expression(elements[1])
            return f"Underscript[{base}, {under}]"
        return content

    def _parse_overscript_box(self, expr: str) -> str:
        """Parse OverscriptBox[base, over]."""
        content = self._extract_box_content(expr, "OverscriptBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 2:
            base = self._parse_expression(elements[0])
            over = self._parse_expression(elements[1])
            return f"Overscript[{base}, {over}]"
        return content

    def _parse_underoverscript_box(self, expr: str) -> str:
        """Parse UnderoverscriptBox[base, under, over]."""
        content = self._extract_box_content(expr, "UnderoverscriptBox")
        elements = self._parse_list_elements(content)

        if len(elements) >= 3:
            base = self._parse_expression(elements[0])
            under = self._parse_expression(elements[1])
            over = self._parse_expression(elements[2])
            return f"Underoverscript[{base}, {under}, {over}]"
        return content

    def _parse_list(self, expr: str) -> str:
        """Parse a list {...}."""
        if not (expr.startswith("{") and expr.endswith("}")):
            return expr

        content = expr[1:-1]
        elements = self._parse_list_elements(content)
        parsed = [self._parse_expression(e) for e in elements]
        return "{" + ", ".join(parsed) + "}"

    def _basic_cleanup(self, content: str) -> str:
        """Fallback cleanup when parsing fails."""
        # Remove box type wrappers
        result = content
        for box_type in [
            "BoxData",
            "RowBox",
            "Cell",
            "FormBox",
            "TagBox",
            "StyleBox",
            "InterpretationBox",
        ]:
            result = re.sub(rf"{box_type}\[", "", result)

        # Balance brackets
        open_brackets = result.count("[") - result.count("]")
        if open_brackets > 0:
            result += "]" * open_brackets

        open_braces = result.count("{") - result.count("}")
        if open_braces > 0:
            result += "}" * open_braces

        # Clean up quotes and whitespace
        result = result.replace('", "', ", ")
        result = result.replace('","', ", ")
        result = re.sub(r'"\s*,\s*"', ", ", result)

        return convert_special_chars(result)


class NotebookParser:
    """
    Parser for Mathematica notebook (.nb) files.

    Usage:
        parser = NotebookParser()
        notebook = parser.parse_file("path/to/notebook.nb")

        # Get all code
        code = notebook.get_code_cells()

        # Get outline
        outline = notebook.get_outline()
    """

    def __init__(self):
        self.box_parser = BoxDataParser()

    def parse_file(self, path: str | Path) -> NotebookStructure:
        """Parse a notebook file and return structured content."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_content(content, str(path))

    def parse_content(self, content: str, path: str = "") -> NotebookStructure:
        """Parse notebook content string."""
        notebook = NotebookStructure(path=path)

        # Extract title if present
        title_match = re.search(r'Cell\["([^"]+)",\s*"Title"', content)
        if title_match:
            notebook.title = title_match.group(1)

        # Find all cells
        cells = self._extract_cells(content)
        notebook.cells = cells

        return notebook

    def _extract_cells(self, content: str) -> list[Cell]:
        """Extract all cells from notebook content."""
        cells = []
        cell_index = 0

        i = 0
        while i < len(content):
            cell_start = content.find("Cell[", i)
            if cell_start == -1:
                break

            next_after_cell = cell_start + 5

            if content[next_after_cell:].startswith("CellGroupData"):
                i = next_after_cell
                continue

            depth = 0
            j = cell_start
            in_string = False
            escape_next = False

            while j < len(content):
                char = content[j]

                if escape_next:
                    escape_next = False
                    j += 1
                    continue

                if char == "\\" and in_string:
                    escape_next = True
                    j += 1
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    j += 1
                    continue

                if not in_string:
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1

            if j >= len(content):
                break

            cell_text = content[cell_start : j + 1]
            cell = self._parse_cell(cell_text, cell_index)

            if cell:
                cells.append(cell)
                cell_index += 1

            i = j + 1

        return cells

    def _parse_cell(self, cell_text: str, index: int) -> Cell | None:
        """Parse a single Cell[...] construct."""
        # Skip CellGroupData, CellChangeTimes, etc.
        if cell_text.startswith("Cell[CellGroupData"):
            return None
        if not cell_text.startswith("Cell["):
            return None

        # Determine cell style
        style = self._detect_style(cell_text)
        if style == CellStyle.UNKNOWN:
            # Skip unknown cell types
            return None

        # Extract cell label if present
        label = ""
        label_match = re.search(r'CellLabel\s*->\s*"([^"]+)"', cell_text)
        if label_match:
            label = label_match.group(1)

        # Extract content
        raw_content = self._extract_cell_content(cell_text)

        # Parse BoxData if present
        if "BoxData[" in raw_content:
            parsed_content = self.box_parser.parse(raw_content)
        elif style in (
            CellStyle.TEXT,
            CellStyle.TITLE,
            CellStyle.SECTION,
            CellStyle.SUBSECTION,
            CellStyle.SUBSUBSECTION,
            CellStyle.CHAPTER,
        ):
            # Text cells often have simple string content
            parsed_content = self._extract_text_content(raw_content)
        else:
            parsed_content = convert_special_chars(raw_content)

        return Cell(
            style=style,
            content=parsed_content,
            raw_content=raw_content,
            cell_label=label,
            cell_index=index,
        )

    def _detect_style(self, cell_text: str) -> CellStyle:
        """Detect the style of a cell."""
        # Look for style specification after content
        # Cell[content, "StyleName", ...]

        style_patterns = [
            (r',\s*"Title"[,\]]', CellStyle.TITLE),
            (r',\s*"Chapter"[,\]]', CellStyle.CHAPTER),
            (r',\s*"Section"[,\]]', CellStyle.SECTION),
            (r',\s*"Subsection"[,\]]', CellStyle.SUBSECTION),
            (r',\s*"Subsubsection"[,\]]', CellStyle.SUBSUBSECTION),
            (r',\s*"Text"[,\]]', CellStyle.TEXT),
            (r',\s*"Input"[,\]]', CellStyle.INPUT),
            (r',\s*"Output"[,\]]', CellStyle.OUTPUT),
            (r',\s*"Code"[,\]]', CellStyle.CODE),
            (r',\s*"Message"[,\]]', CellStyle.MESSAGE),
            (r',\s*"Print"[,\]]', CellStyle.PRINT),
            (r',\s*"Item"[,\]]', CellStyle.ITEM),
            (r',\s*"ItemNumbered"[,\]]', CellStyle.ITEM_NUMBERED),
        ]

        for pattern, style in style_patterns:
            if re.search(pattern, cell_text):
                return style

        return CellStyle.UNKNOWN

    def _extract_cell_content(self, cell_text: str) -> str:
        """Extract the content portion of a Cell[content, style, ...]."""
        # Cell[content, "Style", ...]
        # We need to find where content ends

        if not cell_text.startswith("Cell["):
            return cell_text

        # Start after "Cell["
        start = 5
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(cell_text)):
            char = cell_text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth -= 1
            elif char == "," and depth == 0:
                # Found the end of content
                return cell_text[start:i].strip()

        # If no comma found, take everything except the closing bracket
        return (
            cell_text[start:-1].strip()
            if cell_text.endswith("]")
            else cell_text[start:]
        )

    def _extract_text_content(self, content: str) -> str:
        """Extract plain text from a text cell content."""
        # Text cells often have format: "Some text here"
        if content.startswith('"') and content.endswith('"'):
            return content[1:-1]

        # Or TextData[{...}]
        if content.startswith("TextData["):
            inner = self._extract_box_like_content(content, "TextData")
            return self._parse_text_data(inner)

        return convert_special_chars(content)

    def _extract_box_like_content(self, content: str, box_type: str) -> str:
        """Extract content from BoxType[content]."""
        prefix = f"{box_type}["
        if not content.startswith(prefix):
            return content

        depth = 0
        for i, char in enumerate(content):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return content[len(prefix) : i]

        return (
            content[len(prefix) : -1]
            if content.endswith("]")
            else content[len(prefix) :]
        )

    def _parse_text_data(self, content: str) -> str:
        """Parse TextData content."""
        # TextData is usually a list of strings and StyleBox elements
        result = []

        # Simple extraction: find all quoted strings
        strings = re.findall(r'"([^"]*)"', content)
        for s in strings:
            result.append(s)

        return "".join(result)

    def to_markdown(self, notebook: NotebookStructure) -> str:
        """Convert notebook structure to Markdown."""
        lines = []

        if notebook.title:
            lines.append(f"# {notebook.title}\n")

        for cell in notebook.cells:
            if cell.style == CellStyle.TITLE:
                lines.append(f"# {cell.content}\n")
            elif cell.style == CellStyle.CHAPTER:
                lines.append(f"# {cell.content}\n")
            elif cell.style == CellStyle.SECTION:
                lines.append(f"## {cell.content}\n")
            elif cell.style == CellStyle.SUBSECTION:
                lines.append(f"### {cell.content}\n")
            elif cell.style == CellStyle.SUBSUBSECTION:
                lines.append(f"#### {cell.content}\n")
            elif cell.style == CellStyle.TEXT:
                lines.append(f"{cell.content}\n")
            elif cell.style == CellStyle.ITEM:
                lines.append(f"- {cell.content}\n")
            elif cell.style == CellStyle.ITEM_NUMBERED:
                lines.append(f"1. {cell.content}\n")
            elif cell.style in (CellStyle.INPUT, CellStyle.CODE):
                label = f"  (* {cell.cell_label} *)" if cell.cell_label else ""
                lines.append(f"```wolfram{label}\n{cell.content}\n```\n")
            elif cell.style == CellStyle.OUTPUT:
                lines.append(f"```\n(* Output *)\n{cell.content}\n```\n")

        return "\n".join(lines)

    def to_wolfram_code(self, notebook: NotebookStructure) -> str:
        """Extract only executable Wolfram code from notebook."""
        code_parts = []

        for cell in notebook.cells:
            if cell.is_code():
                # Add cell label as comment
                if cell.cell_label:
                    code_parts.append(f"(* {cell.cell_label} *)")
                code_parts.append(cell.content)
                code_parts.append("")  # Blank line between cells

        return "\n".join(code_parts)

    def to_latex(self, notebook: NotebookStructure) -> str:
        """Convert notebook to LaTeX format."""
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{listings}",
            r"\lstset{language=Mathematica,basicstyle=\ttfamily\small}",
            r"\begin{document}",
            "",
        ]

        if notebook.title:
            lines.append(f"\\title{{{notebook.title}}}")
            lines.append(r"\maketitle")
            lines.append("")

        for cell in notebook.cells:
            if cell.style == CellStyle.TITLE:
                lines.append(f"\\title{{{cell.content}}}")
            elif cell.style == CellStyle.SECTION:
                lines.append(f"\\section{{{cell.content}}}")
            elif cell.style == CellStyle.SUBSECTION:
                lines.append(f"\\subsection{{{cell.content}}}")
            elif cell.style == CellStyle.SUBSUBSECTION:
                lines.append(f"\\subsubsection{{{cell.content}}}")
            elif cell.style == CellStyle.TEXT:
                lines.append(f"{cell.content}\n")
            elif cell.style in (CellStyle.INPUT, CellStyle.CODE):
                lines.append(r"\begin{lstlisting}")
                lines.append(cell.content)
                lines.append(r"\end{lstlisting}")
                lines.append("")

        lines.append(r"\end{document}")
        return "\n".join(lines)


# Convenience functions
def parse_notebook(path: str | Path) -> NotebookStructure:
    """Parse a notebook file and return its structure."""
    parser = NotebookParser()
    return parser.parse_file(path)


def notebook_to_markdown(path: str | Path) -> str:
    """Convert a notebook to Markdown."""
    parser = NotebookParser()
    notebook = parser.parse_file(path)
    return parser.to_markdown(notebook)


def notebook_to_wolfram(path: str | Path) -> str:
    """Extract Wolfram code from a notebook."""
    parser = NotebookParser()
    notebook = parser.parse_file(path)
    return parser.to_wolfram_code(notebook)


def parse_boxdata(content: str) -> str:
    """Parse BoxData content and return clean Wolfram code."""
    parser = BoxDataParser()
    return parser.parse(content)


if __name__ == "__main__":
    # Test with command line argument
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Parsing: {path}")
        print("-" * 60)

        notebook = parse_notebook(path)

        print(f"Title: {notebook.title}")
        print(f"Total cells: {len(notebook.cells)}")
        print(f"Code cells: {len(notebook.get_code_cells())}")
        print(f"Doc cells: {len(notebook.get_documentation_cells())}")
        print()
        print("Outline:")
        for item in notebook.get_outline():
            indent = "  " * (
                ["Title", "Chapter", "Section", "Subsection", "Subsubsection"].index(
                    item["level"]
                )
                if item["level"]
                in ["Title", "Chapter", "Section", "Subsection", "Subsubsection"]
                else 0
            )
            print(f"{indent}{item['level']}: {item['title']}")
        print()
        print("=" * 60)
        print("MARKDOWN OUTPUT:")
        print("=" * 60)
        parser = NotebookParser()
        print(parser.to_markdown(notebook))
    else:
        print("Usage: python notebook_parser.py <path-to-notebook.nb>")
