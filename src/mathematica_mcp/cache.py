import functools
import hashlib
import re
import time
from collections import OrderedDict, namedtuple
from dataclasses import dataclass, field
from typing import Any

from .config import FEATURES
from .wl_scan import scan_clean

# ---------------------------------------------------------------------------
# Code analysis — memoized, single-pass
# ---------------------------------------------------------------------------

_CodeAnalysis = namedtuple("_CodeAnalysis", ["user_symbols", "session_sensitive"])

# Top ~200 System` builtins — static, no symbol_index on hot path
_SYSTEM_BUILTINS = frozenset(
    {
        # Core language
        "If",
        "Which",
        "Switch",
        "Module",
        "Block",
        "With",
        "Do",
        "For",
        "While",
        "Return",
        "Break",
        "Continue",
        "Throw",
        "Catch",
        "Goto",
        "Label",
        "CompoundExpression",
        # Constants
        "True",
        "False",
        "Null",
        "None",
        "All",
        "Automatic",
        "Full",
        "Infinity",
        "Pi",
        "E",
        "I",
        "Degree",
        "GoldenRatio",
        "EulerGamma",
        # Math functions
        "Sin",
        "Cos",
        "Tan",
        "ArcSin",
        "ArcCos",
        "ArcTan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Exp",
        "Log",
        "Log2",
        "Log10",
        "Sqrt",
        "Abs",
        "Sign",
        "Floor",
        "Ceiling",
        "Round",
        "Max",
        "Min",
        "Mod",
        "Power",
        "Plus",
        "Times",
        "Subtract",
        "Divide",
        "Factorial",
        "Binomial",
        "Gamma",
        "Beta",
        # Algebra
        "Integrate",
        "Sum",
        "Product",
        "D",
        "Limit",
        "Series",
        "Residue",
        "Solve",
        "DSolve",
        "RSolve",
        "Reduce",
        "Eliminate",
        "Roots",
        "Simplify",
        "FullSimplify",
        "Expand",
        "Factor",
        "Apart",
        "Together",
        "Cancel",
        "Collect",
        "CoefficientList",
        "Coefficient",
        # Numeric
        "N",
        "NIntegrate",
        "NSolve",
        "NDSolve",
        "FindRoot",
        "NMinimize",
        "NMaximize",
        "LinearSolve",
        "Eigenvalues",
        "Eigenvectors",
        "SingularValueDecomposition",
        # List/functional
        "List",
        "Table",
        "Range",
        "Array",
        "Map",
        "Apply",
        "Select",
        "Cases",
        "Sort",
        "SortBy",
        "Reverse",
        "Length",
        "Dimensions",
        "Flatten",
        "Partition",
        "Take",
        "Drop",
        "Part",
        "First",
        "Last",
        "Rest",
        "Most",
        "Append",
        "Prepend",
        "Join",
        "Union",
        "Intersection",
        "Complement",
        "DeleteDuplicates",
        "Position",
        "MemberQ",
        "FreeQ",
        "Count",
        "Total",
        "Mean",
        "Median",
        "Transpose",
        "Dot",
        "Cross",
        "Norm",
        "Normalize",
        "Fold",
        "FoldList",
        "NestList",
        "Nest",
        "FixedPoint",
        "Through",
        "Association",
        "AssociationThread",
        "KeyValueMap",
        "Keys",
        "Values",
        "Lookup",
        # Patterns
        "Pattern",
        "Blank",
        "BlankSequence",
        "BlankNullSequence",
        "Alternatives",
        "Repeated",
        "Condition",
        "PatternTest",
        "Rule",
        "RuleDelayed",
        "Set",
        "SetDelayed",
        "Unset",
        "MatchQ",
        "ReplaceAll",
        "ReplaceRepeated",
        # Strings
        "StringQ",
        "StringJoin",
        "StringLength",
        "StringTake",
        "StringDrop",
        "StringReplace",
        "StringCases",
        "StringSplit",
        "ToString",
        "ToExpression",
        # I/O
        "Print",
        "Echo",
        "Export",
        "Import",
        "Get",
        "Put",
        "ReadList",
        "FilePrint",
        "FileExistsQ",
        "DirectoryQ",
        # Graphics
        "Plot",
        "Plot3D",
        "ListPlot",
        "ListLinePlot",
        "ListLogPlot",
        "ContourPlot",
        "DensityPlot",
        "ParametricPlot",
        "RegionPlot",
        "Graphics",
        "Graphics3D",
        "Show",
        "GraphicsRow",
        "GraphicsColumn",
        "BarChart",
        "PieChart",
        "Histogram",
        "Point",
        "Line",
        "Circle",
        "Disk",
        "Rectangle",
        "Polygon",
        "Arrow",
        "Text",
        "Inset",
        "Tooltip",
        "Labeled",
        "RGBColor",
        "Hue",
        "Red",
        "Blue",
        "Green",
        "Black",
        "White",
        "Gray",
        "Thick",
        "Thin",
        "Dashed",
        "Dotted",
        "Opacity",
        "Rasterize",
        "Image",
        "ImageSize",
        # Dynamic
        "Manipulate",
        "Dynamic",
        "DynamicModule",
        "Animate",
        "Slider",
        # Format
        "InputForm",
        "FullForm",
        "TeXForm",
        "MatrixForm",
        "TableForm",
        "NumberForm",
        "ScientificForm",
        # Type checking
        "Head",
        "IntegerQ",
        "NumericQ",
        "ListQ",
        "AtomQ",
        "VectorQ",
        "MatrixQ",
        "NumberQ",
        "EvenQ",
        "OddQ",
        "PrimeQ",
        "PositiveQ",
        # Special
        "Quiet",
        "Check",
        "AbortProtect",
        "TimeConstrained",
        "MemoryConstrained",
        "Timing",
        "AbsoluteTiming",
        "Pause",
        "Random",
        "RandomReal",
        "RandomInteger",
        "RandomChoice",
        "RandomSample",
        "SeedRandom",
        "BlockRandom",
        # Entity/knowledge
        "Entity",
        "EntityValue",
        "WolframAlpha",
        "Interpreter",
        # Data
        "Dataset",
        "Query",
        "GroupBy",
        "CountsBy",
    }
)

# Session-sensitive symbols — result changes across kernel mutations even
# though no user symbol appears in the expression
_SESSION_SENSITIVE = frozenset(
    {
        "Names",
        "Contexts",
        "Options",
        "Attributes",
        "Definition",
        "OwnValues",
        "DownValues",
        "SubValues",
        "UpValues",
        "FormatValues",
        "Messages",
        "Information",
        "Symbol",
    }
)

_IDENTIFIER_RE = re.compile(r"\b(\$?[a-zA-Z][a-zA-Z0-9$]*)\b")


@functools.lru_cache(maxsize=1024)
def _analyze_code(normalized_code: str) -> _CodeAnalysis:
    """Analyze Wolfram code for user symbols and session-sensitive references.

    Memoized by normalized code string for hot-path efficiency.
    Uses wl_scan for string/comment stripping.
    """
    scan = scan_clean(normalized_code)
    if not scan.ok:
        # Malformed input → treat as epoch-sensitive (safe degradation)
        return _CodeAnalysis(user_symbols=frozenset({"__malformed__"}), session_sensitive=True)

    cleaned = scan.cleaned

    # Check session-sensitive symbols
    session_sensitive = False
    for sym in _SESSION_SENSITIVE:
        if re.search(rf"\b{sym}\b", cleaned):
            session_sensitive = True
            break
    # Also check $-prefixed globals
    if not session_sensitive and re.search(r"\$[A-Z][a-zA-Z]+", cleaned):
        session_sensitive = True

    # Extract identifiers and filter out system builtins
    all_ids = set(_IDENTIFIER_RE.findall(cleaned))
    user_symbols = frozenset(all_ids - _SYSTEM_BUILTINS - _SESSION_SENSITIVE)

    return _CodeAnalysis(user_symbols=user_symbols, session_sensitive=session_sensitive)


def _is_epoch_insensitive(code: str) -> bool:
    """Return True if an expression is safe to cache across epoch bumps.

    An expression is epoch-insensitive only when:
    - wl_scan reports well-formed code (ok=True)
    - no user-defined symbols are referenced
    - no session-sensitive system symbols are referenced
    """
    analysis = _analyze_code(" ".join(code.split()))
    return len(analysis.user_symbols) == 0 and not analysis.session_sensitive


# ---------------------------------------------------------------------------
# Expression cache
# ---------------------------------------------------------------------------


@dataclass
class CachedExpression:
    name: str
    expression: str
    result: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


_expression_cache: dict[str, CachedExpression] = {}

# Kernel state epoch – monotonically increasing counter included in query
# cache keys.  Bumped by any operation that mutates kernel state (e.g.
# set_variable, clear_variables, restart_kernel, load_package, run_script).
# After a bump, existing cache entries become unreachable by key, ensuring
# no stale results are returned.
_kernel_epoch: int = 0


def bump_kernel_epoch() -> int:
    """Increment the kernel epoch, logically invalidating all query cache entries."""
    global _kernel_epoch
    _kernel_epoch += 1
    return _kernel_epoch


def get_kernel_epoch() -> int:
    """Return the current kernel state epoch."""
    return _kernel_epoch


# ---------------------------------------------------------------------------
# Notebook mutation epoch + opt-in screenshot cache
# ---------------------------------------------------------------------------

# Addon commands that can change VISIBLE notebook content. Mirrors the
# notebook-mutating subset of $MCPStateDeltaCommands in addon/MathematicaMCP.wl.
# Excluded on purpose: reads/navigation (get_notebooks, get_notebook_info,
# get_cells, select_cell, scroll_to_cell), because screenshot_notebook
# rasterizes the whole notebook so selection/scroll position never changes the
# PNG; and kernel-only commands (execute_code, set_variable, run_script,
# restart) that never touch a notebook window. batch_commands stays in: its
# sub-commands may mutate. Over-bumping only costs a cache miss; under-bumping
# serves stale pixels, so ambiguous cases (save_notebook) are included.
MUTATING_COMMANDS = frozenset(
    {
        "create_notebook",
        "close_notebook",
        "save_notebook",
        "open_notebook_file",
        "write_cell",
        "delete_cell",
        "evaluate_cell",
        "execute_code_notebook",
        "execute_selection",
        "batch_commands",
    }
)

# Monotonic counter bumped by connection.send_command after any successful
# MUTATING_COMMANDS response. Folded into screenshot cache keys so a mutation
# makes every prior entry unreachable (same trick as _kernel_epoch above).
_notebook_epoch: int = 0


def bump_notebook_epoch() -> int:
    """Increment the notebook mutation epoch, invalidating cached screenshots."""
    global _notebook_epoch
    _notebook_epoch += 1
    return _notebook_epoch


def get_notebook_epoch() -> int:
    """Return the current notebook mutation epoch."""
    return _notebook_epoch


_MAX_SCREENSHOT_ENTRIES = 8
_screenshot_cache: OrderedDict[tuple, bytes] = OrderedDict()


def get_cached_screenshot(key: tuple) -> bytes | None:
    """Return cached PNG bytes for key (LRU-touch on hit), or None."""
    png = _screenshot_cache.get(key)
    if png is not None:
        _screenshot_cache.move_to_end(key)
    return png


def put_cached_screenshot(key: tuple, png: bytes) -> None:
    """Store PNG bytes under key, evicting the oldest entry past capacity."""
    _screenshot_cache[key] = png
    _screenshot_cache.move_to_end(key)
    while len(_screenshot_cache) > _MAX_SCREENSHOT_ENTRIES:
        _screenshot_cache.popitem(last=False)


NON_CACHEABLE_PATTERNS = [
    "Random",
    "Now",
    "Date",
    "AbsoluteTime",
    "SessionTime",
    "$Line",
    "Dynamic",
    "Button",
    "Manipulate",
    "CurrentValue",
]


class QueryCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash_code(
        self,
        code: str,
        output_format: str,
        render_graphics: bool,
        deterministic_seed: int | None,
        context_key: str | None,
    ) -> str:
        normalized = " ".join(code.split())
        # Epoch-insensitive expressions (pure System, no user symbols, no session-sensitive)
        # omit epoch from hash key so they survive kernel state mutations
        epoch_part = "" if _is_epoch_insensitive(code) else f"|epoch={_kernel_epoch}"
        options = (
            f"|fmt={output_format}|gfx={int(render_graphics)}|seed={deterministic_seed}|ctx={context_key}{epoch_part}"
        )
        return hashlib.sha256((normalized + options).encode()).hexdigest()[:16]

    def _is_cacheable(self, code: str) -> bool:
        return not any(pattern in code for pattern in NON_CACHEABLE_PATTERNS)

    def get(
        self,
        code: str,
        *,
        output_format: str = "text",
        render_graphics: bool = True,
        deterministic_seed: int | None = None,
        context_key: str | None = None,
    ) -> dict[str, Any] | None:
        if not FEATURES.expression_cache or not self._is_cacheable(code):
            return None

        key = self._hash_code(code, output_format, render_graphics, deterministic_seed, context_key)
        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]
        if time.time() - entry["created_at"] > self.ttl:
            del self._cache[key]
            self.misses += 1
            return None

        self._cache.move_to_end(key)
        entry["access_count"] = entry.get("access_count", 0) + 1
        self.hits += 1
        return entry["result"]

    def put(
        self,
        code: str,
        result: dict[str, Any],
        *,
        output_format: str = "text",
        render_graphics: bool = True,
        deterministic_seed: int | None = None,
        context_key: str | None = None,
    ) -> None:
        if not FEATURES.expression_cache or not self._is_cacheable(code):
            return

        key = self._hash_code(code, output_format, render_graphics, deterministic_seed, context_key)
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "code": code,
            "result": result,
            "created_at": time.time(),
            "access_count": 0,
        }

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0


_query_cache = QueryCache()


def cache_expression(name: str, expression: str, result: str) -> bool:
    if not FEATURES.expression_cache:
        return False

    _expression_cache[name] = CachedExpression(
        name=name,
        expression=expression,
        result=result,
        created_at=time.time(),
    )
    return True


def get_cached_expression(name: str) -> CachedExpression | None:
    if not FEATURES.expression_cache:
        return None

    cached = _expression_cache.get(name)
    if cached:
        cached.access_count += 1
        cached.last_accessed = time.time()
    return cached


def list_cached_expressions() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "expression": item.expression,
            "result_preview": item.result[:100] + "..." if len(item.result) > 100 else item.result,
            "access_count": item.access_count,
            "age_seconds": int(time.time() - item.created_at),
        }
        for name, item in _expression_cache.items()
    }


def clear_cache():
    global _expression_cache
    _expression_cache = {}


def remove_cached_expression(name: str) -> bool:
    if name in _expression_cache:
        del _expression_cache[name]
        return True
    return False
