# Example: Notebook Analysis

## Scenario

A user wants to understand the structure and content of an existing Mathematica notebook without opening the Mathematica GUI.

## Agent Conversation

**User:** What's in the SinPlot.nb notebook? Show me the outline first, then the actual code.

**Agent:** I'll get the structural outline first.

```
Tool: read_notebook
Input:
  path = "SinPlot.nb"
  output_format = "outline"

Result:
{
  "success": true,
  "format": "outline",
  "backend": "python_syntax",
  "path": "/path/to/SinPlot.nb",
  "section_count": 0,
  "sections": []
}
```

**Agent:** The outline shows no named sections — it's a flat notebook without section headings. Let me extract the cells directly.

```
Tool: read_notebook
Input:
  path = "SinPlot.nb"
  output_format = "json"

Result:
{
  "success": true,
  "path": "/path/to/SinPlot.nb",
  "backend": "python_syntax",
  "cell_count": 2,
  "code_cells": 1,
  "cells": [
    {
      "index": 0,
      "style": "Input",
      "content": "Plot[Sin[x], {x, 0, 2 Pi}]"
    },
    {
      "index": 1,
      "style": "Output",
      "content": "[Graphics output]"
    }
  ]
}
```

**Agent:** The notebook contains a single computation:

- **Input cell:** `Plot[Sin[x], {x, 0, 2 Pi}]` — plots the sine function from 0 to 2pi
- **Output cell:** The rendered plot (graphics output)

**User:** Can I see it as plain Wolfram code?

**Agent:**

```
Tool: read_notebook
Input:
  path = "SinPlot.nb"
  output_format = "wolfram"

Result:
{
  "success": true,
  "format": "wolfram",
  "backend": "python_syntax",
  "path": "/path/to/SinPlot.nb",
  "code_cells": 1,
  "content": "Plot[Sin[x], {x, 0, 2 Pi}]"
}
```

**Agent:** Here's the Wolfram code extracted from the notebook (Input/Code cells only):

```
Plot[Sin[x], {x, 0, 2 Pi}]
```

## Tools Used

| Tool | Purpose |
|------|---------|
| `read_notebook` | Read notebook in outline, JSON, and Wolfram formats |

## Profile Required

`notebook` (or `full`)

## Notes

- `read_notebook` is the consolidated reading tool that supports multiple output formats: `markdown`, `wolfram`, `outline`, `json`, `plain`
- The `outline` format returns only section/subsection headings — flat notebooks with no headings will have `section_count: 0`
- The `wolfram` format extracts only Input/Code cells, not Output cells
- This works **offline** (no running Mathematica needed) when using the Python parsing backend
- For notebooks requiring full semantic fidelity, the kernel backend can be used instead
