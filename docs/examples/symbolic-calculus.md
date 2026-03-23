# Example: Symbolic Calculus

## Scenario

A researcher asks their AI agent to integrate a function, differentiate the result, and verify the derivation step-by-step.

## Agent Conversation

**User:** Compute the indefinite integral of x^2 * sin(x), then verify by differentiating.

**Agent:** I'll compute the integral first.

```
Tool: execute_code
Input: code = "Integrate[x^2 Sin[x], x]"

Result:
{
  "success": true,
  "output": "2 x Sin[x] - (-2 + x^2) Cos[x]",
  "output_tex": "2 x \\sin (x)-\\left(x^2-2\\right) \\cos (x)"
}
```

**Agent:** The integral is `2x Sin[x] - (x^2 - 2) Cos[x]`. Let me verify by differentiating.

```
Tool: execute_code
Input: code = "D[2 x Sin[x] - (-2 + x^2) Cos[x], x]"

Result:
{
  "success": true,
  "output": "x^2 Sin[x]",
  "output_tex": "x^2 \\sin (x)"
}
```

**Agent:** Differentiating gives back `x^2 Sin[x]`, confirming the integral. Now let me run the formal verification.

```
Tool: verify_derivation
Input:
  steps = ["Integrate[x^2 Sin[x], x]", "2 x Sin[x] - (-2 + x^2) Cos[x]"]

Result:
{
  "success": true,
  "report": "Step 1 → 2: ✓ VALID\n  From: Integrate[x^2 Sin[x], x]\n  To:   2 x Sin[x] - (-2 + x^2) Cos[x]\n\n**Summary**: All steps are valid!\nValid: 1/1 steps",
  "raw_data": {"all_valid": true, "valid_count": 1, "total_steps": 1, "steps": [...]},
  "format": "text"
}
```

**Agent:** Verified. The indefinite integral of x^2 sin(x) is:

**2x sin(x) - (x^2 - 2) cos(x) + C**

## Tools Used

| Tool | Purpose |
|------|---------|
| `execute_code` | Compute the integral and verify via differentiation |
| `verify_derivation` | Formally verify the claimed result matches |

## Profile Required

`math` (or higher)
