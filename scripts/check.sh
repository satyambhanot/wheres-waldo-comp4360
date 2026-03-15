#!/bin/bash
# =============================================================================
# Code Quality Checker — Where's Waldo (COMP 4360)
# Run before every commit: ./scripts/check.sh
# Run with --fix to auto-fix: ./scripts/check.sh --fix
# =============================================================================

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FIX_MODE=false
if [ "$1" = "--fix" ]; then
    FIX_MODE=true
fi

PASS=0
FAIL=0
WARN=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; WARN=$((WARN + 1)); }

echo ""
echo "========================================"
echo "  Code Quality Check"
echo "========================================"
echo ""

# ----- 1. BLACK -----
echo "1. Black (code formatting)"
if [ "$FIX_MODE" = true ]; then
    black src/ 2>/dev/null && pass "Formatted with black" || fail "Black failed"
else
    if black --check src/ 2>/dev/null; then
        pass "All files formatted correctly"
    else
        fail "Files need formatting — run ./scripts/check.sh --fix"
    fi
fi

# ----- 2. ISORT -----
echo "2. isort (import ordering)"
if [ "$FIX_MODE" = true ]; then
    isort src/ 2>/dev/null && pass "Imports sorted" || fail "isort failed"
else
    if isort --check-only src/ 2>/dev/null; then
        pass "Imports ordered correctly"
    else
        fail "Imports need sorting — run ./scripts/check.sh --fix"
    fi
fi

# ----- 3. FLAKE8 -----
echo "3. flake8 (linting)"
FLAKE_OUTPUT=$(flake8 src/ 2>/dev/null || true)
if [ -z "$FLAKE_OUTPUT" ]; then
    pass "No linting errors"
else
    fail "Linting errors found:"
    echo "$FLAKE_OUTPUT" | head -20
fi

# ----- 4. MYPY -----
echo "4. mypy (type checking)"
MYPY_OUTPUT=$(mypy src/ --ignore-missing-imports 2>/dev/null || true)
if echo "$MYPY_OUTPUT" | grep -q "no issues found"; then
    pass "No type errors"
else
    ERRORS=$(echo "$MYPY_OUTPUT" | grep "error" | wc -l | tr -d ' ')
    if [ "$ERRORS" -gt 0 ]; then
        fail "Type errors found ($ERRORS):"
        echo "$MYPY_OUTPUT" | grep "error" | head -10
    else
        pass "No type errors"
    fi
fi

# ----- 5. DOCSTRINGS -----
echo "5. Docstring coverage"
MISSING_DOCSTRINGS=$(python3 << 'PYEOF'
import ast
from pathlib import Path

missing = []
for py_file in sorted(Path("src").rglob("*.py")):
    if py_file.name == "__init__.py":
        continue
    try:
        tree = ast.parse(py_file.read_text())
    except SyntaxError:
        continue
    if not ast.get_docstring(tree):
        missing.append(f"  {py_file}: module docstring missing")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                name = node.name
                if name.startswith("_") and name != "__init__":
                    continue
                missing.append(f"  {py_file}:{node.lineno}: '{name}' missing docstring")
if missing:
    print("\n".join(missing))
else:
    print("ALL_GOOD")
PYEOF
)
if [ "$MISSING_DOCSTRINGS" = "ALL_GOOD" ]; then
    pass "All public functions and classes have docstrings"
else
    fail "Missing docstrings:"
    echo "$MISSING_DOCSTRINGS" | head -15
fi

# ----- 6. TYPE HINTS -----
echo "6. Type hint coverage"
MISSING_HINTS=$(python3 << 'PYEOF'
import ast
from pathlib import Path

missing = []
for py_file in sorted(Path("src").rglob("*.py")):
    if py_file.name == "__init__.py":
        continue
    try:
        tree = ast.parse(py_file.read_text())
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("_") and name != "__init__":
                continue
            if node.returns is None and name != "__init__":
                missing.append(f"  {py_file}:{node.lineno}: '{name}' missing return type")
            for arg in node.args.args:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is None:
                    missing.append(f"  {py_file}:{node.lineno}: '{name}' param '{arg.arg}' missing type")
if missing:
    print("\n".join(missing))
else:
    print("ALL_GOOD")
PYEOF
)
if [ "$MISSING_HINTS" = "ALL_GOOD" ]; then
    pass "All public functions have type hints"
else
    fail "Missing type hints:"
    echo "$MISSING_HINTS" | head -15
fi

# ----- 7. PRINT STATEMENTS -----
echo "7. No print statements"
PRINTS=$(grep -rn "^\s*print(" src/ --include="*.py" 2>/dev/null | grep -v "__pycache__" || true)
if [ -z "$PRINTS" ]; then
    pass "No print statements found"
else
    fail "Found print statements (use logging instead):"
    echo "$PRINTS" | head -10
fi

# ----- 8. MAGIC NUMBERS -----
echo "8. No magic numbers"
MAGIC=$(python3 << 'PYEOF'
import ast
from pathlib import Path

ALLOWED_NUMBERS = {0, 1, 2, 3, -1, 0.0, 1.0, 0.5, 255, 255.0, 100}
found = []
for py_file in sorted(Path("src").rglob("*.py")):
    if py_file.name in ("__init__.py", "config.py"):
        continue
    try:
        text = py_file.read_text()
        tree = ast.parse(text)
    except SyntaxError:
        continue
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in ALLOWED_NUMBERS and abs(node.value) > 3:
                line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                if not line_text.startswith("#"):
                    found.append(f"  {py_file}:{node.lineno}: value {node.value} — should this be in config.py?")
if found:
    print("\n".join(found[:10]))
else:
    print("ALL_GOOD")
PYEOF
)
if [ "$MAGIC" = "ALL_GOOD" ]; then
    pass "No suspicious magic numbers"
else
    warn "Possible magic numbers (check if these belong in config.py):"
    echo "$MAGIC"
fi

# ----- SUMMARY -----
echo ""
echo "========================================"
echo -e "  ${GREEN}$PASS passed${NC}  ${RED}$FAIL failed${NC}  ${YELLOW}$WARN warnings${NC}"
echo "========================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}Fix failures before committing.${NC}"
    echo ""
    exit 1
else
    echo -e "  ${GREEN}All checks passed. Safe to commit.${NC}"
    echo ""
    exit 0
fi
