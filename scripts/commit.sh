#!/bin/bash
# Usage: ./scripts/commit.sh "feat: your commit message"

if [ -z "$1" ]; then
    echo "Usage: ./scripts/commit.sh \"feat: your commit message\""
    echo "Types: feat, fix, refactor, docs, test, chore"
    exit 1
fi

echo "Running code quality checks..."
echo ""

if ./scripts/check.sh --fix; then
    echo ""
    # Stage only tracked files and src/ to avoid committing secrets,
    # large binaries, or generated outputs missed by .gitignore
    git add -u
    git add src/ scripts/ Makefile README.md CODING_STANDARDS.md \
            requirements.txt pyproject.toml .flake8 .gitignore \
            .github/ .vscode/ 2>/dev/null
    git diff --cached --stat
    echo ""
    read -p "Commit and push? [y/N] " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        git commit -m "$1"
        git push
        echo "Done. Pushed to origin/main."
    else
        git reset HEAD --quiet
        echo "Aborted."
    fi
else
    echo "Fix the errors above before committing."
    exit 1
fi
