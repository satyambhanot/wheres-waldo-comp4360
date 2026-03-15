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
    git add .
    git commit -m "$1"
    git push
    echo "Done. Pushed to origin/main."
else
    echo "Fix the errors above before committing."
    exit 1
fi
