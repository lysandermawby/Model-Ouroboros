#!/bin/sh

: << EOF
Sets up the git hooks found useful in development
Only relevant if you intend to contribute to this project. Any git hooks you find useful should be added to the /dev/ directory with their proper installation added to this script.
EOF

# halt on error for safety
set -e

# saving current directory
INITIAL_DIR=$(pwd)

# return to original directory on exit
trap 'cd "$INITIAL_DIR"' EXIT INT TERM

# change to script directory to ensure we find the hook files
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# file name for pre-commit hook file
PRE_COMMIT_FILE='pre-commit'

# check that we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "\033[0;31mError: Not in git working tree\033[0m"
    exit 1
fi

# check that the pre-commit file exists
if [ ! -f "$PRE_COMMIT_FILE" ]; then
    echo "\033[0;31mError: $PRE_COMMIT_FILE not found in dev/ directory\033[0m"
    exit 1
fi

# find the root directory of the git repository
GIT_ROOT=$(git rev-parse --show-toplevel)
GIT_HOOK_PATH="$GIT_ROOT/.git/hooks/$PRE_COMMIT_FILE"

# copy the hook file and make it executable
cp "$PRE_COMMIT_FILE" "$GIT_HOOK_PATH"
chmod +x "$GIT_HOOK_PATH"

echo "\033[0;32mSuccess: Installed $PRE_COMMIT_FILE to git hooks directory\033[0m"
