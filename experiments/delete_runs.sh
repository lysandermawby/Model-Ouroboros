#!/bin/sh

: << EOF
WARNING: Will delete all saved information in all runs.
Only use this script if you are confident that you do not need any of your experiment information, or if you have backed up everything that you did need.
EOF

# should be placed in the experiments directory
DIRECTORY="$(dirname "$0")"

for file in "$DIRECTORY"/run_*; do
    if [ -d "$file" ]; then
        rm -rf "$file"
    fi 
done
