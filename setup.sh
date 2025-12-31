#!/bin/sh

: << EOF
Setup the dependencies for this project
EOF

# halt on error for safety
set -e 

RED="\033[0;31m"
GREEN="\033[0;32m"
NC="\033[0m"

if command -v uv > /dev/null 2>&1 ; then
    echo "${GREEN}Success: uv is installed ${NC}"
    echo "Now installing dependencies \n"
else
    echo "${RED}Error: uv is not installed ${NC}"
    echo "Dependency management in this project should be done with uv"
    echo "If you only have pip and insist on using it, then you can run the command 'pip install -r requirements.txt'"
    echo "Note that this was generated with the command 'uv pip compile pyproject.toml -o requirements.txt' and may not work correctly."
    echo ""
    echo "It is recommended that you install uv. Please find the appropriate installation from the following link:"
    echo "      https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# installing dependencies (excluding PyTorch)
uv sync

# Install PyTorch with appropriate backend
echo "\n${GREEN}Installing PyTorch...${NC}"
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "CUDA detected - installing PyTorch with CUDA 12.1 support"
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
elif [[ $(uname -m) == 'arm64' ]] && [[ $(uname -s) == 'Darwin' ]]; then
    echo "Apple Silicon detected - installing PyTorch (with MPS support)"
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    echo "No GPU acceleration detected - installing CPU-only PyTorch"
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "\n${GREEN}Setup complete!${NC}"
echo "To verify GPU support, run one of the following:"
echo "  CUDA: python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo "  MPS:  python -c 'import torch; print(f\"MPS available: {torch.backends.mps.is_available()}\")'"
