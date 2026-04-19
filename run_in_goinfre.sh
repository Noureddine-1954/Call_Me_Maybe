#!/bin/bash

GOINFRE_CACHE="/goinfre/$USER/.cache"
export UV_CACHE_DIR="$GOINFRE_CACHE/uv"
export HF_HOME="$GOINFRE_CACHE/huggingface"

echo "Routing to /goinfre..."

mkdir -p "$UV_CACHE_DIR"
mkdir -p "$HF_HOME"

if [ -d "$HOME/.cache/uv" ] || [ -d "$HOME/.cache/huggingface" ]; then
    rm -rf "$HOME/.cache/uv" "$HOME/.cache/huggingface"
fi

uv run python3 -m src