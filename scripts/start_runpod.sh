#!/bin/bash
set -euo pipefail

## Setup virtual environment
# python3 -m venv .venv
# source .venv/bin/activate

## Install dependencies
pip install --upgrade pip

## check cuda version and install compatible torch with nvcc --version
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

## Set environment variables
## HF_TOKEN is loaded from your .env (copy .env.example -> .env and add your token).
## It is only required for gated models (LLaMA, Mistral, Gemma).
set -a
[ -f .env ] && . ./.env
set +a
export HF_HUB_DISABLE_XET=1

## activate the virtual environment and run the training script
# source .venv/bin/activate
# bash scripts/launch.sh

## run it with: bash scripts/start_runpod.sh