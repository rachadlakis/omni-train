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

## Configure git
git config --global user.name "rachadlakis"
git config --global user.email "rachadlakis@gmail.com"

## Set environment variables
export HF_TOKEN="${HF_TOKEN:-***REDACTED-HF-TOKEN***}"
export HF_HUB_DISABLE_XET=1

## activate the virtual environment and run the training script
# source .venv/bin/activate
# bash scripts/launch.sh

## run it with: bash scripts/start_runpod.sh