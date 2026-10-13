#!/usr/bin/env bash
# Source this file before running setup.sh / run.sh in Linux or WSL.
BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUK_REPO="$BASELINE_ROOT/AuK"
export SENSEVOICE_MODEL="$BASELINE_ROOT/ckpts/SenseVoiceSmall"
export SAM_AUDIO_MODEL="$BASELINE_ROOT/ckpts/sam-audio-large"
export SAM_AUDIO_T5_PATH="$BASELINE_ROOT/ckpts/t5-base"
export SAM_AUDIO_MODE=lean
# Set SAM_AUDIO_PYTHON to your isolated SAM-Audio environment's Python.
# Set DEEPSEEK_API_KEY for the router and AuK prompt enhancer.
