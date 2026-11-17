#!/usr/bin/env bash
# Source this file before running setup.sh / run.sh in Linux or WSL.
BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUK_REPO="$BASELINE_ROOT/AuK"
export SENSEVOICE_MODEL="$BASELINE_ROOT/ckpts/SenseVoiceSmall"
export SAM_AUDIO_MODEL="$BASELINE_ROOT/ckpts/sam-audio-large"
export SAM_AUDIO_T5_PATH="$BASELINE_ROOT/ckpts/t5-base"
export SAM_AUDIO_MODE=lean
export SAM_AUDIO_PYTHON="${SAM_AUDIO_PYTHON:-$BASELINE_ROOT/sam-audio/.venv/bin/python}"
export SAM_AUDIO_LIB="${SAM_AUDIO_LIB:-$BASELINE_ROOT/sam-audio/.venv/ffmpeg-runtime/usr/lib/x86_64-linux-gnu}"
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
# Set DEEPSEEK_API_KEY for the router and AuK prompt enhancer.
