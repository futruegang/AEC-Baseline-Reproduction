$taskBaselineRoot = $PSScriptRoot
$env:AUK_REPO = Join-Path $taskBaselineRoot 'AuK'
$env:SENSEVOICE_MODEL = Join-Path $taskBaselineRoot 'ckpts/SenseVoiceSmall'
$env:SAM_AUDIO_MODEL = Join-Path $taskBaselineRoot 'ckpts/sam-audio-large'
$env:SAM_AUDIO_T5_PATH = Join-Path $taskBaselineRoot 'ckpts/t5-base'
$env:SAM_AUDIO_MODE = 'lean'
# Set SAM_AUDIO_PYTHON to your isolated environment's Python before launching.
