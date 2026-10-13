---
license: other
license_name: sam-license
license_link: LICENSE
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Reporter
    - Other
  geo: ip_location
  By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Meta Privacy Policy: checkbox
extra_gated_description: >-
  The information you provide will be collected, stored, processed and shared in
  accordance with the [Meta Privacy
  Policy](https://www.facebook.com/privacy/policy/).
extra_gated_button_content: Submit
language:
- en
---

# SAM-Audio: Segment Anything Model for Audio

Segment Anything Model for Audio [[**Blog**](https://ai.meta.com/blog/sam-audio/)] [[**Paper**](https://ai.meta.com/research/publications/sam-audio-segment-anything-in-audio/)] [[**Demo**](https://aidemos.meta.com/segment-anything/editor/segment-audio)]

SAM-Audio is a foundation model for isolating any sound in audio using text, visual, or temporal prompts. It can separate specific sounds from complex audio mixtures based on natural language descriptions, visual cues from video, or time spans.

SAM-Audio and the Judge model crucially rely on [Perception-Encoder Audio-Visual (PE-AV)](https://huggingface.co/facebook/pe-av-large), which you can read more about [here](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/)

## Authentication

Before using SAM-Audio, you need to:
1. Request access to the checkpoints on the [SAM-Audio Hugging Face repo](https://huggingface.co/facebook/sam-audio-large)
2. Authenticate with Hugging Face: `huggingface-cli login`

## Usage

SAM-Audio supports three types of prompting: text, visual, and span. Each method allows you to specify which sounds to isolate in different ways.

### 1. Text Prompting

Use natural language descriptions to isolate sounds.

```python
import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")

# Load audio file
audio_file = "path/to/audio.wav"

# Describe the sound you want to isolate
description = "A man speaking"

# Process and separate
inputs = processor(audios=[audio_file], descriptions=[description]).to(device)
with torch.inference_mode():
    result = model.separate(inputs, predict_spans=True)

# To further improve performance (at the expense of latency), you can add candidate re-ranking
with torch.inference_mode():
   outputs = model.separate(inputs, predict_spans=True, reranking_candidates=8)

# Save results
torchaudio.save("target.wav", result.target[0].unsqueeze(0).cpu(), processor.audio_sampling_rate)
torchaudio.save("residual.wav", result.residual[0].unsqueeze(0).cpu(), processor.audio_sampling_rate)
```

**Examples of text descriptions:**
- "A person coughing"
- "Raindrops are falling heavily, splashing on the ground"
- "A dog barking"
- "Piano playing a melody"
- "Car engine revving"

### 2. Visual Prompting

Isolate sounds associated with specific visual objects in a video using masked video frames.

```python
import torch
import numpy as np
from sam_audio import SAMAudio, SAMAudioProcessor
from torchcodec.decoders import VideoDecoder

# NOTE: Requires SAM3 for creating masks
# pip install git+https://github.com/facebookresearch/sam3.git
from sam3.model_builder import build_sam3_video_predictor

# Load SAM-Audio model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")

# Load video
video_file = "path/to/video.mp4"
decoder = VideoDecoder(video_file)
frames = decoder[:]

# Create mask using SAM3 (example with text prompt)
video_predictor = build_sam3_video_predictor()
response = video_predictor.handle_request({
    "type": "start_session",
    "resource_path": video_file,
})
session_id = response["session_id"]

masks = []
for frame_index in range(len(decoder)):
    response = video_predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_index,
        "text": "The person on the left",  # Visual object to isolate
    })
    mask = response["outputs"]["out_binary_masks"]
    if mask.shape[0] == 0:
        mask = np.zeros_like(frames[0, [0]], dtype=bool)
    masks.append(mask[:1])

mask = torch.from_numpy(np.concatenate(masks)).unsqueeze(1)

# Process with visual prompting
inputs = processor(
    audios=[video_file],
    descriptions=[""],
    masked_videos=processor.mask_videos([frames], [mask]),
).to(device)

with torch.inference_mode():
    result = model.separate(inputs)
```

### 3. Span Prompting (Temporal Anchors)

Specify time ranges where the target sound occurs or doesn't occur.  This provides a specific example to the model of what to isolate

```python
import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")

# Define anchors: [type, start_time, end_time]
# "+" means the sound IS present in this time range
# "-" means the sound is NOT present in this time range
anchors = [
    ["+", 6.3, 7.0],  # Sound occurs between 6.3 and 7.0 seconds
]

# Process with span prompting
inputs = processor(
    audios=[audio_file],
    descriptions=["A horn honking"],
    anchors=[anchors],
).to(device)

with torch.inference_mode():
    result = model.separate(inputs)
```

**Example with multiple anchors:**
```python
anchors = [
    ["+", 2.0, 3.5],   # Sound present from 2.0 to 3.5 seconds
    ["+", 8.0, 9.0],   # Sound present from 8.0 to 9.0 seconds
    ["-", 0.0, 1.0],   # Sound NOT present from 0.0 to 1.0 seconds
]
```

## Output Format

The `model.separate()` method returns a result object with:
- `result.target`: The isolated sound (what you asked for)
- `result.residual`: Everything else (the remainder)


Both are `list[torch.Tensor]` where each tensor is a 1D waveform

## Citation

If you use SAM Audio in your research, please use the following BibTex entry:

```bibtex
@article{shi2025samaudio,
    title={SAM Audio: Segment Anything in Audio},
    author={Bowen Shi and Andros Tjandra and John Hoffman and Helin Wang and Yi-Chiao Wu and Luya Gao and Julius Richter and Matt Le and Apoorv Vyas and Sanyuan Chen and Christoph Feichtenhofer and Piotr Doll{\'a}r and Wei-Ning Hsu and Ann Lee},
    year={2025},
    url={https://arxiv.org/abs/2512.18099}
}
```

## License

This project is licensed under the SAM License. See the [LICENSE](LICENSE) file for details.
