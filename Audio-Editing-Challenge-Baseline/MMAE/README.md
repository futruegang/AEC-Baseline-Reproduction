# MMAE: A Massive Multitask Audio Editing Benchmark
[**📖 arXiv**](https://arxiv.org/abs/2606.07229) | [**🎬 MMAE Demo Video**](https://youtu.be/6At5nTWhlXI) | [**🛠️ GitHub Code**](https://github.com/ddlBoJack/MMAE) | [**🔊 HuggingFace Audio Download**](https://huggingface.co/datasets/BoJack/MMAE)

<p align="center"><img src="assets/logo.png" alt="MMAE Benchmark Logo" width="300"/></p>

## Overview of MMAE
We introduce **MMAE**, a **M**assive **M**ultitask **A**udio **E**diting benchmark, serving as the first comprehensive evaluation testbed designed for general-purpose instruction-based audio editing. 
MMAE extends to a broad spectrum of real-world scenarios, encompassing 7 distinct audio modalities, including sound, speech, music, and their mixtures. 
MMAE establish a comprehensive taxonomy spanning 6 levels of task complexity, from basic modifications to multi-hop reasoning and multi-round editing, 2 levels of granularity, and 8 distinct operation types. 
Meticulously curated through human-agent collaboration, MMAE comprises 2,000 high-fidelity samples paired with a pioneering rubric-based evaluation framework. By decomposing free-form tasks into 17,741 verifiable criteria, this robust rubric-based paradigm enables a precise, multi-dimensional assessment of both instruction following and context consistency. 
We hope MMAE will serve as a catalyst for future advances in the intelligent creation community, providing a clear diagnostic roadmap and establishing a standardized, long-lasting evaluation paradigm for next-generation audio editing systems. 

Examples of the MMAE benchmark:

![Example](assets/examples.png)

Distribution of the MMAE benchmark across three taxonomy dimensions: modality, difficulty, and operation. 
<p float="left">
  <img src="assets/pie_modality.png" width="32%" />
  <img src="assets/pie_difficulty.png" width="32%" />
  <img src="assets/pie_operation.png" width="32%" />
</p>

## Data Curation Pipeline
MMAE is constructed through a systematic five-stage pipeline designed to ensure both diversity and high-quality of the benchmark:
1. Brainstorming.
2. Taxonomy & Paradigm Construction.
3. Instruction-Centric Data Collection.
4. Rubrics Annotation.
5. Quality Inspection.
![Pipeline](assets/pipeline.png)

## Evaluation

We use [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) as the judge model to evaluate audio editing outputs against our rubric-based criteria.

### Step 1: Connect to the judge API (recommended for a single-GPU baseline)

The official deployment script uses **Qwen/Qwen3-Omni-30B-A3B-Instruct**,
served under the alias `Qwen3Omni-Instruct`. A remote API can host the same judge,
so the evaluation process does not need local judge weights or GPUs.
The provider must confirm the actual checkpoint and support **multiple audio
inputs in one chat-completions request**. A commercial Qwen-Omni variant or a
text-only API is not automatically equivalent to the official judge.

Only `requirements-api.txt` is needed for the remote evaluation client:

```bash
pip install -r requirements-api.txt
export MMAE_JUDGE_BASE_URL="https://api.siliconflow.cn/v1"
export MMAE_JUDGE_API_KEY="your_api_key"
export MMAE_JUDGE_MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"

python -m eval.score \
  --predictions ../outputs/agent/predictions.json \
  --audio_root . \
  --output_dir ../outputs/agent_scores \
  --concurrency 1 \
  --limit 1
```

Run from this `MMAE/` directory after merging the local audio assets into it.
This example uses SiliconFlow: its [model documentation](https://docs.siliconflow.cn/docs/userguide/guides/fim)
lists the exact Instruct model, and its [multimodal API documentation](https://docs.siliconflow.cn/docs/userguide/capabilities/multimodal-vision)
documents `audio_url` Base64 inputs. Use your own account/key and confirm the
model is available to that account. A real multiple-audio request has not been
tested in this checkout; check the one-sample results for API errors before
running the full benchmark. Other hosts serving the same checkpoint can be
configured with the same environment variables.
After the one-sample check succeeds, remove `--limit 1` for full evaluation.
Keep `--retries 3`, the prompts and generation settings unchanged for comparison.
Audio files are sent as Base64 content; the remote service does not need access
to your server's filesystem. The API key is read from an environment variable
and is not written into reports. Remote API evaluation can incur provider fees.

By default, requests retain the original vLLM-compatible `audio_url` data-URI
format. If the provider documents OpenAI-style `input_audio` blocks, add
`--audio-format input_audio` (WAV/MP3). Add `--stream` only if required by its API.
Neither flag changes the judge model, rubric prompts or majority-vote scoring.

`judge_config.json` records the requested model ID and scoring settings. Confirm
the provider's underlying checkpoint separately; an API alias alone is not proof.

### Alternative: Deploy Qwen3-Omni locally

Clone the official repository and set up the environment following their instructions:

```bash
git clone https://github.com/QwenLM/Qwen3-Omni.git
cd Qwen3-Omni
# Follow the official README to install dependencies
```

Then launch the vLLM serving instances. We provide a reference deployment script [`launch_qwen3_omni.sh`](eval/launch_qwen3_omni.sh).

This starts two Qwen3-Omni instances (tensor-parallel=4 each) on 8 GPUs, serving at ports 8001 and 8002. Edit `MODEL_DIR` in the script to point to your local model weights.

### Step 2: Prepare Predictions

Run your audio editing model on the MMAE benchmark inputs ([meta data](MMAE-meta.json)). Modify the original chatml-format `messages` with an appended `assistant` turn pointing to the output audio path, e.g.,

```json
[
  {
    "id": "69e897fbf1844435bec75eca",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract the music component from the audio."},
          {"type": "audio", "audio_url": "wav/69e897fbf1844435bec75eca/audio1.wav"}
        ]
      },
      {
        "role": "assistant",
        "content": [
          {"type": "audio", "audio_url": "your_output_wav_path"}
        ]
      }
    ]
  }
]
```

The `audio_url` paths can be absolute or relative to the predictions file's parent directory (or the `--audio_root` you specify). Save the modified meta data as another JSON file indicating your model's predictions.

### Step 3: Run Evaluation

```bash
python -m eval.score \
  --predictions path/to/your_predictions.json \
  --base_urls "http://localhost:8001/v1,http://localhost:8002/v1" \
  --audio_root path/to/audio_root \
  --output_dir outputs/your_model \
  --concurrency 8
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--predictions` | **(required)** Path to your predictions JSON file. |
| `--base_urls` / `--base-url` | Comma-separated endpoint base URLs including `/v1`; alternatively set `MMAE_JUDGE_BASE_URL`. |
| `--api-key-env` | API-key environment variable name. Default: `MMAE_JUDGE_API_KEY`. |
| `--audio-format` | `audio_url` (original default) or `input_audio` (WAV/MP3). |
| `--stream` | Read streaming chat-completions responses; disabled by default. |
| `--metadata` | Path to MMAE metadata. Default: `MMAE-meta.json`. |
| `--audio_root` | Base directory for resolving relative audio paths. Default: parent directory of predictions file. |
| `--output_dir` | Where to write results. Default: `outputs/scores`. |
| `--concurrency` | Number of samples scored in parallel. Default: 16. |
| `--retries` | Number of valid judge responses to collect per rubric. Should be 3. |
| `--max_attempts` | Max total attempts (including failures) per rubric. Default: 10. |
| `--timeout` | Timeout in seconds per judge request. Default: 300. |
| `--model` | API model ID; default from `MMAE_JUDGE_MODEL`, otherwise `Qwen3Omni-Instruct`. |

**Output files** (written to `--output_dir`):

| File | Description |
|------|-------------|
| `results.jsonl` | Per-rubric detailed results: each rubric's 3 judge responses, per-attempt choices, scores, and raw model outputs. |
| `per_sample.json` | Per-sample aggregated scores: Instruction Following Rate, Consistency Rate, and Exact Match Rate for each data entry. |
| `taxonomy.json` | Scores grouped by modality, complexity, cross dimensions, and operation type. |
| `judge_config.json` | Requested judge ID and generation/scoring settings, without API credentials. |
