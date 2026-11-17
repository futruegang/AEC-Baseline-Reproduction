# 本地 baseline 资源

## 此服务器的运行环境（2026-11-14）

- Miniforge：`/home/yijiefu/miniforge3`，conda 26.9.4，Bash 已初始化，base 不自动激活。
  当前终端可执行 `source /home/yijiefu/miniforge3/etc/profile.d/conda.sh` 使用 conda。
- baseline/MMAE：`.venv`，Python 3.10.22，已安装 44 个包；依赖检查、导入及 DSP 音频处理通过。
  已安装 `MMAE/requirements-api.txt` 的评估客户端依赖，无本地 Judge。
- AuK：`AuK/.venv`，Python 3.10.22；按 AuK README 的 `uv pip install -e ".[gradio]"`
  安装完整推理、Prompt Enhancer 与 ASR 依赖，并补充 baseline HTTP 服务所需的 fastapi、uvicorn、python-multipart。
  共 152 个包，依赖一致性检查、AukInfer/PromptEnhancer/SenseVoiceSmallASR 导入及 GPU 矩阵运算通过。
  主要版本：torch/torchaudio 2.7.1+cu128、torchvision 0.22.1+cu128、transformers 4.57.6、funasr 1.4.26。
  Agent 启动时已成功加载 AuK、Qwen 与 SenseVoice；本次和弦编辑被 Prompt Enhancer 判为 unsupported，未完成 AuK 生成。
- SAM-Audio：`sam-audio/.venv`，Python 3.12.3；按 SAM README 的完整包安装流程，使用
  `uv pip install . torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0`，未使用 `--no-deps`。
  已补充 baseline HTTP 服务所需的 fastapi、uvicorn、python-multipart。
  模型/处理器导入、MMAE 音频解码、GPU bfloat16 矩阵运算及服务 `--help` 通过；
  已成功加载本地 SAM-Audio-Large，并通过 Agent 实际完成一条 MMAE 音乐分离样本。
  为解决实际 ABI 冲突，将自动安装的 TorchCodec 0.17.0 调整为匹配 torch 2.8 的 0.7.0。
  实际模型加载暴露 Hub 2.x API 冲突后，调整为 transformers 4.57.6、huggingface-hub 0.36.2，关联 datasets 为 5.0.1。
- 之前误写到 home 的 uv 缓存及临时下载文件已清理。后续 AuK 安装使用 scratch 缓存。
- uv 缓存使用 `/data/scratch/yijiefu/uv-cache`；`.bashrc` 的 `UV_CACHE_DIR` 设置保留。
  `/home/yijiefu/.cache/uv` 链接至上述 scratch 目录，使未加载 `.bashrc` 的命令也使用 scratch。
  后续安装命令仍应显式设置 `UV_CACHE_DIR=/data/scratch/yijiefu/uv-cache`。
  缓存与项目跨文件系统，AuK 安装使用 `UV_LINK_MODE=copy`，无需硬链接。
- FFmpeg：系统缺少 `libavdevice.so.60`，已将 FFmpeg 6.1.1 及缺失的运行库解压到
  `sam-audio/.venv/ffmpeg-runtime`，未修改系统包。下载缓存位于 scratch 的 `sam-ffmpeg-debs`。
  `baseline-local-env.sh` 已配置 `SAM_AUDIO_PYTHON` 和 `SAM_AUDIO_LIB`，SAM 启动脚本据此设置动态库路径。
  直接在 SAM Python 中调试时，需要设置 `LD_LIBRARY_PATH="$SAM_AUDIO_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`。
- SAM 环境的 decord 0.6.0 wheel 内部仍标记为 `cp36-cp36m`，因此 `uv pip check` 和 `pip check`
  报平台标记问题；decord 可正常导入，baseline 默认文本分离路径不使用它。未修改其第三方包元数据。
- 模型权重、源代码、MMAE 音频保留；18 项模型哈希及 2185 个引用音频读取检查通过。
- 服务器四张 RTX PRO 6000 Blackwell 显卡正常；Codex 沙箱内无法访问设备，GPU 验证须在可访问设备的终端执行。
  已连接用户指定的百炼 API，Agent 使用 deepseek-v4-flash-0731；密钥仅从隐藏标准输入进入进程，未写入文件。
- 2 条 Agent 试运行输出：`outputs/baseline-ds-v4-flash-smoke-20261114/`。
  第一条 `69e897fbf1844435bec75eca` 分离成功，无错误/回退，输出约 3.08 秒。
  第二条 `69e897fbf1844435bec75ecc` 的和弦编辑被 Prompt Enhancer 判为 unsupported，7 次工具错误后回退；
  输出约 2.271 秒且与输入完全相同，不能计为编辑成功。运行结束后本次模型服务均已关闭。

只准备默认 agent 调用（`predict_spans=False`、`reranking_candidates=1`）实际需要的模型。

| 资源 | 项目内路径 | 来源 |
| --- | --- | --- |
| AuK 代码 | `AuK/` | 已有官方 checkout |
| AuK base + VAE | `AuK/ckpts/AuK/` | Tencent-Hunyuan/AuK |
| Qwen2.5-Omni-3B | `AuK/ckpts/Qwen2.5-Omni-3B/` | Qwen/Qwen2.5-Omni-3B |
| SenseVoiceSmall | `ckpts/SenseVoiceSmall/` | iic/SenseVoiceSmall |
| SAM-Audio 官方代码 | `sam-audio/` | facebookresearch/sam-audio |
| SAM-Audio-Large | `ckpts/sam-audio-large/` | ModelScope: facebook/sam-audio-large |
| T5-base 文本编码器 | `ckpts/t5-base/` | ModelScope: AI-ModelScope/t5-base |
| SAM 基础依赖源码 | `sam-audio/external/{dacvae,perception_models}/` | 官方 Git 仓库，perception_models 使用 unpin-deps 分支 |
| MMAE 元数据和音频 | `MMAE/` | 已有数据 |

SAM-Audio 使用 `lean` 加载，保留相同的 Large 主模型、文本编码器及默认分离参数，关闭不参与默认调用的 rankers/span predictor。不下载 Judge、CLAP、ImageBind 或自动跨度预测器；多候选重排序和自动跨度预测不在此配置范围内。此配置不声称与 full 模式完成了数值一致性实测。

## 手动上传与验证

源码在 `AEC/github/`，大权重和数据集音频在 `AEC/server-assets/`。将两组中的同名项目目录合并上传到服务器，然后执行：

```bash
python verify_baseline_resources.py
```

校验脚本只使用 Python 标准库及 `ckpts/baseline-verification.json` 中已记录的 SHA-256，不联网，不下载资源。未提交权重明细与数据集汇总位于上一级 `未提交.md`。模型 LICENSE 保留在相应配置目录。

## 运行路径

原项目的 `setup.sh`、`run.sh` 是 Linux shell 脚本。Linux/WSL 中先加载路径：

```bash
source ./baseline-local-env.sh
export SAM_AUDIO_PYTHON=/path/to/sam-env/bin/python
export DEEPSEEK_API_KEY=your_key
./setup.sh agent
./run.sh agent --meta ./MMAE/MMAE-meta.json --output-dir ./outputs/agent --limit 1
```

SAM 环境须独立安装官方包（Python >=3.11）和 HTTP 服务依赖 fastapi、uvicorn、python-multipart。
本机使用 uv 独立环境；原始资源准备阶段未进行 GPU 推理验证。

Windows PowerShell 的路径配置可用 `. ./baseline-local-env.ps1`；此脚本只设置变量，不将 Linux 启动脚本转换为 Windows 版本。

服务新增 `--t5-path` / `SAM_AUDIO_T5_PATH` 支持本地 T5，不改下载的原始模型配置。原有 `--model` 参数直接加载本地 `ckpts/sam-audio-large`，无需 HF SAM 访问权限。

## 使用远程 API 评估

MMAE 评分可通过远程 API 调用官方评估模型 `Qwen/Qwen3-Omni-30B-A3B-Instruct`，不在租用的 GPU 上部署评估模型。API 地址、密钥和模型 ID 分别通过 `MMAE_JUDGE_BASE_URL`、`MMAE_JUDGE_API_KEY`、`MMAE_JUDGE_MODEL` 配置。

从 `MMAE/` 目录运行 `python -m eval.score`，先使用 `--limit 1 --concurrency 1` 验证接口。具体命令见 [MMAE API 评估说明](MMAE/README.md#step-1-connect-to-the-judge-api-recommended-for-a-single-gpu-baseline)。API 评估只需 `MMAE/requirements-api.txt` 中的 HTTP 客户端依赖，不需要本地 Qwen3-Omni 权重或 vLLM。
