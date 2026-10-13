# 本地 baseline 资源

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

SAM 环境须独立安装官方包（Python >=3.11）和 HTTP 服务依赖 fastapi、uvicorn、python-multipart。本次仅准备源码、权重和路径配置，没有安装/更改 Python、CUDA 环境，也没有进行 GPU 推理验证。

Windows PowerShell 的路径配置可用 `. ./baseline-local-env.ps1`；此脚本只设置变量，不将 Linux 启动脚本转换为 Windows 版本。

服务新增 `--t5-path` / `SAM_AUDIO_T5_PATH` 支持本地 T5，不改下载的原始模型配置。原有 `--model` 参数直接加载本地 `ckpts/sam-audio-large`，无需 HF SAM 访问权限。

## 使用远程 API 评估

MMAE 评分可通过远程 API 调用官方评估模型 `Qwen/Qwen3-Omni-30B-A3B-Instruct`，不在租用的 GPU 上部署评估模型。API 地址、密钥和模型 ID 分别通过 `MMAE_JUDGE_BASE_URL`、`MMAE_JUDGE_API_KEY`、`MMAE_JUDGE_MODEL` 配置。

从 `MMAE/` 目录运行 `python -m eval.score`，先使用 `--limit 1 --concurrency 1` 验证接口。具体命令见 [MMAE API 评估说明](MMAE/README.md#step-1-connect-to-the-judge-api-recommended-for-a-single-gpu-baseline)。API 评估只需 `MMAE/requirements-api.txt` 中的 HTTP 客户端依赖，不需要本地 Qwen3-Omni 权重或 vLLM。
