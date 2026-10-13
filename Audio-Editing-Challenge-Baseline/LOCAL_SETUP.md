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

## 下载与验证

使用含 modelscope 的 Python 执行：

```bash
python download_sam_resources.py
python verify_baseline_resources.py
```

校验结果写入 `ckpts/baseline-verification.json`。下载器保留模型 LICENSE。

从 GitHub 仓库恢复全部未上传权重和 MMAE 音频时，执行 `python download_baseline_resources.py`；只需权重则附加 `--skip-audio`。逐文件遗漏清单位于上一级 `未提交.md`。

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
