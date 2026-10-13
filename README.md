# AEC baseline 复现准备

此仓库保存 ICASSP 2027 Audio Editing Challenge baseline 的源码、必要依赖源码、模型配置、MMAE 元数据/评估工具，以及本地资源下载与校验脚本。运行环境安装不在本次准备范围内。

- 项目入口：[Audio-Editing-Challenge-Baseline/README.md](Audio-Editing-Challenge-Baseline/README.md)
- 本地配置与运行说明：[LOCAL_SETUP.md](Audio-Editing-Challenge-Baseline/LOCAL_SETUP.md)
- 未上传资源的逐文件清单：[未提交.md](未提交.md)
- 打包文件清单与上游 Git 版本：[submission-manifest.json](submission-manifest.json)

## 恢复未上传的资源

在具备 modelscope、huggingface_hub 的 Python 环境中：

```bash
cd Audio-Editing-Challenge-Baseline
python download_baseline_resources.py
python verify_baseline_resources.py
```

下载器不安装环境。无需 HF SAM-Audio 访问权限，SAM-Audio-Large 和其他模型通过 ModelScope 下载，MMAE 音频通过公开的 HF 数据仓库下载。已有资源会复用。

采用默认单候选文本分离所需的 `SAM_AUDIO_MODE=lean`；没有下载 Judge、CLAP、ImageBind 或自动跨度预测器。保留 upstream baseline 的 agent 调用参数。尚未执行 GPU 推理或验证数值复现结果。

## 源码与许可证

嵌套仓库以普通源码目录保存，不使用 submodule。上游来源及对应 commit 记录在清单中。各项目的许可证文件保留在其目录内，模型 LICENSE 也保留在相应 `ckpts/` 目录。本仓库不重新授权第三方源码或模型。

本地 `prepare_submission.py` 用于生成一个独立提交目录，排除大权重、MMAE 音频及本地缓存，同时生成 `未提交.md`，不会删除本地资源。
