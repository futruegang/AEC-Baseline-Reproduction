# AEC baseline 复现准备

此仓库保存 ICASSP 2027 Audio Editing Challenge baseline 的源码、必要依赖源码、模型配置、MMAE 元数据/评估工具和运行/校验脚本。运行环境安装不在本次准备范围内。

- 项目入口：[Audio-Editing-Challenge-Baseline/README.md](Audio-Editing-Challenge-Baseline/README.md)
- 本地配置与运行说明：[LOCAL_SETUP.md](Audio-Editing-Challenge-Baseline/LOCAL_SETUP.md)
- 未上传权重明细与数据集汇总：[未提交.md](未提交.md)
- 打包文件清单与上游 Git 版本：[submission-manifest.json](submission-manifest.json)

## 手动上传服务器

本地 `AEC/` 分成两组，组内保留相同的项目相对路径：

- `可提交GitHub/`：本仓库的源码和小文件。
- `需手动上传服务器/`：已下载的大权重和 MMAE 音频。

在服务器上 clone 本仓库，然后把 `需手动上传服务器/` 中的 `Audio-Editing-Challenge-Baseline/` 合并到仓库内同名目录，保留源码、配置和权重。

合并完成后可使用纯 Python 校验脚本：

```bash
cd Audio-Editing-Challenge-Baseline
python verify_baseline_resources.py
```

权重和音频已经在本地准备好，不提供恢复下载脚本。校验脚本依据保留的 SHA-256 记录检查手动复制的文件，无需网络。

采用默认单候选文本分离所需的 `SAM_AUDIO_MODE=lean`；没有下载 Judge、CLAP、ImageBind 或自动跨度预测器。保留 upstream baseline 的 agent 调用参数。尚未执行 GPU 推理或验证数值复现结果。

## 源码与许可证

嵌套仓库以普通源码目录保存，不使用 submodule。上游来源及对应 commit 记录在清单中。各项目的许可证文件保留在其目录内，模型 LICENSE 也保留在相应 `ckpts/` 目录。本仓库不重新授权第三方源码或模型。

