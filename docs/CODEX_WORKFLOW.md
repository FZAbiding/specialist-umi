# 文档与验证流程

本工程参照 `/home/xense/cad/taccap-industrial/AGENTS.md`、`docs/CODEX_WORKFLOW.md` 和 `docs/EVIDENCE_RULES.md`，采用单一中文 HTML 主报告与结构化证据。

1. 读取项目状态与版本，检查工作区；不重置已有修改。
2. 安装与实验用 mamba 环境，记录实际命令、退出码、源码提交、输出位置和限制。
3. 编辑 `reports/specialist-official-demo.src.html`，关键结论在附近引用来源 ID；将来源和本机验证登记到 `reports/evidence/`。
4. 更新 `document-version.json`、`CHANGELOG.md` 和 `AGENT-STATUS.md`。候选版采用 `vYYYY.MM.DD.rc.N`，不自动正式发布。
5. `bash tools/render_report.sh`；修复全部渲染错误并查看其输出预览图。可用 SPECIALIST_VE_RENDERER 指定技能的正规渲染器，不复制或修改 kit 内部实现。
6. `python tools/verify_report.py` 与 `git diff --check`；按改动运行必要软件检查。

原始录像、运行日志、仿真 npz、视频和 RRD 存放 `logs/`，不提交大文件。模拟数据明确标识来源，不冒充相机采集。实体标定和相机实拍未执行时保持待验。

默认只交付 HTML；不自动导出 PDF、部署网站、发布带厂商 SDK 的二进制包或发送外部消息。

本机技能依赖的旧 Chromium 134 不能完成其 SVG 几何检查。当前使用已有较新 Chromium 缓存，通过 `logs/setup/ve-browser` 的隔离路径供正规渲染器调用；该目录中的链接是本机运行配置，不打包或提交。其他机器需要配置兼容的浏览器，不能把“Visual check skipped”当成预览通过。
