# Specialist复现与二次开发协作约定

- 开工先读README.md、AGENT-STATUS.md、document-version.json、CHANGELOG.md，检查git status；保护已有工作区修改。
- 当前优先级是官方采集/重建、数据协议、specialist_sim验证；01迁移和π0.5适配列在后续计划，只有用户进一步要求时才推进实体执行。用户最新指令优先于本文件的阶段描述。
- 环境使用mamba specialist-umi。uv仅作环境内安装/锁文件导出工具；不要执行uv run另建.venv。辅助源提交见configs/sources.lock.json，不重置别人已有的源码checkout。
- 中文技术说明统一维护reports/specialist-official-demo.src.html，并正规渲染为同名.html。使用visual-explainer skill、修复全部检查问题并看预览；临时章节预览放/tmp。版本、状态、来源和变更日志同步更新。
- 证据保存在reports/evidence，真实录像、实拍截图、RRD、训练权重及运行日志放gitignored logs。保留原始INSV与哈希；没有实物或外部真值就不能写成真机/绝对精度验收。合成数据显著标明。
- 原始官方convert/estimate/state及优化器未为本次扩展改写；本地动作协议集中在src/specialist/pose_actions.py和scripts/export_poses.py。后续扩展要继续说明官方与本地代码归属。
- 变换命名A_T_B表示把B坐标映射到A。world、固定initial-frame、相邻delta、相对当前观测的chunk是不同量；旋转用矩阵/SO(3)，不对欧拉角或四元数分量直接相减。
- 默认body delta为inv(T_now) @ T_target；6维向量是平移m和旋转向量rad，不是完整SE(3) log twist。动作带时间戳/dt；保留完整未来目标，不伪造末尾动作。夹爪目标保持绝对opening弧度。
- 双手共享初始基准才能保留手间距；目前真实数据只有右手。单手TCP朝向沿用底座Tag轴是本地约定，尚无01工具标定。绝不能把当前几何导出称为已完成的π0.5适配或机器人控制器。
- 数学变换、数据语义或流水线有实质变更时运行相关测试和真实回读；通过后仅在新证据要求时扩测。常规检查为pytest、ruff、mypy、shell语法、报告校验与git diff --check。
- 本轮2026-11-17用户明确要求提交并推送已完成工作到origin；该授权适用于本次交付，无需再次确认。不要强推、部署网站或上传大原片；后续提交/推送按用户具体指令执行。
- 默认不导出PDF。不得因软件导入、虚拟总线、合成示例或仿真通过而声称实体设备通过。
