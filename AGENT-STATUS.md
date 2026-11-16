# 当前交接状态

日期：2026-11-17。版本：v2026.11.17.rc.1。当前开发/实验已完成，用户本轮明确授权提交并推送到origin/main；最终远端状态以Git记录为准。

## 已完成

- mamba specialist-umi环境、官方CUDA重建及5个辅助仓库安装；辅助源锁定在configs/sources.lock.json。SDK通过只代表软件验收。
- 官方specialist_sim PickPlace由本地脚本完成抓取放置并导出NPZ/视频/RRD；不是训练策略。
- 原始X3六文件复制并逐文件哈希校验，用户U盘可拔出；013只归档。014：完整estimate 573关键帧、前镜头重投影RMSE约2.08px、高频位姿57155行。
- 原始014结束tap检测拒绝，保留原片/完整estimate，单独裁定前50秒base；原规则输出3.09–46.67秒的state/right，4359行100Hz状态、1306图像。
- process_014.sh已导出world_eef、initial_eef、delta_eef_body/world及可选动作块；参考取完整estimate的1.0秒静止段，默认单步动作4358行。原有poses CSV18列兼容，增加actions CSV和元数据。
- view_014.sh默认新开完整仪表盘：三维模型、淡蓝整段路径/红色近期尾迹、双鱼眼Tag框ID、IMU六轴、opening；--official才是原始布局。检测为约10Hz最近关键帧，空行清除；原始数据层不改写。
- AprilTag合成图案检测/PnP、错误尺寸、720次角点噪声实验和真实014 MEI输入demo；EEF初始重定向数学demo。源码、公式、交互实验、复现范围和计划统一在中文报告。

## 最近验证

- 完整pytest：384 passed，14 warnings，14.48s；其中新增EEF/AprilTag几何13项。默认8GiB GPU光流门槛保留，没有降低生产阈值。
- 官方sim 81项、硬件SDK58项通过1项真实相机跳过的历史结果保留；本轮没有改动这些辅助仓库。
- 014 delta目标矩阵往返误差约1e-15，body delta与全局坐标换基不变性通过；开合、Tag角点/ID和纳秒时钟回读一致。没有外部位姿真值。
- 暂存快照在独立目录报告校验通过，重编译原生扩展后新增13项测试通过；复用现有mamba依赖/头文件缓存，不等于新机器重装验收。
- 新的ruff、格式、mypy、报告渲染/链接与其他检查结果见reports/evidence/learning-delivery.json；完整日志在logs/learning-20261117。

## 尚未完成的实体与模型工作

- 真实左手及双机同步；标签尺寸独立尺量、全链路外部定位精度与重复采集验证。
- udev实际安装、X3 USB实时流、舵机上电/开闭限位/方向标定；没有执行舵机运动。
- 真实014单右手驱动官方sim；上游sim-replay按双手读state，尚未适配。
- 人手/机器人TCP轴与初始姿态标定；01非夕平台IK、碰撞及实机闭环。
- π0.5数据适配、训练、动作逆变换与策略闭环；本轮只核读官方配置并制定几何落盘协议。
- 官网外部手机录像CAD叠加工具未验证公开可用；当前没有该外部录像。

## 下一位开发者的入口

读README与AGENTS后运行demo_apriltag.py、demo_eef_actions.py；有本机014数据则运行process_014.sh、view_014.sh。按报告“后续计划”先做采集协议和真实数据官方仿真重放，再迁移01，最后接模型。

报告只编辑.src.html，使用tools/render_report.sh正规渲染；当缺少本地visual-explainer时可直接读已生成HTML。默认报告校验允许gitignored logs不存在，本机可加--require-local-data。

014实际输入位于logs/real-capture-20261116/raw/IAQEB2604VWWN7；处理位于logs/014-processing。真实原片与大产物不进入Git。历史细节保留在CHANGELOG与reports/evidence；不再把历史“未求解/显存不足”当当前状态。
