# 当前交接状态

日期：2026-11-18。候选版本 v2026.11.18.rc.3。本次交付包含训练前工具、AprilTag 手册与双手现场手册。当前真实验收样本仍为 014 单右手。

## 最新交接

- 下一次到场先按总报告 #dual-capture 录左右签名短片，保留两对 INSV、原始哈希、手别与序列号、工位尺量、共同开录及末敲时间。首片完成四路鱼眼、双 IMU、双开合与共享初始 EEF 核对后，再录任务和压力工况。
- 新入口 scripts/make_session_review.py 从实际 state 确定手别，复用原有实测角点叠加。014 已生成单手布局和带框 RRD。官方模拟 episode 另做双手状态和共同基准导出，自检范围见 reports/evidence/bimanual-review.json。真实双机音频对齐、手别与叠影待新实录。
- 总报告 #public-progress 核对官方与外部实验分支；外部分支的多段协议和 IMU 同步后备只作参考。打印工具索引在 #apriltag-generator。

## 既有结果

- 已按要求另存 AprilTag 图解手册，在 reports/apriltag-handbook.html 与 reports/apriltag-handbook.pdf，13 张内嵌图，A4 共 9 页。图版和 PDF 页内标题均已核对。

- 图 24–31 按表格顺序覆盖八种码族。点码族名可跳到带测量线、采样点和行和的逐格图，另有八图总览。八张官方 ID 0 图按码位回读均匹配本机码本。入口 #tag-bit-layout，证据见 reports/evidence/tag-bit-layouts.json。

- 补充8种本机预置AprilTag family真实ID0图案、码本参数与当前官方额外类型范围；大小标签均41h12。核对Specialist右/上/外与官方右/下/内均右手系，4359个EEF旋转叉积/det误差约8.88e-16。旧世界轴示意改成明确俯视与侧视，避免未标景深的歧义。仅文档/图案与证据变更，未修改求解和采集。证据见reports/evidence/tag-families.json。

- export_calibration.py读取原始014 offset_v3的40项数组，保留双镜头原始参考参数、解码尺寸MEI、body_T_lens和滚动快门时间。厂家参考11904×5952，原片每镜头2880×2880；MEI与base回读误差0，外参浮点序列化误差约3.18e-8。输出在logs/pretrain-20261117/factory-calibration.json，不把设备私有参数文件提交。
- prepare_capture.py生成episode014：4359状态、4358动作、1306原始JPEG、4356组因果图像—动作样本。前两行没有过去图像，不进入样本索引；全部状态仍保留。70个区间触发工程速度复查（含6个开合触发），没有自动删帧/平滑/补零。尺量协议的物理值仍待用户提供。
- replay_capture_sim.py适配真实单右手，复用官方Floating动力学和鱼眼渲染。4359状态完整重放未失稳，位置跟随中位0.174mm、P95 0.644mm、最大0.931mm；不是外部定位精度。1306帧43.6秒四宫格视频在logs/pretrain-20261117/specialist-sim/replay-overlay.mp4，包含实拍，保持本机。
- check_forward01.py只读当前xense-description资产与xense-description-sim生成器，所有派生模型留在本工程logs。01只做mj_forward与有界IK，右手目标按平台home×逆人手1秒初始参考×人手当前TCP映射，轴对应尚未标定。左臂和01夹爪保持平台默认，不伪造左手采集、不直接迁移UMI开合角。
- 01结果：4309/4359目标满足2mm/1°；45.47–45.88秒42个、46.28–46.35秒8个未达阈值。394区间超源关节速度限值，348个两端IK均有效。全部帧有平台home已存在的左右双指碰撞网格重叠约6.69mm；另有46.18–46.25秒8帧right_link4与工具motor新增接触，最大约3.70mm。源44个collision几何均使用，台面/平台结构无源碰撞几何，凸包接触不等于精确间隙；整段未接受为执行计划。
- 一键命令bash scripts/run_pretrain_checks.sh已实际跑通，输出logs/pretrain-20261117。source-root/generator/protocol/output均可显式配置；不修改外部仓库、不启动实体控制。

## 验证

- 完整pytest391 passed、14 warnings、13.95s；新增7项因果图像/有界IK检查。既有默认8GiB GPU光流门槛未改动。
- ruff、格式、mypy、shell、正规HTML渲染与证据链接结果见reports/evidence/pretrain-checks.json及logs/pretrain-20261117。官方sim历史81项、SDK历史58通过1真实相机跳过不等于本轮硬件验收。
- 只读探测：没有X3 USB，没有舵机串口；未测试实时流、反馈或动作。物理尺量、外部精度与重复任务仍待输入。

## 原有数据与入口

014 session为20261116_192018_IAQEB2604VWWN7；原始六文件已哈希复制，U盘不需要。013只归档。完整014 estimate有573关键帧、57155高频位姿、front RMSE约2.08px；原片结束tap在83%被拒绝，明确派生前50秒base后得到3.09–46.67秒state/right。原始检测和优化器未更改。

bash scripts/process_014.sh导出world/initial/body与world delta，参考取完整estimate的1.0秒；bash scripts/view_014.sh默认完整Rerun，包括淡蓝全路径、红尾迹、双鱼眼Tag、IMU和开合。默认Tag显示最近关键帧约10Hz，不是30fps重检测。

## 下一步

1. 取得墙上内框横纵100mm、夹爪10mm实际尺量，填写configs/capture_protocol.example.json的本地副本；独立重复采集任务并复查高速区间。
2. 确认UMI/01工具轴与开度映射；复查默认双指碰撞几何及允许接触，约束连续IK、消除新增手臂碰撞并做时间参数化。
3. 明确1秒初始参考到首有效状态约259.8mm的前置运动；补充实际台面/工件碰撞输入，验证过渡和整段路径，之后才进入另行验收的实体执行。
4. 真实左手/双机同步、长期IMU噪声与独立时序、Quest手持、SortLego任务、可选catalog仍未验收。π0.5适配和训练留待用户后续安排。

主报告只编辑reports/specialist-official-demo.src.html并用tools/render_report.sh正规渲染。源码/证据与可选本机logs链接分开，默认不导出PDF。不得以数值跟随、局部IK成功或模型预览替代标定、碰撞覆盖、连续可执行性和实物验收。
