# specialist-umi

本工程是 [specialist-ai/specialist](https://github.com/specialist-ai/specialist) 的开发 fork。先跑通官方采集、重建与仿真，再验证初始姿态重定向，最后迁移01非夕平台和适配π0.5。

[中文完整报告](reports/specialist-official-demo.html) · [动作坐标与字段](reports/specialist-official-demo.html#action-contract) · [AprilTag原理、公式与交互demo](reports/specialist-official-demo.html#apriltag-course) · [标定与定位的区别](reports/specialist-official-demo.html#calibration-vs-localization) · [剩余复现与计划](reports/specialist-official-demo.html#next)

## 当前已完成

- mamba环境、官方重建与硬件SDK安装验证；本机完整测试384项通过（含新增几何/AprilTag demo测试13项），官方仿真81项通过，SDK58项通过、1项真实相机测试跳过。
- 官方specialist_sim PickPlace脚本演示，落盘episode、视频与状态RRD；使用本地控制脚本，未训练策略。
- 真实X3的014单右夹爪：INSV双镜头/IMU → base → estimate/viz → 有效state → world/initial/delta EEF。
- 014有效状态为3.09–46.67秒，4359行100Hz状态；默认1步动作4358行。参考初始姿态取完整estimate的1.0秒静止段。
- Rerun默认显示三维模型、淡蓝整段轨迹、红色近期轨迹、双鱼眼Tag框/ID、六轴IMU及开合折线。
- AprilTag合成几何、错误尺寸、角点噪声实验及可选真实鱼眼PnP；EEF初始姿态重定向的数学示例。

目前没有真实左夹爪数据，没有验证USB实时取流、舵机动作、机器人TCP/初始姿态标定或真实数据驱动01平台。未训练或部署π0.5。轨迹连续和低像素误差不等于外部定位精度已验收。

## 用mamba安装与运行

从本仓库根目录执行，需要Linux、mamba、uv及C++17构建工具。默认环境为specialist-umi：

```bash
bash scripts/setup_mamba.sh
mamba activate specialist-umi
python -m pip check
```

mamba管理环境；uv只在该环境中安装Python依赖及从原锁文件导出约束。不要用`uv run`创建第二个`.venv`。辅助源码按[configs/sources.lock.json](configs/sources.lock.json)锁定提交，默认放在仓库相邻目录。若已有源码HEAD不同，脚本会保留它并退出；用`SPECIALIST_SOURCES_DIR=/another/path`安装独立副本。

仅在网络需要时显式设置`SPECIALIST_PROXY`；脚本不包含凭据。新环境的定义见[environment.yml](environment.yml)，其中包含教学绘图需要的Matplotlib。设备权限规则只提供文件，不自动提权安装；实体设备尚未验收。

## 没有真实录像也能运行的demo

```bash
bash scripts/run_official_demo.sh
python scripts/demo_apriltag.py
python scripts/demo_eef_actions.py
```

默认输出分别在`logs/official-demo`、`logs/apriltag-demo`和`logs/eef-actions-demo`。AprilTag合成图有已知位姿真值；数学重定向的robot_targets是几何目标，没有执行IK或机器人。结果图和字段含义见中文报告。

## 本机014真实数据

原片和大数据没有进入Git。当前机器已经校验复制，U盘不再需要；其他机器先自行提供合法的原始INSV配对录像。014特定目录结构和裁定依据见报告，不能把前50秒裁定直接套到其他episode。

```bash
bash scripts/process_014.sh
bash scripts/view_014.sh
```

默认启动新Rerun窗口，包含IMU、开合和Tag叠加；`--imu`为兼容别名，`--official`只打开原始官方布局。Tag框约10Hz更新，显示最近关键帧的原始检测；不等于对29.97FPS每帧重新检测。

通用导出例子：

```bash
python scripts/export_poses.py path/to/state.rrd --hand right \
  --reference-estimate path/to/estimate.rrd --reference-time 1.0 \
  --delta-steps 5 --chunk-length 16 --output-dir logs/my-actions
```

- `world_eef`：世界下的TCP矩阵；`initial_eef`：固定初始参考下的矩阵。
- `delta_eef_body_T = inverse(T_now) @ T_target`；默认`delta_eef`为本体平移m＋SO(3)旋转向量rad，另保留世界轴差分版本。
- 动作块中的每个未来目标相对同一个观测时刻；读取时间戳和dt，不能当逐步delta盲目累乘。
- 夹爪目标保留绝对opening弧度，float64，正向张开；0为URDF参考零位，不是全闭。
- 原有poses CSV保持18列；NPZ增加动作/初始字段，并输出actions CSV与定义元数据。只保存完整未来目标，不用0填充尾部。
- 不指定参考时刻时，以首个保留状态为初始帧；这不一定是录像开始。双手应传入同一个`--initial-frame` JSON以保留基线。

这些字段是本fork的几何协议，不是已完成的π0.5数据适配器。绝对/增量与EEF/关节是不同维度；openpi具体配置决定动作语义。CSV/NPZ也没有自动包含完整的训练图像与语言样本，训练适配要根据state中的图像真实时间进一步组装。

## 开发入口与检查

先读[AGENTS.md](AGENTS.md)、[AGENT-STATUS.md](AGENT-STATUS.md)、[document-version.json](document-version.json)与[CHANGELOG.md](CHANGELOG.md)。官方流水线入口仍是convert.py、estimate.py、state.py；本地新增pose_actions.py、export_poses.py、make_capture_dashboard.py及demo脚本。

```bash
mamba run -n specialist-umi pytest -q
mamba run -n specialist-umi ruff check
mamba run -n specialist-umi ruff format --check
mamba run -n specialist-umi mypy
mamba run -n specialist-umi python tools/verify_report.py
```

报告只维护`reports/specialist-official-demo.src.html`，用`bash tools/render_report.sh`生成同名HTML并查看正规预览；需要本地visual-explainer skill，可用`SPECIALIST_VE_RENDERER`指定其render.mjs路径。阅读已生成的HTML不需要该skill。

报告中的`logs/`链接属于本地复现输出，Git克隆后没有这些数据属于预期情况；当前机器可用`python tools/verify_report.py --require-local-data`严格核对。证据规则见[docs/EVIDENCE_RULES.md](docs/EVIDENCE_RULES.md)。运行数据、实拍截图、权重、硬件私有标定和自动预览图不提交；推送不等于发布网站。

## 后续顺序

先规范重复采集与尺度检查，再做真实014单右手在specialist_sim中的重放；随后验证01平台的TCP轴、IK和碰撞，最后实现模型数据适配与闭环评测。初始姿态对齐是可行的数据约定，但不能替代工具轴定义、对象相对位置与可达性验证。

---

下方保留官方上游说明。其`uv`用法对应上游工作流；本fork本机工作请使用上面的mamba入口。

6DoF tracking of gripper-mounted Insta360 dual-fisheye + IMU cameras in an
AprilTag wall-anchor frame, for human fingertip data collection.

Each gripper carries an Insta360 (two back-to-back fisheye cameras) and an IMU.
AprilTags sit rigidly on the gripper body. A wall- or ceiling-mounted
anchor tag pins the world frame and the metric scale. One GTSAM factor graph
fuses each camera's visual odometry, IMU preintegration and tag-corner
reprojection. With more than one gripper, sightings of each other's tags tie
the cameras together. The output is globally consistent metric trajectories and
finger joint angles.

A transform is a_T_b: p_a = a_T_b · p_b, the pose of frame b in frame a. a_R_b
is its rotation, and a_t_b is b's origin in frame a.

## Install

```
uv sync
```

`uv sync`, and any `uv run`, installs the default `dev` and `gpu` groups. The
`gpu` group is the CUDA solve stack. It installs on Linux only, and
`scripts/estimate.py` exits without it.

`uv sync` also compiles the native MEI factor (`_mei_factor_cpp`). It needs a
C++17 compiler and `git`. The gtsam wheel ships no headers, so the first build
clones them into a gitignored `.build/` cache. It also downloads Boost headers
there when the system has none.

The `.so` is specific to the architecture and gitignored, so run `uv sync` on
each host. To rebuild after you edit the C++ source:

```
uv run build-mei-factor
```

## Test

```
uv run pytest
```

## Run

A capture is one camera's `.insv` pair. A session is the simultaneous captures,
one per camera, converted into one base recording. The session id names the
first capture's start instant and camera serial. It is also the Rerun recording
id, and so the segment id, of each layer file of the session.

A session goes through four steps:

- The raw captures convert once into a base recording.
- Every solve reads that recording and writes its two layers beside it.
- The state layer distills both into what a policy trains on.
- A catalog serves the state layer.

Records are laid out layer-first: one directory per layer, one file per session.
So you can fetch or regenerate any layer again without touching the rest:

```
base/<session-id>.rrd          # remuxed per-lens video, IMU, calibration, clock shifts
estimate/<session-id>.rrd      # the run's answers: poses, joint angles
viz/<session-id>.rrd           # what is only drawn: ribbon, overlays
assets/gripper-<sha>-<slot>.rrd # one slot's gripper model, shared by every session
state_s8_r100/<session-id>.rrd  # what the policy sees: 360x360 images, 100 Hz state
mocap/<session-id>.rrd         # ground-truth layer (eval branch)
```

**1. Convert.** Pass one `_00_` master `.insv` per camera; the converter finds
its `_10_` rear sibling. The first capture's clock is the session clock. The
converter audio-syncs the other captures onto it. The converter trims the
recording to the span that every capture covers.

```
uv run python scripts/convert.py \
  --input-insvs path/to/g0_00.insv path/to/g1_00.insv --output-dir logs
```

This writes `logs/base/<session-id>.rrd`. The converter stops when the
audio-sync peak is ambiguous.

**2. Estimate.** `estimate.py` reads the base recording. The recording carries
the camera slots, the session window and the session id, so you pass none of
them again. The solve discovers each camera's own base and finger tags from its
rear lens against the gripper URDF, so you name only the anchors. The first
`--anchor.tag-ids` entry is the reference anchor, which pins the world gauge.
`--urdf-path` overrides the packaged gripper description.

```
uv run python scripts/estimate.py --base logs/base/<session-id>.rrd \
  --anchor.tag-ids <anchor-tag-id> --output-dir logs
```

A single camera uses the same command, with no cross-camera factors.

The run writes the estimate and viz layers under the base recording's id. A
re-run refuses to overwrite either layer unless you pass `--overwrite`, so point
experimental runs at their own `--output-dir`. Every solve also writes
`logs/blueprints/specialist.rbl`. To look at a session, load its layers and the
gripper asset. The shared id merges them:

```
uv run rerun logs/blueprints/specialist.rbl logs/base/<id>.rrd \
  logs/estimate/<id>.rrd logs/viz/<id>.rrd \
  --asset logs/assets/gripper-<sha>.rrd
```

**3. State.** A trainer builds the policy's observation from the images and
past rows, and its action from future rows.

```
state ──┬── images ───── augment ── normalize ──────────────┐
        │                                                   ├──►  obs
        ├── past rows ── transform (egocentric, …) ─────────┘
        │
        └── future rows ─ transform (egocentric, …) ──►  action   (exact, unaugmented)
```

`state.py` writes the layer from the base recording and the estimate, keyed by
hand under `state/left` and `state/right`. A lone camera rides the right hand.
Per hand, the layer holds:

- the rear lens, block-averaged to 360x360 (`--scale 8`) and blanked outside
  the fisheye disc, as one JPEG for every frame inside the usable window;
- the gripper's base-tag pose and twist in the world frame on every row,
  interpolated between the estimate's samples;
- the coupled finger angle.

The window comes from the operator's bracket gestures: a hold at the start and
a tap at the end. `--margin` seconds stay clear of each. The writer finds which
camera rode which hand.

A trainer indexes the layer by the integer `row` timeline, `round(t * --f-row)`.
Every row keeps its true time beside that label. Both hands cover one dense row
range.

```
uv run python scripts/state.py --base logs/base/<session-id>.rrd --output-dir logs
```

The layer lands in `logs/state_s8_r100/<session-id>.rrd`. The directory names
the configuration, so several configurations sit side by side in one dataset.

**4. Register** (optional) one state configuration on a Rerun catalog, for
dataset inspection and training. The one argument is the configuration's
directory. The catalog dataset takes the name of the dataset root, the parent of
that directory. It holds only the state layer and refuses a second
configuration. It installs the state dashboard as its default blueprint and also
writes it to `logs/blueprints/state.rbl`.

```
uv run python scripts/catalog.py logs/state_s8_r100 \
  --catalog-url rerun+http://host:port
```

## Layout

```
packages/
├── specialist_camera/   mei (MEI projection both ways, rescaling), image (the
│                        lens as the policy sees it)
└── specialist_dataset/  the dataset contract: schema (entity paths, timelines,
                         sessions), read (layers as numpy), frames (lens frame
                         times, H.264 decoding), poses, fiducials, state
                         (state-stream helpers)

src/specialist/
├── vi_ba/       the VI-BA solve: frontend, initialization, factor graph, the
│                native cpp/ MEI factor, LM, the joint multi-camera solve
├── record/      the rrd writers and the solve's reader: base + video +
│                offset_v3 (base recording), layer (estimate and viz
│                layers), asset (gripper model), session (a base recording
│                as the solve reads it)
└── loader/      parse_insv, capture, audio_sync

scripts/
├── convert.py          captures → one session base recording
├── estimate.py         the VI-BA driver (N ≥ 1 cameras) → estimate + viz layers
├── state.py            base + estimate → the state layer
├── catalog.py          a dataset's state layer → a Rerun catalog dataset
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
└── _probe.py           summary output the probes share
```
