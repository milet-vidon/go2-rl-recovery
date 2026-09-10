# Go2 · Standing, Locomotion & Recovery

> milet-vidon 的四足机器人强化学习作品集：从失败姿态诊断，到可复核的站立、行走与停止。

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-PPO-76B900)](https://isaac-sim.github.io/IsaacLab/)
[![Robot](https://img.shields.io/badge/Robot-Unitree%20Go2-333333)](https://www.unitree.com/go2)
[![Simulation](https://img.shields.io/badge/Validation-Simulation%20only-blue)](docs/stance-and-gait-20260909.md)
[![License](https://img.shields.io/badge/License-MIT%20%2B%20BSD--3--Clause-blue)](LICENSE)

**当前成果：平地静止支撑、0.5 m/s 指令下对角小跑、停止后回到四足站立，并通过横向速度冲击测试。** 当前推荐模型是 `natural_stop_diagonal_model2250.pt`：它在三组随机种子下通过固定走停测试，并在 `0.5 m/s` 横向速度冲击下无摔倒、无机身触地。另有推扰强化模型 `model_3648.pt`，在三组种子下通过无扰、左右转向和高速走停回归；`0.75 m/s` 推扰后仍会短时卸载足端，因此只作为实验候选，不替代推荐模型。此前 Natural Stop 1349 和 Natural 950 的结果也完整保留。仍有轻微倾斜和足端轨迹差异；没有完成参考视频全部行为、AMP 复现或实机验证。

**恢复站姿改进（2026-09-10）：** 新权重 `recovery_aligned_model3448.pt` 在两个随机种子、每类合计40次受控起步中，直立保持40/40、30°侧倾恢复39/40、30°前后倾恢复40/40，通过 `stance_geometry_v1` 且片尾保持有效。正面和斜侧面均已查看关键帧：恢复后前腿不再交叉，过度外展改善，仍有轻微左右差异。一次侧倾试验翻至背部朝下后未恢复；这些仿真结果尚不能代表任意跌倒恢复，也不替换行走推荐模型。[诊断、完整验收与复测方法](docs/recovery-stance-20260910.md)。

此前 `recovery_condposture_model3498.pt` 的两段视频虽显示旧 SUCCESS，实际前腿交叉，已撤销正常恢复资格。新验收要求连续3秒同时满足四足垂直支撑、无机身触地、左右足端和膝部位置及逐关节约束。本次经过 Uncrossed 与 Aligned 两轮续训，共1200轮、14,745,600新增环境步。旧 Recovery-Lift、角度专化和高度门控结果作为历史筛查及失败对照保留，不能据其接触/高度计数声称正常恢复。[历史研究记录](docs/research-notes.md)。

![站立、行走、停止的连续仿真关键帧](docs/images/stop1349_stance.jpg)

## 演示与测量

| 场景 | 完整视频 | 原始报告 |
| --- | --- | --- |
| **Aligned 3448：30°恢复，左正面 / 右斜侧面并排** | [侧倾视频](videos/recovery-aligned3448/recovery_aligned_model3448_side_front_oblique.mp4) · [前后倾视频](videos/recovery-aligned3448/recovery_aligned_model3448_fore_aft_front_oblique.mp4) | [原始视频来源](videos/recovery-aligned3448/paired_views.json) · [seed18评估](evaluations/aligned3448-30/model_3448.pt_recovery_metrics.json) · [seed19评估](evaluations/aligned3448-30-seed20260919/recovery_aligned_model3448.pt_recovery_metrics.json) |
| **Diagonal 2250：站立 4 s → 对角行走 8 s → 停止 6 s** | [观看视频](evaluations/diagonal-2250-video/model_2250_stand_walk_stop.mp4) | [JSON](evaluations/diagonal-2250-video/model_2250_stand_walk_stop.json) · [逐帧 CSV](evaluations/diagonal-2250-video/model_2250_stand_walk_stop.csv) |
| **Diagonal 2250：横向 `0.5 m/s` 冲击** | — | [JSON](evaluations/diagonal-2250-push/model_2250_stand_walk_stop.json) |
| **Robust Push 3648：无扰走停** | [观看视频](evaluations/robust-push-3648-video/model_3648_stand_walk_stop.mp4) | [JSON](evaluations/robust-push-3648-video/model_3648_stand_walk_stop.json) · [关键帧](evaluations/robust-push-3648-video/model_3648_contact_sheet.jpg) |
| **Robust Push 3648：`0.75 m/s` 横向冲击** | [观看视频](evaluations/robust-push-3648-video-push/model_3648_stand_walk_stop.mp4) | [JSON](evaluations/robust-push-3648-video-push/model_3648_stand_walk_stop.json) · [关键帧](evaluations/robust-push-3648-video-push/model_3648_contact_sheet.jpg) |
| 历史 Recovery Lift 2300：15°旧接触/高度筛查，未复核站姿几何 | [观看视频](evaluations/recovery-lift-success15-model2300/model_2300.pt_side.mp4) | [JSON](evaluations/recovery-lift-success15-model2300/model_2300.pt_recovery_metrics.json) · [关键帧](evaluations/recovery-lift-success15-model2300/model_2300_side_contact_sheet.jpg) |
| Recovery Lift 2300：30° 侧翻/前后翻混合诊断 | [侧翻视频](evaluations/recovery-lift-angle30-model2300/model_2300.pt_side.mp4) · [前后翻视频](evaluations/recovery-lift-angle30-model2300/model_2300.pt_fore_aft.mp4) | [JSON](evaluations/recovery-lift-angle30-model2300/model_2300.pt_recovery_metrics.json) · [关键帧](evaluations/recovery-lift-angle30-model2300/model_2300_side_contact_sheet.jpg) |
| 3498 失败对照：前腿交叉（旧视频 SUCCESS 标记已撤销） | [侧翻视频](evaluations/recovery-condposture-strict-angle30-video-model3498/model_3498.pt_side.mp4) · [前后翻视频](evaluations/recovery-condposture-strict-angle30-video-model3498/model_3498.pt_fore_aft.mp4) | [更正报告](evaluations/crossed-stance-audit3498/recovery_condposture_model3498.pt_recovery_metrics.json) |
| Recovery Lift mixed 2699：旧接触/高度筛查，未检查交叉站姿 | [30°视频](evaluations/recovery-liftmixed-angle30-video-model2699/model_2699.pt_side.mp4) | [15° JSON](evaluations/recovery-liftmixed-angle15-model2699/model_2699.pt_recovery_metrics.json) · [30° JSON](evaluations/recovery-liftmixed-angle30-model2699/model_2699.pt_recovery_metrics.json) · [45° JSON](evaluations/recovery-liftmixed-angle45-model2699/model_2699.pt_recovery_metrics.json) |
| Natural Stop 1349：站立 4 s → 行走 8 s → 停止 6 s | [观看视频](videos/stop1349_stand_walk_stop.mp4) | [JSON](evaluations/natural-20260909/stop1349_video.json) · [逐帧 CSV](evaluations/natural-20260909/stop1349_video.csv) |
| Natural Stop 1349：三阶段横向冲击 | [观看视频](videos/stop1349_lateral_impulses.mp4) | [JSON](evaluations/natural-20260909/stop1349_push.json) |
| 此前 Natural 950：站走停对照 | [观看视频](videos/natural950_stand_walk_stop.mp4) | [JSON](evaluations/natural-20260909/natural950_seed09.json) |

视频保留起步、停止和受扰过程，关闭跌倒自动重置。冲击是在 2、8、15 秒施加 +0.5、−0.5、+0.5 m/s 的横向速度增量，并非已标定的物理推力。

| Natural Stop 1349 三次无扰测试范围 | 测量结果 |
| --- | --- |
| 静止 / 行走 / 停止平均机身高度 | 0.307–0.316 / 0.352–0.366 / 0.309–0.321 m |
| 0.5 m/s 指令的实测平均前进速度 | 0.549–0.572 m/s |
| 停止速度 P95 | 0.006–0.021 m/s |
| 停止后进入连续 1 s 稳定四足支撑 | 0.52–0.70 s |
| 稳态站立及停止的四足接触比例 | 100% |
| 机身接地 / 自动重置 | 0 / 0 |

| Diagonal 2250 三次无扰测试 | 测量结果 |
| --- | --- |
| 行走最大倾角 | 4.77–5.05° |
| 实测平均前进速度 | 0.54–0.56 m/s |
| 行走对角支撑比例 | 0.886–0.943 |
| 停止速度 P95 | 0.000–0.020 m/s |
| 停止阶段四足接触比例 | 100% |
| 横向 `0.5 m/s` 冲击：reset / base contact | 0 / 0 |

统计排除各阶段前 1 秒，视频保留全程。种子为 20260909、20260910、20260911，变化来自初始质量/摩擦随机化；起始位姿固定为正常朝上。三次测试不代表任意环境下的成功率。[协议、失败分析与论文笔记](docs/stance-and-gait-20260909.md)。

![机身高度、速度和足端高度](docs/images/stop1349_stand_walk_stop_metrics.png)

## 方法与个人工作

基于 NVIDIA Isaac Lab、Isaac Sim 5.1 与 RSL-RL PPO。Go2 使用 48 维观测、12 维关节位置动作，控制频率 50 Hz，物理频率 200 Hz，PD 参数 25 / 0.5。

- 审计 USD 资产、接地平面和关节姿态，纠正先前错误的高度解释。
- 将 Isaac Lab Spot 的摆动时长、对角同步、滑移及关节姿态奖励适配至 Go2，加入零指令训练。
- 建立连续站走停测试，记录足端接触和真实运动，识别“站起来但拖脚”和“训练更久却不停步”的失败模式。
- 保存训练参数、模型哈希、完整视频、原始 CSV 和失败报告，按验证表现选择 checkpoint。

这是**显式步态奖励 PPO**。Escontrela 等人的 AMP 工作及参考视频对应的 Wu 等人论文用于方法研究；本仓库尚未实现动作判别器和犬类动作重定向，不能称为 AMP 复现。[引用、AMP 源码审计与恢复课程记录](docs/research-notes.md)。

推扰强化任务 `Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0` 从 `model_2849.pt` 接续 800 轮，推扰范围扩大到 x/y ±0.9 m/s，并提高静态站立与水平姿态奖励。`model_3648.pt` 在 seed 20260909 的无扰走停报告为通过（行走最大倾角 3.84°、对角支撑 0.929、停止速度 P95 0.006 m/s）；三组种子的 normal/left/right/fast 回归均通过。按 `0.75 m/s` 冲击协议，模型无重置且无机身触地，但停止阶段四足接触比例约 0.92，未达到仓库的 >0.95 筛选线，因此不作为推荐模型。

## 快速复测（Windows / E 盘）

已验证环境：RTX 4060 Laptop 8 GB，Isaac Sim 5.1，Isaac Lab 源码版本 `37ddf626871758333d6ed89cf64ad702aef127d0`。先安装 Isaac Lab：Python 环境位于 `E:\IsaacLab\env`，源码位于 `E:\IsaacLab\repo`。本仓库放在 `E:\IsaacLab\go2-rl-open-source`。本项目是任务覆盖层，不包含模拟器或机器人 USD 资产。

```powershell
# 安装任务和评估脚本；旧文件先备份到 E 盘
& E:\IsaacLab\go2-rl-open-source\scripts\install.ps1

# 恢复站姿复测：每类20次，四足支撑与腿部几何连续保持3秒
& E:\IsaacLab\go2-rl-open-source\scripts\evaluate_recovery.ps1 `
  -Task Isaac-Recovery-Aligned-Flat-Unitree-Go2-Play-v0 `
  -Checkpoint E:\IsaacLab\go2-rl-open-source\models\recovery\recovery_aligned_model3448.pt `
  -OutputDir E:\IsaacLab\evaluations\my-aligned3448-30 `
  -Trials 20 -AngleDeg 30 -Seed 20260918 -Poses upright,side,fore_aft

# 连续站走停测试：输出 JSON、CSV、MP4
& E:\IsaacLab\go2-rl-open-source\scripts\evaluate_stand_walk_stop.ps1 `
  -Task Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0 `
  -Checkpoint E:\IsaacLab\go2-rl-open-source\models\locomotion\natural_stop_diagonal_model2250.pt `
  -OutputDir E:\IsaacLab\evaluations\my-go2-test
```

恢复视频可按[诊断文档](docs/recovery-stance-20260910.md#在本机复测)使用 `-Trials 1 -Video -View front` 单独录制。在走停评估命令后加 `-PushSpeed 0.5` 可加入横向冲击，建议改用另一个输出目录。GUI 单独观察走停可运行 `scripts/play.ps1 -Stage natural_stop -Static` 或 `-ForwardCommand`，并传入行走模型的完整 checkpoint 路径。正式验收使用固定评估脚本，避免随机命令与自动重置影响判断。

脚本将临时文件、Isaac Sim 用户数据及 pip 缓存指向 E 盘。可选资产路径为 `E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd` 和 `Environments\Grid\default_environment.usd`；安装脚本加入缓存路径支持，缺少缓存时使用 NVIDIA 官方资源地址。

## 训练与模型选择

```powershell
& E:\IsaacLab\go2-rl-open-source\scripts\train.ps1 `
  -Stage natural -NumEnvs 512 -MaxIterations 1000 -Headless
```

上面从头训练，**不等于复现发布权重的训练历程**。Natural 950 从 calibrated standard 598 微调，后者从官方 flat Go2 checkpoint 初始化。来源见 [manifest](configs/experiment_manifest.json) 和 [训练配置](configs/natural-20260909/)。完整 Natural 训练新增 12,288,000 环境步；最终 1597 权重出现零指令继续行走，拒绝作为推荐模型，保留其[失败记录](evaluations/natural-20260909/natural1597_failed_stop.json)。

推荐的 Natural Stop 1349 从 Natural 950 继续训练 400 轮（4,915,200 环境步），修正零指令下仍能获得步态奖励的条件，并提高姿态约束。原始参数在 [configs/natural-stop-20260909](configs/natural-stop-20260909/)。推理没有额外的脚本强制站姿。继续训练可使用 `-Stage natural_stop`；从指定权重恢复需要 `-LoadRun` 和 `-Checkpoint`，其中运行目录在 `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_flat` 下。

Diagonal 2250 在上述 Natural Stop 1349 基础上继续训练 400 轮（约 4,915,200 环境步），使用正确的对角配对 `FL↔RR`、`FR↔RL` 摆动对称奖励。运行目录为 `2026-09-09_03-42-13_natural_stop_diagonal_20260909`，选择 `model_2250.pt` 是因为它在三 seed 回归中比末尾 `model_2347.pt` 具有更低的行走倾角和更高的对角支撑比例。最终模型哈希和验收路径记录在 [experiment_manifest.json](configs/experiment_manifest.json)。

## 目录

```text
src/go2_recovery/       Go2 环境、奖励和注册配置
scripts/               安装、训练、播放、评估和媒体生成
patches/               E 盘资产缓存支持
configs/               模型清单、参数和来源
models/locomotion/      行走模型及历史模型（当前推荐 `natural_stop_diagonal_model2250.pt`）
models/recovery/        当前恢复模型 `recovery_aligned_model3448.pt` 与历史失败对照
evaluations/           验收报告、逐帧数据和失败记录
videos/                完整演示与历史调试视频
docs/                  方法笔记、局限和测量图片
```

## 局限与后续方向

目前验证平地、固定正向速度、连续转向、高速走停及有限横向冲击。Aligned 3448在单种子的 upright 与30°受控侧倾/前后倾中各通过20/20，正面视频中前腿不交叉；仍有轻微左右差异，尚未证明更大倾角、预先静止倒地、倒置和任意初始姿态的可靠恢复。恢复与走停使用独立模型，尚未完成行为切换或参考视频全部行为。复杂地形、AMP与实机部署也未完成。历史恢复报告缺少逐腿几何检查，不能作为正常站姿证明；3498保留为交叉腿失败对照。
两轮后续负对照也已归档：显式硬侧翻课程 `model_3699.pt` 在 30° 侧倾/前后倾均为 0/20，在 45° 为侧倾 0/20、前后倾 1/20；收紧为四足接触终端奖励的 `model_3199.pt` 在 30° 和 45° 均为 0/20。它们没有替换推荐模型。

分阶段高度门控训练 `model_2898.pt` 和 30°定向续训的末尾 `model_2599.pt` 同样没有晋级：前者在 15°为侧倾/前后倾各 4/20、30°为 4/20、1/20；后者在 30°为 3/20、4/20，45°为 0/20、6/20。该结果进一步表明，阶段奖励和更长 PPO 接续本身不足以得到可靠翻身行为；后续应验证显式行为选择器与参考起身轨迹，而不是将这些模型用作正常站姿或自恢复演示。

历史 `robust_model799.pt` 存在塌陷，`standard_stance_model950.pt` 是早期实验，均不是本页的新 Natural 950。旧视频保留用于比较，不作为成功演示。独立恢复模型曾在 30° 初始姿态测试中得到前后倾 17/20、侧倾 1/20；大侧翻和倒置恢复仍未解决，这些成绩不属于行走模型。

## 致谢与许可证

直接使用 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的环境和 Spot 步态奖励，以及 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) PPO。参考 [AMP for Hardware](https://github.com/escontra/AMP_for_hardware)、[legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) 的研究和复现形式。源自 Isaac Lab 的代码保留 BSD-3-Clause 归属；新增脚本与文档使用 [MIT](LICENSE)。机器人资产和模拟器遵循其原有许可证，不随仓库分发。
