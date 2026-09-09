# Go2 自恢复与协调行走

新增任务全部位于 `E:\IsaacLab\repo`，运行数据、缓存和日志也都在 E 盘：

- `Isaac-Recovery-Flat-Unitree-Go2-v0`：从侧躺、前后倒、仰躺和随机姿态恢复站立；速度命令固定为零。
- `Isaac-Recovery-Locomotion-Flat-Unitree-Go2-v0`：从恢复 checkpoint 继续训练；站起后跟踪速度命令，并在直立状态下鼓励对角 trot、低足滑和低动作突变。

这两个任务保持相同的 Go2 观测和动作维度。两阶段训练是有意设计：翻身动作不受 trot 约束，步态约束只在机身已直立且存在水平速度命令时施加。

## 训练前检查

先确认 Isaac Sim 能访问 Go2 的 USD 资产，并核对任务会解析成四个足端和两个对角 pair：

```powershell
Set-Location E:\IsaacLab\repo
.\isaaclab.bat -p scripts\environments\check_go2_recovery_task.py --headless --device cuda:0 --kit_args=--/app/vulkan=false
```

该检查需要访问 Isaac Sim 的远程资产源。若日志停在 `Go2/go2.usd` 的 server availability 检查，应先恢复该网络访问或将该 USD 及其依赖预缓存到本机，再开始训练；不要把缺失资产误判为策略或奖励配置故障。

如果已经把资产放到 `E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd`，训练和回放脚本会自动设置 `ISAACLAB_GO2_USD` 并优先走本地文件；地面文件 `E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd` 会自动设置 `ISAACLAB_GROUND_USD`。未找到本地文件时才使用官方 Nucleus URL。

## 训练

先运行自恢复阶段。RTX 4060 8 GB 建议从 128 个并行环境开始；若显存不足改为 64。

```powershell
E:\IsaacLab\artifacts\codex-2026-09-06-new-chat\outputs\isaaclab_go2_recovery_train.ps1 -Stage recovery -NumEnvs 128 -MaxIterations 3000
```

训练完成后，记下该次日志目录名称，例如 `2026-09-08_21-30-00_recovery`。再从该次恢复 checkpoint 微调行走：

```powershell
E:\IsaacLab\artifacts\codex-2026-09-06-new-chat\outputs\isaaclab_go2_recovery_train.ps1 -Stage locomotion -NumEnvs 128 -MaxIterations 4000 -LoadRun 2026-09-08_21-30-00_recovery -Checkpoint model_3000.pt
```

RSL-RL 的日志目录为：

```text
E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery
```

回放恢复或步态策略时使用完整 checkpoint 路径：

```powershell
E:\IsaacLab\artifacts\codex-2026-09-06-new-chat\outputs\isaaclab_go2_recovery_play.ps1 -Stage recovery -Checkpoint E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-08_21-30-00_recovery\model_3000.pt

E:\IsaacLab\artifacts\codex-2026-09-06-new-chat\outputs\isaaclab_go2_recovery_play.ps1 -Stage locomotion -ForwardCommand -Checkpoint E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-08_22-50-00_locomotion\model_4000.pt
```

`-Headless` 可用于长训练；默认会打开视窗。所有脚本均可先追加 `-DryRun` 检查命令，不会启动仿真。

## 初始姿态与成功定义

恢复训练的 reset 分布为：30% 轻倾斜直立、25% 左右侧躺、25% 前后倒地、10% 仰躺、10% 随机 SO(3)。倒地时根节点高度会降低到 0.16-0.22 m，避免从站立高度掉落造成非代表性的冲击。关节从默认站姿的小扰动开始，并会被软关节限位裁剪。

不要只看平均 reward。每个姿态类别至少独立评估 100 次，并报告：恢复成功率、恢复时间中位数、P90 恢复时间、恢复后连续站立 3 秒的比率、速度跟踪误差和足端滑移。一个合理的第一阶段仿真目标是每类恢复成功率不低于 90%，P90 恢复时间不超过 3 秒；达不到时先分析失败姿态，再调整课程或随机化，不能用单段视频替代评估。

## 论文依据

- 本轮检索时间：2026-09-08。以下链接均已打开并核对摘要或正文可见内容；论文中的实验数字不是本项目的保证。
- Lee, Hwangbo, Hutter, *Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning*, arXiv:1901.07517。摘要报告了“行为选择器 + 三个恢复行为”的层级结构，在 ANYmal 上对任意摔倒姿态恢复时间少于 5 s、100 次以上测试成功率高于 97%。因此本项目保留“恢复策略”和“行走策略”分阶段训练，并把姿态分桶评估作为必需项。
- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science Robotics, 2019. [DOI: 10.1126/scirobotics.aau5872](https://doi.org/10.1126/scirobotics.aau5872)。采用扰动鲁棒训练和仿真到实物迁移的思路。
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots*, RSS 2021. [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)。摘要明确采用 base policy + adaptation module，在仿真中训练并快速适应地形、载荷和磨损；当前 Go2 版本先保留噪声和随机化接口，后续 sim-to-real 再加入历史观测适应模块。
- Zhang et al., *Research on Self-Recovery Control Algorithm of Quadruped Robot Fall Based on Reinforcement Learning*, Actuators, 2023. [DOI: 10.3390/act12030110](https://doi.org/10.3390/act12030110)。正文报告其 DDPG 平地恢复仿真约 2.25 s，并强调减少突然速度变化和重复摩擦；本项目对应加入早期稳定奖励、动作率惩罚和足端滑移惩罚，但不把 2.25 s 当作 Go2 保证。
- Sheng et al., *Bio-Inspired Rhythmic Locomotion for Quadruped Robots*, IEEE Robotics and Automation Letters, 2022. [DOI: 10.1109/LRA.2022.3177289](https://doi.org/10.1109/LRA.2022.3177289)。IEEE 摘要强调哺乳动物节律运动和可切换的 rhythmic motor patterns；本项目用可测量的对角同步、交替摆动、足端离地高度、低滑移和低动作突变实现工程化近似，并非动物动作捕捉复现。

当前实现并非动作捕捉模仿。要复制特定动物的步态、起身轨迹或风格，需要相应的参考运动数据，并采用 motion imitation/AMP 等目标；不能仅凭一般的 gait reward 声称完成生物动作复现。

## AMP 源码审计（2026-09-09）

本轮还核对了 [AMP for Hardware](https://github.com/escontra/AMP_for_hardware) 的源码（commit `bfb0dbdcf32bdf83a916790bddf193fffc7e79b8`）。其判别器把当前状态和下一状态拼接后输入 ReLU MLP，专家样本使用梯度范数正则（lambda=10），判别器输出再按 `clamp(1 - 0.25 * (d - 1)^2, min=0)` 转成模仿奖励。当前 Go2 项目没有专家动作数据、判别器、专家数据加载或 AMP 奖励混合，因此仍是显式步态奖励 PPO；这里只记录方法依据，没有把 AMP 代码未经验证地混入训练。

## 恢复课程审计（2026-09-09）

早期恢复重训把 reset 课程进度固定在一百万仿真步。对 500 轮、1024 环境的短接续实验，进度只约为 0.012，且从 checkpoint 恢复时不保证沿用上次的环境计数，不能视作完成了渐进课程。为便于可复核的硬姿态微调，`recovery_mdp.reset_root_state_mixed` 现在读取 `ISAACLAB_RECOVERY_CURRICULUM_STEPS`；训练脚本通过 `-CurriculumSteps` 设置它，默认值仍是一百万步，正值为 1 可使目标分布从首个 reset 生效。该开关不会修改历史权重。

从 `recovery_stage30_model2200.pt` 接续的 500 轮、45° 侧翻实验生成了 `model_2699.pt`，但同协议评估为 45° 侧倾 0/20、30° 侧倾 0/20，因此已拒绝作为候选。它说明在未显式控制课程和奖励遗忘时，单纯增加训练轮数可能退化；后续硬侧翻实验必须单独记录并与 2200 对照。

后续两轮负对照也已完成。显式硬侧翻课程 `model_3699.pt` 在 30° 侧倾/前后倾均为 0/20，在 45° 为侧倾 0/20、前后倾 1/20。随后把 Stable Recovery 的终端支撑奖励收紧为四足接触并训练 `model_3199.pt`，30° 和 45° 两种姿态均为 0/20。报告归档在 `evaluations/recovery-hard45-cur1-angle30/`、`evaluations/recovery-hard45-cur1-angle45/`、`evaluations/recovery-fourfeet-angle30/` 和 `evaluations/recovery-fourfeet-angle45/`；两个 checkpoint 均未晋级。失败轨迹显示策略会停在约 0.26--0.29 m 的低三足姿态，说明还缺少明确的翻身动作阶段或参考动作目标，不能靠增加 PPO 轮数声称完成恢复。

## 本轮实际验证结果（2026-09-08）

评估器位于 `E:\IsaacLab\repo\scripts\environments\evaluate_go2_recovery.py`，成功定义为：重力误差 < 0.35、根部高度 0.30–0.55 m、线速度 < 0.50 m/s、角速度 < 1.00 rad/s、至少两个足端接触力 > 5 N，并连续保持 3 s。每个姿态类别独立 100 次；视频中的姿态是受控 drop start，不是把机器人预先摆成“已恢复”的姿态。

严格结果：

| checkpoint | upright | side | fore-aft | upside-down | random |
|---|---:|---:|---:|---:|---:|
| `model_600.pt` | 97/100 | 0/100 | 0/100 | 0/100 | 1/100 |
| `model_1399.pt` | 93/100 | 0/100 | 0/100 | 0/100 | 1/100 |
| `model_1799.pt` | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |

因此当前代码和训练结果**尚不能声称已经完成四足机器人自恢复**。`model_1399.pt` 仅作为当前最佳可复现实验点；`model_1799.pt` 已发生退化，不用于演示。视频也明确分为成功的直立保持片段和失败的侧倒片段，不能把前者当作侧倒恢复证明。

## Recovery Lift continuation (2026-09-09)

The phased height-gated continuation (`model_2898.pt`) still converged to a
low base pose and scored only 4/20 side and 1/20 fore-aft at 30 degrees. A
gentler continuation was therefore branched from the previously validated
Stable `model_2200.pt`. `Recovery-Lift` preserves the Stable reward scale and
adds two terms that activate only for an upright, three-foot-supported body
below 0.30 m: a bounded low-height penalty and a positive upward-velocity cue.
The 100-iteration checkpoint `model_2300.pt` improved the 15-degree screen to
16/20 side and 15/20 fore-aft, but reached only 2/20 and 6/20 at 30 degrees and
0/20 and 1/20 at 45 degrees. It is archived as an experimental checkpoint,
not a replacement for the locomotion recommendation or proof of arbitrary
fall recovery. The 15-degree single-environment video is a genuine success
sample; the 30-degree videos intentionally show the diagnostic environment and
must be read together with the 20-trial JSON report.

A further 300-iteration continuation used a fixed 30-degree fall cap. Its best
intermediate `model_2500.pt` scored 4/20 side and 6/20 fore-aft at 30 degrees,
but the same checkpoint scored only 5/20 and 12/20 at 15 degrees and 1/20 and
6/20 at 45 degrees. This is a documented regression from angle specialization,
so it remains a negative control and is not copied into `models/recovery`. The
final `model_2599.pt` did not improve that conclusion: 3/20 side and 4/20
fore-aft at 30 degrees, then 0/20 and 6/20 at 45 degrees. Those reports are
archived under `evaluations/recovery-lift30-angle{15,30,45}-model*`.

The independently phased, height-gated run resumed from `model_2499.pt` for
400 iterations with 512 environments (4,915,200 additional environment
steps). It selected brace, lift, land, and stand rewards from body state and
required a normal-height four-foot stand reward behind a 0.30 m height gate.
The earlier `model_2700.pt` and `model_2800.pt` both scored 0/20 for side and
fore-aft at 15 degrees. The final `model_2898.pt` improved only to 4/20 for
each category at 15 degrees and 4/20 side, 1/20 fore-aft at 30 degrees. This
is evidence that the height gate did not solve the low-base failure mode, not
evidence of usable self-righting; the checkpoint is intentionally not copied
to `models/recovery`.

## Mixed-pose Recovery-Lift continuation (2026-09-10)

To reduce angle specialization, a second continuation resumed the Stable
`model_2200.pt` with the normal mixed-pose reset distribution for 500 PPO
iterations (`2026-09-09_23-37-25_recovery_lift_mixed_from2200_20260910`). The
best saved checkpoint, `model_2699.pt`, scored 15/20 side and 13/20 fore-aft
at 15 degrees, 9/20 side and 8/20 fore-aft at 30 degrees, and 5/20 for both
categories at 45 degrees. These are the strongest recovery-lift screens so
far, but they remain below a usable self-righting target. Reports are archived under
`evaluations/recovery-liftmixed-angle15-model2699/` and
`evaluations/recovery-liftmixed-angle30-model{2300,2400,2500,2600,2699}/`.
The checkpoint therefore remains an experimental artifact and is not copied
into `models/recovery` or used in the locomotion demonstrations.

## 条件站姿续训与严格验收（2026-09-10）

论文与开源实现（Lee et al. 2019、DreamRiser、FR-Net、AFR）共同指出：翻身、抬升和站立目标存在冲突，且过大的足端接触奖励会产生低位 frog-squat。基于此，在 `Recovery-Lift` 中新增 `conditional_stand_posture`：只有重力误差小于 0.25 且基座高于 0.28 m 时，才奖励回到 Go2 默认对称关节姿态；翻滚阶段不施加该姿态约束。

从混合姿态 `model_2699.pt` 接续 800 轮得到 `model_3498.pt`。正式评估器已改为至少四足接触（不再把两足支撑算作成功），并要求正常高度、直立、低速、连续保持 3 s。20 次结果为：15° 侧翻/前后翻 15/20、15/20；30° 为 11/20、11/20；45° 为 11/20、7/20。单环境严格视频保存在 `evaluations/recovery-condposture-strict-angle30-video-model3498/`。这仍是仿真研究候选，未达到任意跌倒恢复或参考视频完整复现标准，也未复制为推荐模型。

该结果支持继续采用分层结构：self-right → settle/stand → locomotion，并在后续实现相对当前关节的 recovery action、站立阶段 nominal-pose action 和带迟滞的行为选择器，而不是无限增加单一 PPO 的训练轮数。

## 实机前的必要步骤

本任务的训练结果仅代表仿真。进入 Go2 实机前，应逐步加入并验证质量/质心、关节阻尼、执行器延迟与强度、地面摩擦、接触、观测噪声和命令延迟的 domain randomization；先使用安全吊挂、低幅度动作和硬件扭矩/速度限制。翻身时尤其要设置关节温度和机身碰撞安全边界。
