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

## 本轮实际验证结果（2026-09-08）

评估器位于 `E:\IsaacLab\repo\scripts\environments\evaluate_go2_recovery.py`，成功定义为：重力误差 < 0.35、根部高度 0.30–0.55 m、线速度 < 0.50 m/s、角速度 < 1.00 rad/s、至少两个足端接触力 > 5 N，并连续保持 3 s。每个姿态类别独立 100 次；视频中的姿态是受控 drop start，不是把机器人预先摆成“已恢复”的姿态。

严格结果：

| checkpoint | upright | side | fore-aft | upside-down | random |
|---|---:|---:|---:|---:|---:|
| `model_600.pt` | 97/100 | 0/100 | 0/100 | 0/100 | 1/100 |
| `model_1399.pt` | 93/100 | 0/100 | 0/100 | 0/100 | 1/100 |
| `model_1799.pt` | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |

因此当前代码和训练结果**尚不能声称已经完成四足机器人自恢复**。`model_1399.pt` 仅作为当前最佳可复现实验点；`model_1799.pt` 已发生退化，不用于演示。视频也明确分为成功的直立保持片段和失败的侧倒片段，不能把前者当作侧倒恢复证明。

## 实机前的必要步骤

本任务的训练结果仅代表仿真。进入 Go2 实机前，应逐步加入并验证质量/质心、关节阻尼、执行器延迟与强度、地面摩擦、接触、观测噪声和命令延迟的 domain randomization；先使用安全吊挂、低幅度动作和硬件扭矩/速度限制。翻身时尤其要设置关节温度和机身碰撞安全边界。
