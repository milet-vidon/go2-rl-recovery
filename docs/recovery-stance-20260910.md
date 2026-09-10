# 恢复后前腿交叉：诊断、重训与验收（2026-09-10）

本轮最终模型 `recovery_aligned_model3448.pt` 在 seed `20260918` 的 upright、30°侧倾与30°前后倾受控起步中各通过20/20，片尾有效站立也各为20/20。两段正面视频已检查关键帧：恢复后前腿不交叉，过度外展改善，仍有轻微左右差异。当前结果不代表完美对称、任意跌倒恢复或参考视频完整复现。

改动源于旧 `recovery_condposture_model3498.pt` 两段视频中的真实前腿交叉。接触数量和机身高度达标不能证明站姿正常，因此撤销旧模型的正常恢复演示资格，新增站姿几何验收，并完成 Uncrossed 与 Aligned 两轮独立续训。

## 旧视频诊断

在原任务 `Isaac-Recovery-Lift-Flat-Unitree-Go2-Play-v0` 下，使用原 seed `20260918`、每类 1 次、30°侧倾/前后倾重放两段视频。两类旧接触/高度计数均为 1/1；加入几何检查后均为 0/1。

机器人机身坐标为 +x 朝前、+y 朝左。前后倾案例最终左前足的 body-y 为 −0.1774 m、右前足为 +0.0715 m，左前膝为 −0.0865 m；左前髋相对默认关节角偏移 −1.1479 rad。左右前腿已进入错误侧，不能归因于拍摄透视。原视频中的 SUCCESS 字样属于已撤销的旧验收结论。

证据：[3498 审计 JSON](../evaluations/crossed-stance-audit3498/recovery_condposture_model3498.pt_recovery_metrics.json)、[逐帧 CSV](../evaluations/crossed-stance-audit3498/recovery_condposture_model3498.pt_recovery_trace.csv)。该审计用于核实旧视频中的失败姿态；任务、试验数量及评估器修订阶段与下方新训练比较不同，不混入其成功率表，也不据此计算跨任务提升幅度。

## 当前验收：stance_geometry_v1

评估脚本为 [evaluate_go2_recovery.py](../scripts/evaluate_go2_recovery.py)，几何判定为 [recovery_math.py](../src/go2_recovery/recovery_math.py)。同一时刻必须满足以下全部条件，并连续保持 3 s：

| 项目 | 阈值或定义 |
| --- | --- |
| 身体直立程度 | 机身坐标下重力方向与 `(0, 0, -1)` 的欧氏距离 < 0.35 |
| 机身高度 | 相对本环境地面原点的根部高度 > 0.30 m 且 < 0.55 m |
| 运动幅度 | 线速度范数 < 0.50 m/s，角速度范数 < 1.00 rad/s |
| 四足支撑 | 四只脚当帧的世界坐标垂直接触力各 > 5 N；不以历史窗口内各脚先后触地替代同时支撑 |
| 机身接触 | base 接触合力范数不得 > 1 N |
| 左右足端位置 | 左足 body-y > 0.06 m，右足 body-y < −0.06 m，且各足横向绝对位置 < 0.30 m |
| 左右膝位置 | 左膝 body-y > 0.04 m，右膝 body-y < −0.04 m；使用对应 calf 刚体位置 |
| 前后足端位置 | 前足 body-x > 0.08 m，后足 body-x < −0.08 m |
| 每个关节 | 相对 Go2 默认关节角的绝对偏差均 < 0.65 rad；不以平均误差掩盖单腿严重扭转 |

这些是平地正常静态站立的筛选条件，不用于要求翻滚过程每一帧都保持站姿，也不是网格碰撞检测。训练任务另行开启机器人自身碰撞。

报告中的 `successes` 是试验期间曾连续合格 3 s 的次数；`final_valid_stands` 是试验结束时仍处于连续合格至少 3 s 的次数。两者分别报告，避免将短暂站起后再次失稳算作片尾稳定。`final_geometry_passes` 只表示最后一帧通过腿部几何检查，不能单独视为恢复成功。`legacy_contact_height_successes` 仅保留作旧判据的诊断对照。

当前包装脚本设恢复时间窗为 8 s，并追加 3 s 保持观察，总仿真为 11 s。恢复时间记为首次连续合格区间的起点，而非确认 3 s 保持完成的时刻。评估使用默认关节姿态包围体上方的受控 drop start，未将机器人预先摆成静止倒地状态；禁止跌倒自动重置。

## 第一轮：Uncrossed 同任务、同种子比较

以下四个模型均使用 `Isaac-Recovery-Uncrossed-Flat-Unitree-Go2-Play-v0`、seed `20260918`，分别对 `upright`、`side`、`fore_aft` 各评估 20 次，`AngleDeg=30`。所有结果直接来自对应 JSON 的 `successes` 和 `final_valid_stands`，每列分母均为 20。30°参数用于 side/fore_aft，不能解释为 upright 或训练全部 reset 的统一初始倾角。

| checkpoint | upright 成功 | upright 片尾有效 | side 成功 | side 片尾有效 | fore_aft 成功 | fore_aft 片尾有效 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [基线 locomotion 2250](../evaluations/uncrossed-baseline2250-30/natural_stop_diagonal_model2250.pt_recovery_metrics.json) | 19/20 | 19/20 | 0/20 | 0/20 | 4/20 | 4/20 |
| [Uncrossed 2600](../evaluations/uncrossed2600-30/model_2600.pt_recovery_metrics.json) | 20/20 | 20/20 | 5/20 | 5/20 | 18/20 | 18/20 |
| [Uncrossed 2800](../evaluations/uncrossed2800-30/model_2800.pt_recovery_metrics.json) | 19/20 | 19/20 | 14/20 | 14/20 | 11/20 | 11/20 |
| [Uncrossed 2849](../evaluations/uncrossed2849-30/model_2849.pt_recovery_metrics.json) | 18/20 | 18/20 | 14/20 | 14/20 | 14/20 | 14/20 |

基线2250在本轮新增 `self_collisions_enabled` 报告字段之前评估，因此 JSON 未包含该字段；其任务为同一开启自身碰撞的 Uncrossed Play 任务。2600、2800、2849报告均明确记录该字段为 true。基线与新训练均已使用当帧垂直接触力和几何条件。

2849的侧倾和前后倾均通过14/20，优于基线的0/20和4/20；但 upright 从基线19/20降至18/20，且侧倾仍有6/20未达到完整验收。2600的前后倾结果优于2849，说明增加训练轮数并未使所有姿态同步改善。2849的侧倾终态几何通过20/20而完整成功仅14/20，也直接说明“腿没有交叉”不能等同于“已稳定站立”。上述是单一随机种子的受控起步筛查，不代表任意跌倒恢复率。

## 第一轮训练改动与来源

新任务 `Isaac-Recovery-Uncrossed-Flat-Unitree-Go2-v0` 从已有正常走停模型 `natural_stop_diagonal_model2250.pt` 初始化，未沿用3498的交叉腿局部最优。接近直立时连续惩罚跨越机身中线、前后足错位及关节偏移；正常站立终端奖励要求腿部几何、当帧四足支撑、无机身接触、低速度与正常高度同时满足。移除新任务中仍可能奖励交叉腿的旧 static/stable/conditional 奖励项，历史任务保持原配置。推理没有脚本强制摆正足端。

运行目录为 `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-10_16-50-41_uncrossed_from_locomotion2250`。本轮共600轮PPO、512个并行环境、每轮每环境24个控制步，新增环境步为 `600 × 512 × 24 = 7,372,800`。配置快照为 [configs/recovery-uncrossed-20260910](../configs/recovery-uncrossed-20260910/)，包含 `env.yaml` 与 `agent.yaml`。

训练使用 `FallAngleDeg=30`、`CurriculumSteps=4800`；4800个公共控制步约为200轮PPO，而非4800个聚合环境步。课程仍包含倒置与随机SO(3)起步，不能把30°解释为训练中所有初始姿态的最大角度。

父模型SHA-256：`33be609b3daab4a2f773d70aeeb41bc42c5cbc843780efc963ed04d5dd368417`。末尾 `model_2849.pt` 的SHA-256：`849abe1b0a6810d42433a86c284bc015c884a5f76497f2cfab87f691860703fe`。

## 在本机复测

所有新输出写入E盘。当前复测使用已归档的最终3448模型，必须显式指定 Aligned Play 任务，不能沿用包装脚本默认的旧 Stable 任务。需要复核前文历史模型时，应使用其对应的 Uncrossed Play 任务与权重。

```powershell
& E:\IsaacLab\go2-rl-open-source\scripts\install.ps1

& E:\IsaacLab\go2-rl-open-source\scripts\evaluate_recovery.ps1 `
  -Task Isaac-Recovery-Aligned-Flat-Unitree-Go2-Play-v0 `
  -Checkpoint E:\IsaacLab\go2-rl-open-source\models\recovery\recovery_aligned_model3448.pt `
  -OutputDir E:\IsaacLab\evaluations\my-aligned3448-30 `
  -Trials 20 -AngleDeg 30 -Seed 20260918 -Poses upright,side,fore_aft
```

另录正面单环境诊断视频，保留恢复动作与末尾站姿：

```powershell
& E:\IsaacLab\go2-rl-open-source\scripts\evaluate_recovery.ps1 `
  -Task Isaac-Recovery-Aligned-Flat-Unitree-Go2-Play-v0 `
  -Checkpoint E:\IsaacLab\go2-rl-open-source\models\recovery\recovery_aligned_model3448.pt `
  -OutputDir E:\IsaacLab\evaluations\my-aligned3448-front `
  -Trials 1 -AngleDeg 30 -Seed 20260918 -Poses side,fore_aft -Video -View front
```

单环境视频不替代20次统计，也不保证与20环境报告的第0号试验轨迹完全一致。正面视频的验证状态和历史失败样本见下文。

## 第二轮：左右对称与落脚宽度续训

第一轮2849两段正面视频已经提取0、0.5、1、2、5、10秒关键帧进行视觉检查：[侧倾](../evaluations/uncrossed2849-video-front/model_2849.pt_side.mp4)通过3秒保持，但落脚仍不对称；[前后倾](../evaluations/uncrossed2849-video-front/model_2849.pt_fore_aft.mp4)左前腿过度外展，判为失败。两段不再出现3498的前腿交叉，但不能作为“正常站姿已修复”的成套演示。[视频报告](../evaluations/uncrossed2849-video-front/model_2849.pt_recovery_metrics.json)。首段第一帧的RGB初始化黑帧也已记录，后续录制在计时前初始化渲染输出。

新增独立 `Isaac-Recovery-Aligned-Flat-Unitree-Go2-v0`，从2849继续学习。终端高度目标从0.32调整至0.33 m，宽松高度引导从0.31调整至0.32 m，避免停在0.30 m验收边缘。接近直立时，增加足端左右镜像误差与约0.32 m总站宽的连续惩罚；不会在推理时直接修改关节或足端位置。旧Uncrossed任务仍使用原来的奖励参数。

该训练保留原600轮Uncrossed模型作为中间对照，使用512环境、额外600轮、30°侧倾/前后倾与原混合姿态分布，课程从首轮生效。配置位于[configs/recovery-aligned-20260910](../configs/recovery-aligned-20260910/)。奖励单元测试使用2849前后倾视频实际落脚横坐标，确认过宽姿态受到惩罚、左右镜像分数一致，且左前脚修正方向朝内。

运行目录为 `E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-10_17-15-48_aligned_from_uncrossed2849`，最终保存 `model_3448.pt`。第二轮新增7,372,800环境步；两轮合计1200轮、14,745,600新增环境步。最终权重已归档为 [models/recovery/recovery_aligned_model3448.pt](../models/recovery/recovery_aligned_model3448.pt)，SHA-256为 `1f546523baa57c7997ad2883689667d6e738eac098f14fb5fa595aae778a2de2`。

以下两项均使用新任务 `Isaac-Recovery-Aligned-Flat-Unitree-Go2-Play-v0`、seed `20260918`、每姿态20次、`AngleDeg=30`，验收条件仍为前述 `stance_geometry_v1`，自身碰撞开启。单列 Aligned 结果以保留任务差异，不将其与旧3498的视频审计混成同任务统计。

| checkpoint | upright 成功 | upright 片尾有效 | side 成功 | side 片尾有效 | fore_aft 成功 | fore_aft 片尾有效 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Aligned 3200](../evaluations/aligned3200-30/model_3200.pt_recovery_metrics.json) | 20/20 | 20/20 | 17/20 | 17/20 | 19/20 | 19/20 |
| [Aligned 3448](../evaluations/aligned3448-30/model_3448.pt_recovery_metrics.json) | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

3448的侧倾恢复时间中位数为0.64 s、P90为0.78 s；前后倾中位数为0.54 s、P90为0.742 s。时间均为连续合格区间起点，随后还需保持3 s才算成功。20/20只是这一随机种子与受控起步协议下的观测结果，不能推断真实成功率必为100%。

第二随机种子 `20260919` 采用相同30°与完整几何协议，各20次：[补测报告](../evaluations/aligned3448-30-seed20260919/recovery_aligned_model3448.pt_recovery_metrics.json)。upright为20/20，side为19/20，fore_aft为20/20，片尾有效数相同。侧倾第12号试验翻至背部朝下，最终高度0.057 m、重力误差2.0、四足垂直支撑数0，未能恢复。两种子合计直立40/40、侧倾39/40、前后倾40/40；因此不宣称100%可靠恢复。

## 最终模型双视角视频核验

使用归档3448权重、新Aligned任务、seed `20260918`、30°、每类1次重新录制。[正面视频报告](../evaluations/aligned3448-video-front/recovery_aligned_model3448.pt_recovery_metrics.json)中，侧倾和前后倾的成功数与片尾有效站立数均为1/1。

| 场景 | 已录制视频 | 核验结论 |
| --- | --- | --- |
| 30°侧倾 | [正面视频](../evaluations/aligned3448-video-front/recovery_aligned_model3448.pt_side.mp4) | 已查看关键帧；恢复后前腿不交叉，站姿通过3秒连续验收 |
| 30°前后倾 | [正面视频](../evaluations/aligned3448-video-front/recovery_aligned_model3448.pt_fore_aft.mp4) | 已查看关键帧；前腿不交叉，2849的过度外展改善，站姿通过3秒连续验收 |

斜侧面也按同一seed与两类起始姿态独立重放，[斜侧视频报告](../evaluations/aligned3448-video-oblique/recovery_aligned_model3448.pt_recovery_metrics.json)两类均为1/1且片尾有效。正面、斜侧面全部四段均查看0、0.5、1、2、5、10秒关键帧，确认恢复后前腿分居正确两侧、机身由四足支撑。两段仍存在轻微左右差异，不标为完美对称。

最终[侧倾双视角视频](../videos/recovery-aligned3448/recovery_aligned_model3448_side_front_oblique.mp4)和[前后倾双视角视频](../videos/recovery-aligned3448/recovery_aligned_model3448_fore_aft_front_oblique.mp4)左侧为正面、右侧为斜侧面。两栏是相同seed的独立重放，均保留全部11秒过程，未裁去恢复过渡；使用H.264编码以便浏览器播放。[来源与帧数](../videos/recovery-aligned3448/paired_views.json)。

后续仍需扩大随机种子、倾角与真实倒地起步评估，并独立确认走停姿态和扰动恢复；本轮恢复训练不替换现有走停推荐模型。
