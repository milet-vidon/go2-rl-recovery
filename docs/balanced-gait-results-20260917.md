# Balanced-gait 匹配实验最终结果（2026-09-17）

## 决策：拒绝晋级，保留已有模型

**control4246与balanced4246均未通过旧预注册验收；不晋级、不替换默认模型。** 两组各21案已全部完成：control 11/21（common 4/6、original 7/15），balanced 13/21（common 5/6、original 8/15）。balanced的平均对角支撑占空比差仅改善 **12.3510%**，未达到预声明的≥25%。通过案例数增加或速度接近指令都不能覆盖这两个独立拒绝原因。

本结果核对截至 **2026-09-17 17:44:48 +08:00**。control screen完成于17:15:28，balanced完成于17:35:24。新增滑移训练等后续训练**尚未启动**；当时连续0.5 m/s流程还在独立smoke-v4诊断中，不能写成完整恢复→行走→停止已经验证，更不能称小跑/快跑/动物动作复现。

## 实验与身份审计范围

两arm独立从同一冻结3947完整actor/critic/std/Adam出发，seed42、128 env×300更新×24 steps/env，各921600环境步和7200控制区间，最终4246。共同使用self-collision ON、nominal soft-clamped action、固定base mass的common物理，旧command/noise/reset/push/PPO及旧reward保留；两arm唯一处理差异是command-gated completed-duration variance新增项weight 0与-10。最终点事先指定，不筛选中间checkpoint。

身份审计两arm各通过：21唯一case、105个唯一case-artifact路径、18,900条实际控制记录，共37,800条。它绑定arm/physics/seed/command/model receipt/summary/report/trace/CSV；本次又只用PowerShell读取全部42实际报告，重算报告SHA并与summary和audit逐case匹配，确认报告retention_passed一致。全部done计数为0。

**身份审计通过不等于行为通过。** checkpoint张量/完整配置的深层验证由既有冻结wrapper承担；该身份审计不是重新训练证明、视觉审核或硬件测试。42案均满足报告中的静止几何、current四足垂直支撑、current base清空及无reset/无base contact要求；但control original left/seed09的quiet_stand_stop失败，不能泛称“所有停止功能完好”。

## 全部21案配对结果（未删除失败）

common/recovery为一套恢复兼容物理，original为旧locomotion物理。每行列出完整失败项；“通过”只指原物理行为门，不表示步态style、泛化或全流程通过。

| physics | case | seed | control4246 | balanced4246 |
|---|---|---:|---|---|
| recovery | normal | 20260909 | 通过 | 通过 |
| recovery | retained08 | 20260909 | 通过 | 通过 |
| recovery | left | 20260909 | 通过 | 通过 |
| recovery | right | 20260909 | lateral_tracking | 通过 |
| recovery | push05 | 20260909 | 通过 | 通过 |
| recovery | target10 | 20260909 | limited_slip | limited_slip |
| original | normal | 20260909 | walk_tracking | walk_tracking |
| original | normal | 20260910 | 通过 | 通过 |
| original | normal | 20260911 | 通过 | 通过 |
| original | retained08 | 20260909 | walk_tracking, straight_drift | walk_tracking, straight_drift |
| original | retained08 | 20260910 | walk_tracking | walk_tracking |
| original | retained08 | 20260911 | walk_tracking | walk_tracking |
| original | left | 20260909 | walk_tracking, quiet_stand_stop, yaw_tracking | walk_tracking, yaw_tracking |
| original | left | 20260910 | yaw_tracking | 通过 |
| original | left | 20260911 | yaw_tracking | 通过 |
| original | right | 20260909 | lateral_tracking | walk_tracking |
| original | right | 20260910 | 通过 | 通过 |
| original | right | 20260911 | 通过 | 通过 |
| original | push05 | 20260909 | 通过 | walk_tracking |
| original | push05 | 20260910 | 通过 | 通过 |
| original | push05 | 20260911 | 通过 | 通过 |

common 1.0仍是滑移失败：control实际vx=0.936924、slip=0.125988；balanced实际vx=0.975033、slip=0.123287 m/s，均未满足原有slip<0.12。balanced common .8的slip=0.112129，高于control的0.082405，虽仍过界但不是所有指标都改善。

## 真实支撑时序：略有改善，仍明显不均衡

matched comparison使用每CSV完整900行中的固定350个walk样本（time_s>5，即5.02–12.00 s，50 Hz），不取静止或挑选子窗口。raw duty依据**当前垂直足力>5 N**；不是history-contact并集，也不是训练timer的current norm>1 N。gap定义为 `abs((d_FL+d_RR−d_FR−d_RL)/2)`，单元格占空比与gap均为0–1比例。

| arm | 指令m/s | FL duty | FR duty | RL duty | RR duty | diagonal gap | exact对角支撑比例 | sampled全足腾空 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | .5 | 0.814286 | 0.277143 | 0.260000 | 0.797143 | 0.537143 | 0.902857 | 0.000000 |
| control | .8 | 0.742857 | 0.314286 | 0.331429 | 0.742857 | 0.420000 | 0.914286 | 0.000000 |
| control | 1.0 | 0.725714 | 0.362857 | 0.365714 | 0.725714 | 0.361429 | 0.908571 | 0.000000 |
| balanced | .5 | 0.780000 | 0.302857 | 0.280000 | 0.760000 | 0.478571 | 0.908571 | 0.000000 |
| balanced | .8 | 0.705714 | 0.368571 | 0.348571 | 0.711429 | 0.350000 | 0.917143 | 0.000000 |
| balanced | 1.0 | 0.685714 | 0.365714 | 0.360000 | 0.694286 | 0.327143 | 0.917143 | 0.000000 |

三案均值：0.4395238095→0.3852380952，下降12.3510%；≥25%要求balanced≤0.3296428571，实际未达到。分母非零，此处不存在0/0改善误判。三个个案均改善，且没有个案gap恶化>.05；这只支持“本次有限定向改善”，不支持充分消除偏置或宣称方法普遍有效。

cycle统计另用连续两个50 Hz样本确认状态变化（40 ms persistence），剔除窗口首尾不完整周期。各脚完整周期数control为.5全部11、.8全部14、1.0全部15；balanced为.5 [12,11,11,12]、.8 [13,14,14,13]、1.0全部15（均按FL/FR/RL/RR）。24个case-foot都没有<150 ms完整周期，故“每脚≥2完整周期”和“未新增短周期”门通过。但150 ms只是诊断标记，不是步态定义；去抖后无短周期不能证明原始单帧抖动不存在。

支撑偏置不只是边界采样现象。下表由matched comparison列出的全部完整cycles分别取stance/swing均值（秒），示例为balanced四脚：

| 指令m/s | FL stance/swing | FR stance/swing | RL stance/swing | RR stance/swing |
|---|---:|---:|---:|---:|
| .5 | 0.4267 / 0.1233 | 0.1709 / 0.3800 | 0.1564 / 0.3945 | 0.4167 / 0.1350 |
| .8 | 0.3323 / 0.1369 | 0.1714 / 0.2971 | 0.1629 / 0.3071 | 0.3354 / 0.1338 |
| 1.0 | 0.3040 / 0.1373 | 0.1600 / 0.2813 | 0.1587 / 0.2840 | 0.3080 / 0.1333 |

例如balanced .5的FL/RR平均stance约0.427/0.417 s，FR/RL只有0.171/0.156 s；1.0时前一对仍约0.304/0.308 s，后一对约0.160/0.159 s。大量exact对角支撑并不等于两对脚自然均衡交替。六个窗口的50 Hz全足腾空sample均为0；这既不能凭速度称快跑，也不能单凭离散采样0去否定所有可能的亚采样腾空或规定所有trot必须腾空。本诊断没有完成新的步态视觉验收。

## 旧预注册逐项决定

| 预声明门 | 结果 |
|---|---|
| balanced全部21旧物理标准 | 失败：13/21 |
| common .5/.8/1.0 mean duty gap改善≥25% | 失败：12.3510% |
| 无个案gap恶化>.05 | 通过 |
| 每脚≥2完整去抖周期 | 通过 |
| 每case-foot短周期不多于control | 通过 |

matched文件原值 `diagnostic_selection_passed=false`、`promotion_performed=false`；身份审计也明确 `behavior_accepted=false`。按“失败不晋级”保留3947与原恢复1999/3547，不能把以后common-only研究范围倒用于修改本轮21/21判定，也不能用个别视频覆盖失败报告。

## 对下一滑移提案的限制

**后续状态说明（GitHub 发布时补充）：** 下述“尚未验证／没有新增训练”描述的是本文完成时刻。此后单个侧倒初态的低速连续 v3 已完成，另一个预声明减滑移对照已启动；见[连续流程视觉检查](continuous-flow-video-review-20260917.md)及[发布范围](github-snapshot-20260917.md)。这些后续工作不改变本轮步态对照未晋级的结论。

[滑移提案](commonphysics-slip-next-hypothesis-20260917.md)仍是一个**可检验但更窄的假说**：common 1.0两arm确有实际滑移超标，existing foot_slip weight -0.5→-1.0的匹配A/B可以隔离该权重对滑移的影响。42案已完成，但连续.5仍未完整验证，故提案的全部启动前提尚未满足，本文不授权自动启动或延长任何训练。

需明确以下限制，不能将其包装为“下一轮自然小跑/快跑训练已找到正确答案”：

1. balanced4246是本轮被拒的研究候选，不是已验证自然基线。若未来独立实验仍选择其为起点，两arm必须从同一冻结final完整resume、固定seed42及128×300预算，其他权重/physics/commands不变；结果只解释新增标量作用，不能追认本轮成功。
2. 现有训练foot_slip使用**history norm>1 N的平面脚速求和**，评估slip使用**current norm>5 N所选接触脚的条件均值**，而上述duty用**current垂直5 N**。这些口径不同；加大惩罚可能选择更慢或更长期单对支撑，不保证提升目标评估，更不保证自然性。
3. 滑移达标但速度下降、四脚不抬、站停/转向/推扰/连续恢复起步退化，都必须拒绝。尤须保留上述完整duty和stance/swing结果：**即使1.0速度+slip过关，只要这种支撑偏置尚未解决，就只能说两个数值指标通过，不能称自然trot/快跑。** 下轮自然性目标必须在运行前独立声明；不能把本轮25%不足改名为足够，也不能再用“相对改善”代替绝对动作质量。
4. 用户的核心是同一common物理内恢复→严格站稳→运动→停止的完整功能保留，不天然要求单一actor跨两套physics皆过；但原21/21规则仍约束本轮。未来common部署研究可单独界定范围，original旧环境回归仍完整公开，3947原功能与模型保留。不得把保留旧文件当成新连续流程已实测通过。
5. 最終的style命名仍需脚间相位、完整周期/摆动、滑移、机身姿态、实际前/斜侧视频及连续流程证据。速度数字、训练reward或身份hash都不是动物模仿/论文复现证据。当前没有新增训练，更没有新增完整流程成功视频。

## 文件与hash证据

- [完整matched comparison](../evaluations/20260917-gait-balanced-4246-first/matched_comparison.json)，SHA `3bec33ce08de6ead1672b43d866a3afea564ffc8743edac385ae97caf9b4726c`。
- [control全部21案summary](../evaluations/20260917-gait-control-4246-first/summary.json)，SHA `6bf38ccf74f3ed18a8e1f4d5fc6682b65707eeed07bad58ee5ba9afc0aeb0e6d`。
- [control身份审计](../evaluations/20260917-gait-control-4246-first/identity_audit_v1.json)，SHA `9da94ad375b133e65ec7b162dcac51ca985b1cc0566cb8bf9568082b0f2fc1b3`，每arm18,900记录、105唯一路径。
- [balanced全部21案summary](../evaluations/20260917-gait-balanced-4246-first/summary.json)，SHA `213a6351d1f909fa8a800131c0cdd2eb75d02b328b0e6078200fbaad2a83d188`。
- [balanced身份审计](../evaluations/20260917-gait-balanced-4246-first/identity_audit_v1.json)，SHA `47db123af51fca8689fc6341aa004a4fe7049db8ef2f9c420383e21f492517be`，每arm18,900记录、105唯一路径。

comparison source SHA `7518b1e1431207d7779778348a48ab825f7855d6d4f9f2b2974d3a01c2ece702`；独立identity auditor source SHA `f21c7872ea72d935a2eb458a4a25a6d44955a1938d222c2dc5f9d08815a1cc9f`。原预声明见 [commonphysics-gait-next-experiment-20260917.md](commonphysics-gait-next-experiment-20260917.md)。对应六个gait CSV的完整路径/hash均保留在matched comparison，原始完整周期列表也在其中，没有删除或替换。本次只读PowerShell与新增本文档；未运行Python/仿真、未改现有pin或任何旧结果。
