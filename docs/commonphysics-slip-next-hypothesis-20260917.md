# Common-physics 滑移：唯一下一有限训练假设（2026-09-17）

## 状态与不可越过的边界

2026-09-17 18:14 补充：下面17:30快照保留为当时提案依据，不代表当前未完成。两套21案、身份审计和步态比较已全部结束；control11/21、balanced13/21，步态占空差仅改善12.35%，原晋级判定仍失败。新独立continuous v3的balanced .5单侧倒起步全流程1004区间已通过17检查与完整逐步审计；v2因未保存传感器索引不能作为训练前置，保留其失败审计。新slip trainer已实现，正在最后审查和准备独立smoke；尚未开始正式训练。单侧倒低速通过只允许开展这个有限研究实验，不等于自然步态、完整功能或快跑通过。

**仅提案，未实现、未启动、未获得行为通过结论。** 截至本次结果读取 **2026-09-17 17:30:56 +08:00**，control的21案已完成，balanced只完成16/21；42案比较尚未完成，连续0.5 m/s流程也尚未完成验证。在这两项工作完成并审查之前，不得启动本提案，不得编辑现行冻结代码或借此自动延长训练。

当前实验仍按原预注册的 **21/21物理/旧功能保留及额外步态检查** 决策：已有失败就不能晋级。本文不把它事后改成仅看common 5/6，也不重新标记任何报告为成功。以下“同一物理中的全流程功能保留”是**未来独立实验的范围**，不是现有实验的补考或放宽。

## 已完成证据与有限解释

| 测量 | 原3947（original物理） | control4246 | balanced4246（截至读取） |
|---|---:|---:|---:|
| original旧功能 | 15/15 | 7/15 | 4/10，剩余5案未完成 |
| common物理六案 | 不作跨物理继承 | 4/6 | 5/6 |
| original 0.8，seed09/10/11实际vx | .748739 / .751960 / .755199 | .594267 / .649485 / .639772 | .609455 / .673330 / .658495 |
| original 0.5，seed09实际vx | .470208 | .378446 | .342240 |
| common 1.0，实际vx / 接触滑移均值 | 不作跨物理比较 | .936924 / .125988 | .975033 / .123287 |

速度及滑移单位m/s。common 1.0两arm均满足速度误差界，但都因滑移不小于0.12失败；control common右转另因vy=-.132383超出0.12界失败。balanced common左右转当前均过，但左转vy=-.114521已接近界限，不能宣称裕量充足。原3947在其原物理中1.0三seed均因速度不足失败（vx .840405/.842149/.832786），不是已具备快跑后被新模型丢失。

训练实际receipt表明，两arm分别从同一个3947完整状态训练128×300，旧命令、奖励、噪声、upright reset、push与PPO保留。共同真实改动是self-collision ON、nominal soft-clamped action，以及取消base-mass的[-1,+3] kg startup随机化；material和CoM原本就固定，不能把它们写成新增干预。唯一arm间处理差异是completed-duration项weight 0与-10。

control没有新duration惩罚也出现original低速/.8退化，因此该新项不是退化的必要原因。观察符合**目标物理适配伴随跨分布退化**的解释，但不能从同一训练seed、多个共同物理变更中判定是哪个因素单独造成，亦不能排除继续PPO的策略漂移。保留所有失败，不把命令空间、速度追踪或静止站姿问题混为一谈：control original left/seed09确有quiet_stand_stop失败，不能泛称所有静止表现都保留。

用户需要的是在**同一实际物理设置**中倒地恢复、站稳、运动、受扰和停止功能共存；这不天然等价于同一actor在两套不同自碰撞/动作处理/质量分布中都全通过。旧3947与原环境及其15/15功能证据继续保留；original全套回归仍应完整报告作为跨域诊断，不能隐藏。但新独立common部署范围也不应因为两域总体分数，跳过最重要的同一机器人连续流程检查。

## 现有 foot_slip 的真实语义与依据

安装的官方Spot实现 [rewards.py:237](E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py:237)：

```text
contact_i = max_over_sensor_history(norm(net_forces_w_i)) > threshold
raw_penalty = sum_over_four_feet(contact_i * norm(world_foot_velocity_xy_i))
```

来源SHA256：`14e2d61f0061f27191e23228469e590c27ffef001646812993b9d4e0cce929a6`。实际balanced训练 [env.yaml:718](E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_16-33-35_20260917-gait-balanced-128x300-first/params/env.yaml:718) 配置四足body匹配 `.*_foot`，threshold=1.0 N，weight=-0.5；该YAML SHA为 `7345a2df4932258222c4cd4d8848d65623d9fb6844f082063db6e18cda7f273c`。正式训练receipt确认该旧reward未改动。

这不是current垂直力>5 N的duty，也不等同报告中的滑移：现有评估每个sample以**当前力norm>5 N**选中contact foot，再对这些脚的平面速度取条件均值；记录与聚合见 [evaluate_go2_stand_walk_stop.py:379](../scripts/evaluate_go2_stand_walk_stop.py#L379) 和 [同文件:427](../scripts/evaluate_go2_stand_walk_stop.py#L427)。静止支撑/步态duty另用current垂直5 N。历史接触求和与当前接触条件均值在时间口径、阈值与归一化上都有差别，故加大现有惩罚是否改善评估滑移只能实测，不能代数保证。所有reward还由RewardManager乘dt=.02；weight=-1不代表每步直接减1。

## 唯一假设与固定匹配实验

**假设：** 在common物理与现有命令分布不变时，现有足端滑移惩罚强度不足；只将其weight从-0.5改为-1.0，能降低1.0 m/s接触滑移，同时保留common .5/.8、转向、受扰和停止。也可能导致减速、抬脚/占空比取巧或退化；这些都按原验收拒绝。此处不再添加新reward、curriculum、phase输入、动作滤波或物理随机化。

1. 先完成并审查当前42案、匹配步态比较、身份审计及连续.5诊断。若证据指出接触奖励/状态接口错误，先拒绝并修正提案，不能照单启动。
2. 两arm都从**同一个已冻结的balanced final4246**完整resume，仅作为实验起点，不视为已接受的新默认模型。固定SHA `882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc`，路径见下。A保留foot_slip=-0.5，B只改为-1.0；两边既有balanced_duration=-10及其他reward全相同。不能拿此前control4246充当新A，也不能一边从3947、一边从4246。
3. 每arm先独立16×2 smoke，丢弃smoke权重；正式均为 **128 env × 300更新，seed42，24 steps/env**，各921600环境步/7200控制区间。恢复actor、critic、std及全部Adam moments/counters，初始scalar LR须等于该4246保存的真实optimizer LR，不任意重置为旧3947的值。保留原训练噪声、命令范围/30%stand、重采样、reset、push及所有common物理参数。
4. 串行跑；一项标量差异与run metadata外均相同。仅选择各自预声明的最终checkpoint，不筛选中间权重，不因失败自动追加预算。完整保存失败及实际计数/配置/父模型证据。旧roll1999、stand3547、refiner3746及3947均不覆盖。
5. 对两final用同一预声明common场景/种子集合做实测，比较1.0滑移下降与其他行为是否受损；A也继续训练，才能区分“多训练300更新”和weight效果。一个matched训练seed最多支持本次有限效果，不构成一般因果或跨硬件结论。

## 功能保留与名称边界

未来新实验只可在运行前单独冻结其common物理范围及完整验收，不得回写旧实验结论。至少保留common .5/.8/左右转/push/1.0及站停全套，旧三seed矩阵不能只挑seed09；固定物理/初态产生字节相同轨迹的seed不算独立泛化。之后同一物理、同一连续轨迹测试原恢复链→150个已完成严格站稳区间→运动→停止，不清history、不重置、不换物理、不移植终态；新原始站立起步测试不能替代倒地恢复后的连续起步测试。

既有标准保持不变，尤其：

- settled站停的stance_geometry每sample均通过、current base清空，四current垂直足力>5 N的比例严格>95%；signed foot lateral>0.06 m且abs<0.30 m、knee signed lateral>0.04 m、signed前后foot x>0.08 m、每关节abs offset<0.65 rad。
- 步行速度误差严格<0.12 m/s，现有直行漂移、转向lateral<0.12及yaw误差<0.15 rad/s照旧；接触滑移均值严格<0.12 m/s，四足各自world高度p95严格>0.04 m。
- 原settled height、tilt及无reset/无base contact检查不放宽；普通站停speed p95<0.06 m/s，推扰案沿用已存在的0.20界，不新增特例。恢复阶段另保留原严格站稳height/orientation/速度/150区间标准，不用步行0.25 m高度界替代恢复标准。
- 同一common物理下不能以1.0滑移变好交换已保留.5/.8/转向/受扰/站停；脚不滑但不抬、减速未达标、占空比退化或恢复后起步摔倒均不能通过。original全部回归结果仍单独列出，保留旧3947，不偷换“原模型仍在”与“新common流程实测通过”。
- “小跑/trot/快跑/自然恢复”必须由实际接触相位、完整步周期、身体姿态、滑移及正面/斜侧面连续视频支撑。达到1.0 m/s只是速度事实，不是步态style标签，更不是动物动作模仿/论文复现。

## 可追溯的截至时刻快照

以下summary为可继续更新的文件；SHA仅标识上面读取时刻取得的字节，不宣称其后保持不变。

### control

- [summary](../evaluations/20260917-gait-control-4246-first/summary.json)，读取 2026-09-17T17:30:56.9088369+08:00，status=`completed_screening_only`，11/21；common 4/6、original 7/15。
- 该时刻summary SHA：`6bf38ccf74f3ed18a8e1f4d5fc6682b65707eeed07bad58ee5ba9afc0aeb0e6d`。
- [正式训练receipt](E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_16-26-13_20260917-gait-control-128x300-first/common_physics_training_result.json)，SHA `6ff3cb73fdb78ec33ac5ba7e66082903186692f5779dc45c6065217a8da6df6a`。
- [final4246](E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_16-26-13_20260917-gait-control-128x300-first/model_4246.pt)，SHA `ea9a8c19af0cdbcfbde3841551b7039be9ced62927fedc2790e7835aba6ee935`。

### balanced

- [summary](../evaluations/20260917-gait-balanced-4246-first/summary.json)，读取 2026-09-17T17:30:56.9305405+08:00，status=`running`，9/16；common 5/6、original 4/10。
- 该时刻summary SHA：`bdf248944f12778e34a87b9b535d86b14788914e617ec843c3d0a7139ddc1225`。
- [正式训练receipt](E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_16-33-35_20260917-gait-balanced-128x300-first/common_physics_training_result.json)，SHA `299b45ea7c89e10107c4273e1d4867e546c3017bb8e18ee48cc9664427c91b9e`。
- [final4246](E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_16-33-35_20260917-gait-balanced-128x300-first/model_4246.pt)，SHA `882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc`。

原3947基准：[18案summary](../evaluations/20260917-speed-retention-control3947/summary.json)，其中旧功能15/15、1.0为0/3；上表速度重新读取各实际report，不从不适用的summary字段猜测。现有训练设计/actual receipt背景见 [commonphysics-gait-next-experiment-20260917.md](commonphysics-gait-next-experiment-20260917.md)。本次只读分析与新文档写入未运行Python、训练或仿真，也未编辑任何活动pin。
