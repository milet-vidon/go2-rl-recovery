# 论文与源码复现审计：恢复方向尚需实验验证

审计日期：2026-09-16。范围是本机 Go2 仿真；不执行任何第三方实机部署命令，不替换现有 Python/Isaac Lab 环境。公开仓库里的演示和成功率不等于本项目已经复现。

## 当前证据与结论

旧恢复模型及几次续训仍未解决稳定落地后的翻身。最近 NominalTarget512 实验在约584轮发生 PhysX CUDA 显存不足；完整保存的500检查点复测侧躺0/20、背躺0/20，全部起点符合落地资格。侧躺最终平均倾角61.82°、高度0.149m；背躺仍约180°、0.057m。不得将中断或奖励上升视为成功。

新128环境配对现已完成：两组相同初始网络、相同PPO/奖励/状态库，各2000轮、6,144,000环境步；只比较默认关节角与控制周期开始时当前关节角作为目标参考。双方9个姿态组的成功数、片尾有效站立及最终几何均为0/20。其依据是Lee2019的动作设计，但本次结果并不奏效，也不是论文完整复现。新旧并行环境数量不同，不能把两批训练合并为同一控制变量实验。[双视角完整负结果](video-review-20260916.md)已交付；下一步转向独立奖励结构实验，不原样延长。

## 逐项核对

| 来源 | 已核实的方法或代码 | 与当前 Go2 实验的差异 | 采用边界 |
| --- | --- | --- | --- |
| [Lee et al., 2019](https://arxiv.org/html/1901.07517)，II-D | 将翻正、起立和移动作为独立行为训练；翻正/起立的关节目标参考当前关节位置；运动平滑惩罚有训练课程 | 当前只测试动作参考，仍是单策略、固定奖励和名义PD落地状态库；机器人及算法也不同 | 支持动作表示这一可检验假设，不等于完整复现或成功证明 |
| [Smith et al. 官方代码](https://github.com/lauramsmith/fine-tuning-locomotion/tree/583f1de43e91cdd24d632d783872528eb1337480) | 可训练恢复任务，有恢复检查点，Apache-2.0；单策略翻正／起立分段奖励，真实被动落地调用链 | A1/PyBullet/SAC(可选REDQ)，非Go2/Isaac Lab/PPO；动作处理、传感历史、初态及奖励不同 | 当前配对若仍停滞，首选逐项移植的参考；保留Go2动力学，不直接加载A1权重 |
| [Deng et al., 2025](https://arxiv.org/html/2506.05516v1) | 随机机身和关节、零力矩落地2s、动态回合奖励与课程；关节动作参考默认位置 | 平台是轮足KYON/Go2-W；本文Go2没有轮子，当前初态有名义PD；论文Eq.2与Eq.4的DS覆盖范围也不能混为一谈 | [作者项目页](https://boyuandeng.github.io/L2R-WheelLegCoordination/)本次未确认可下载完整训练源码；不能声称直接复现，也不能套用文中成功率 |
| [MetalHead 固定源码](https://github.com/inspirai/MetalHead/tree/1f55679c1e42300d12f25a6e03f7f5bbf1b5a691) | AMP训练、动作数据及示例权重；配置有180/300/300Nm的关节effort，且环境实际使用 | 远高于本实验23.5Nm；其随机姿态高处释放不等于本项目的落地恢复协议 | 可供后续动物式动作先验研究，不能靠搬入高力矩设置获得虚假恢复效果 |
| [RAAI2024 A1恢复源码](https://github.com/yerenkl/quad-fall-recovery-rl/tree/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8) | PPO及完整环境源码，动作映射URDF关节范围 | 未发现许可证、依赖锁定和测试所需权重；复位未明确清除旧位置PD/动作历史，测试还会在奖励阈值后保持旧动作 | 只作方法对照，不复制进本项目许可证下，也不采用其测试作为本项目验收 |

## 首选后续基准：Smith 官方恢复任务

固定提交 `583f1de43e91cdd24d632d783872528eb1337480`；[作者项目页](https://xbpeng.github.io/projects/Finetuning_Locomotion/index.html)链接官方仓库。已核对训练入口、环境构造、任务、基础机器人与A1执行层，而非仅阅读README。

### 奖励：单策略，不是两个网络切换

[reset_task.py](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/envs/env_wrappers/reset_task.py)令 `c=world_up · body_up`。翻正奖励为 `((1+c)/2)^2`；只有 `c > cos(0.2π)`（倾斜小于36°）才加入站起奖励。站起项由高度、逐关节加权姿态误差及关节速度组成，最终以等权合并翻正项和站起项。A1目标高度和关节姿态不能作为Go2标准站姿直接照搬。原文件还包含单独RollTask/StandTask，但默认`--train_reset`实际创建的是ResetTask。

### 初态与控制调用链

[env_builder.py](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/envs/env_builder.py)构造恢复任务，采用33个1ms物理步/控制步、150控制步/回合，约4.95s。初态混合20%站立、20%坐姿、60%倒地。倒地关节偏向站姿插值到限位，而不是始终默认腿形。

[`locomotion_gym_env.py`](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/envs/locomotion_gym_env.py) → [`minitaur.py`](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/robots/minitaur.py) → [`a1.py`](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/motion_imitation/robots/a1.py)确认：模拟重置清零基座速度，并令关节速度控制的force为0；`reset_duration=0`不会进入站姿PD，随后任务执行1000物理步被动落地。另有`robot.is_safe`早停，不能简单称作只有超时结束。

目标角采用默认姿态加动作，但执行层又将目标限制在当前角±0.2rad及关节范围的交集，并有动作滤波/插值。这既不是本项目旧nominal+.25a，也不等于Lee式current+.25a。原A1的Kp100、Kd[1,2,2]和35.5Nm不可直接替换当前Go2参数。

[sac_configs.py](https://github.com/lauramsmith/fine-tuning-locomotion/blob/583f1de43e91cdd24d632d783872528eb1337480/sac_dev/sac_configs.py)使用SAC，可选REDQ，非PPO。默认预算不是论文实际耗时证据；不能据此承诺本机若干分钟必定训练成功。

## 有界决策，不继续无变化堆轮数

1. 已完成128环境动作参考配对及相同协议的确定性测试，双方失败、不晋升；双视角视频已实际检查并交付。不能因奖励高或随机采样偶然翻身就晋升。
2. 如果仍无真实倒地恢复，**不要再次原样延长**。先创建独立Smith-inspired奖励任务，保持同一Go2、动作、初态库和PPO，单独检验翻正阶段不强迫最终站姿的奖励结构。此时应称方法级适配，不称完整论文复现。
3. 再单独收集随机关节、真正零驱动力矩的被动落地状态库。当前DCMotor的零策略动作仍有PD；仅把目标设0也不是被动。要在执行器输出层验证每物理步确为零驱动力矩，并做冷启动复放。两个状态库的控制方式、schema、哈希和结果必须分开。
4. 只有出现明确进展，才进一步匹配动作滤波、历史观测或SAC。一次同时更改所有机制会失去失败原因的可追踪性。
5. 最终验收继续使用当前四足垂直支撑、无机身触地、不交叉、连续3秒且片尾有效；另测正常站立/走停、新未见初态与种子，并实际查看正面及斜侧面视频。不得放宽标准或提升虚构力矩来得到“成功”。

本次只读核对外部资料，没有将第三方源码混入作品集、安装其依赖或执行其检查点。记录的是可复核方向及下一步试验，不是尚未得到的成功结论。
