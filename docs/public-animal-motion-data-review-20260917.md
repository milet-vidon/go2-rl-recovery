# 公开动物示范数据路线核查

截至 2026-09-17 18:12 +08:00。仅研究：读取官方网页、代码及几个不足 33 KB 的文本片段到内存；未保存数据、下载模型、安装依赖、运行 Python/仿真或启动训练。

结论：**公开动物示范路线确实存在，不能由另一个 Go2 仓库的数据未公开，推断 AMP 无数据可用。** 但现有文件不是 Go2 即插即用策略，数据的非商业许可也不能被代码根许可证覆盖。建议单独开展 Go2 重定向与数据质量小试，不把当前手写步态奖励实验称为动物模仿复现。

## 1. 实际拿得到什么

`erwincoumans/motion_imitation` 核查版本为 `d0e7b963c5a301984352d25a3ee0820266fa4218`。实际解析结果如下；时长按仓库加载器的 `(帧数−1)×FrameDuration` 计算，不把最后重复边界多算一帧。

|公开文件|实际结构|时间信息|
|---|---|---|
|[dog_trot.txt](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/data/motions/dog_trot.txt)|33 帧，每帧 19 数：根位置 3、四元数 4、关节角 12|dt=0.01667 s，周期 0.53344 s|
|[dog_pace.txt](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/data/motions/dog_pace.txt)|39 帧，每帧同为 19 数|dt=0.01667 s，周期 0.63346 s|
|[dog_trot_joint_pos.txt](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/retarget_motion/data/dog_trot_joint_pos.txt)、[dog_pace_joint_pos.txt](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/retarget_motion/data/dog_pace_joint_pos.txt)|分别 33/39 行、81 列，即每帧 27 个 xyz 关键点；逗号分隔|与机器人 12 个关节角不是同一表示|

前两文件均 `Wrap`、允许周期位置偏移、不允许周期旋转偏移；没有力矩、策略动作、真实接触标签。[加载器](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/utilities/motion_data.py)据姿态计算速度。当前[重定向脚本](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/retarget_motion/retarget_motion.py)默认导入 **Laikago** 配置，A1 导入行被注释；不能仅凭 `dog_trot` 文件名认定其为 A1/Go2 数据。脚本用源髋/脚关键点、尺度、机器人 IK、关节限制和默认姿态做重定向；其零重力逐帧摆姿展示不是动力学可行性验证。

同一脚本明确列出 pace 来自 `dog_walk00[162:201]`，trot 来自 `dog_walk03[448:481]`，另有 trot2/canter/turn 源片段；[原始名称映射表](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/retarget_motion/data/dog_clips_info.txt)保留 `D1_...` 到片段名及裁剪信息。应保留这条来源链，不能只记录最终动作文件名。

## 2. 数据许可不能混同代码许可

仓库[根 LICENSE](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/LICENSE.txt)是 Apache-2.0，**但 [retarget_motion/data/LICENSE.txt](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/retarget_motion/data/LICENSE.txt)单独规定 mocap 为 CC BY-NC 4.0，并指向 AI4Animation**。上游 [AI4Animation Copyright Information](https://github.com/sebastianstarke/AI4Animation#copyright-information)再次区分研究/教育用途和非商业数据许可；2018 四足项目关联 Zhang、Starke 等人的动物动捕数据。

[CC BY-NC 4.0 正文](https://creativecommons.org/licenses/by-nc/4.0/legalcode)允许依其条件进行非商业共享/改编，需要署名、许可信息和改动说明。个人作品集不应自动等同于非商业许可已满足；商业用途需另确认授权。未找到 `motion_imitation/data/motions` 内独立放宽这些衍生数据限制的声明，因此保守地保留上游数据限制，不把重定向数据重新宣布为 Apache/MIT。训练权重及生成演示的法律地位也不能仅凭此代码许可证推定；正式公开数据/模型前应单独确认。以上为许可文本核查，不是法律意见。

## 3. AMP 官方路线也有公开数据，但并非 Go2 完整复现

[AMP 项目页](https://xbpeng.github.io/projects/AMP_Locomotion/index.html)的旧代码地址 `Alescontrela/AMP_for_hardware` 本次返回 404；通过作者当前账号找到其[官方仓库 escontra/AMP_for_hardware](https://github.com/escontra/AMP_for_hardware)，不是第三方替代实现。核查版本 `bfb0dbdcf32bdf83a916790bddf193fffc7e79b8`。[论文 III-C](https://arxiv.org/html/2203.15103v1#S3.SS3)说明将 Zhang/Starke 德牧关键点重定向到 **A1**，经 IK/FK 和有限差分生成示范状态，并从示范相邻状态学习 style reward；这与手写占空比/滑移奖励有本质区别。

实际浏览 `datasets/mocap_motions` 目录得到 `leftturn0、rightturn0、pace0、pace1、trot0、trot1.txt` 六个文件。[trot0](https://github.com/escontra/AMP_for_hardware/blob/bfb0dbdcf32bdf83a916790bddf193fffc7e79b8/datasets/mocap_motions/trot0.txt)实际为 33×61 数，dt=0.021 s，周期 0.672 s，`MotionWeight=0.5`；不是上述 19 维文件的直接改名。论文描述含 canter，但所浏览的这个目录没有同名 canter 文件：不能据论文就承诺当前六文件等同完整实验数据。

[AMPLoader](https://github.com/escontra/AMP_for_hardware/blob/bfb0dbdcf32bdf83a916790bddf193fffc7e79b8/rsl_rl/rsl_rl/datasets/motion_loader.py)明确分块：根位姿 7、关节位置 12、局部脚位置 12、根线/角速度各 3、关节速度 12、局部脚速度 12；还把 PyBullet 的 FR/FL/RR/RL 转为 Isaac Gym 的 FL/FR/RL/RR。其训练状态读取与 discriminator 特征子集应照实现核对，不能把 61 维全帧、AMP 特征、策略观测混用。[A1 配置](https://github.com/escontra/AMP_for_hardware/blob/bfb0dbdcf32bdf83a916790bddf193fffc7e79b8/legged_gym/envs/a1/a1_amp_config.py)用 42 维策略观测、48 维 privileged observation、不同 PD 和控制间隔，并非当前 Go2 的观测/控制接口。[仓库软件许可证](https://github.com/escontra/AMP_for_hardware/blob/bfb0dbdcf32bdf83a916790bddf193fffc7e79b8/LICENSE)不构成动物源数据重新授权的证据；本次未确认 AMP 数据的单独扩展授权。GitHub tree API 中途限流，未声称已穷尽所有分支/许可。

## 4. 最小下一步：先验证 Go2 数据，再决定 AMP 训练

以下是工程提案，尚未实施，不替代当前已冻结控制器，也不启动新训练：

1. 从一段公开 trot 关键点做 Go2 重定向小试，保持来源/裁剪/许可记录；pace 是另一种步态，不为增加数据量盲目混入。参考 [retarget_config_a1.py](https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/retarget_motion/retarget_config_a1.py)，但重做 Go2 URDF/USD 的名字映射、髋/脚位置、尺度、根高、IK 限制与默认姿态。A1 的 toe ID `[5,15,10,20]`、hip ID `[1,11,6,16]` 是该 URDF 索引，绝不能直接用于 Go2。
2. 显式转换坐标轴/四元数 xyzw↔wxyz 和**完整关节顺序**；当前 Isaac Lab 原生顺序不能等同旧 Isaac Gym 的按腿分组。以 50 Hz 构造相邻状态，使用正确旋转插值、统一时间差分、Go2 FK 重新计算脚位置/速度；不复制 A1 脚位置，也不凭脚高度伪造真实接触标签。
3. 先检验循环边界、关节/速度限幅、脚交叉、自碰撞、触地滑移和各腿支撑相位，再做独立有限 AMP 与无 AMP 对照。几何回放通过不等于物理跟踪通过；速度达标也不等于 trot/快跑。应以对角支撑相位、占空比、摆脚轨迹和实际视频共同检验风格。
4. 保留当前倒地恢复、站立、停止模型与评测；新 locomotion 必须在**同一实际物理配置**下复测恢复→行走→停止、低速/转向/推扰，并保留旧拒绝标准。公开数据让路线值得验证，但尚无本项目 Go2 自然小跑、快跑或参考视频完整复现的证据。
