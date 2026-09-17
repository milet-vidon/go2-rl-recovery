# Primary-source check before the next locomotion experiment

Reviewed September17 while combined recovery videos were being recorded. This is a design note, not a claim that either paper has been reproduced.

## Command curriculum

The authors' [Rapid Locomotion repository](https://github.com/Improbable-AI/rapid-locomotion-rl) implements a command-grid curriculum and teacher/student adaptation. Its [curriculum source](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py) samples command bins by weights, tests linear AND angular reward thresholds, and increases both successful-bin weights and neighboring-bin weights. The local scalar positive-speed cap is therefore not an exact reproduction of that mechanism. The [paper](https://arxiv.org/abs/2205.02824) identifies command curriculum and online system identification as key components; its robot/results do not establish performance on this Go2 setup.

Local evidence: speedgate3996 failed old-function retention10/15 and target-speed0/3. It never reached cap1.0; a mean exponential reward threshold is not the unchanged behavioral speed-error/slip test. Do not extend this rejected branch unchanged or infer that the authors' grid curriculum has failed. A future grid-based adaptation needs explicit command-bin coverage, sufficiently many complete windows, matched-budget controls, and the existing rest/turn/push/slip retention screen. These are engineering requirements inferred from our negative evidence, not claims made by that paper.

## Animal-like motion

The authors' [Learning Agile Robotic Locomotion Skills by Imitating Animals project](https://xbpeng.github.io/projects/Robotic_Imitation/index.html) and [reference implementation](https://github.com/erwincoumans/motion_imitation) use reference-motion imitation; the repository identifies its paper branch and supplies motion clips. Consequently, hand-shaped foot spacing and velocity rewards alone do not reproduce the motion-imitation method or certify animal-like gait. Directly swapping its pretrained policies into Go2 is not justified: retargeting, observation/action interfaces, robot morphology and dynamics need separate verification.

Priority remains completing and visually inspecting the actual combined recovery controller, then new initial-state validation and an uninterrupted recovery/stand/walk/stop interface diagnostic with unchanged physical state and action history. Fast running and reference-motion learning remain independent uncompleted work; do not combine separate clips into a purported successful continuous demo.

## Local integration review

Independent read-only audit of saved control3947/roll1999/stand3547 configurations found the same ordered48-dimensional observation contract: body linear/angular velocity, projected gravity, command, default-relative joint position/velocity, and real previous12actions. All actors are128x128x128 ELU with no actor observation normalizer; nominal default joints, action scale0.25, PD25/0.5, torque23.5Nm and50Hz control match. Runtime joint ordering must still be checked, not inferred only from regex configuration.

However, control3947 uses standard default-offset JointPositionAction and self-collision OFF; recovery uses nominal ControlStepJointPositionAction with explicit soft-limit target clamp and self-collision ON. Recovery bank evaluation also disables mass/CoM randomization whereas the locomotion evaluator retains its task model randomization. Equal48-dimensional shapes are therefore insufficient evidence of safe direct composition.

Next finite diagnostic: frozen control3947 under recovery-compatible physics/action processing, commands0→0.5→0 and0→0.8→0 for the original three seeds. Record actual observation contract, native joint order, action/history, unclamped/executed targets, clamp fractions, torque and physical/randomization values. Keep all existing stand/stop/tracking/slip/straight-drift gates. Only if retained, expand the full18-case screen, then separately test actual continuous recovery→3sstrictstand→walk→stop with no reset, extraPD, cleared history or transplanted terminal state. Any regression requires isolated action-clamp/self-collision comparisons rather than turning collision off to conceal recovery problems. This plan is NOT yet implemented or executed.
