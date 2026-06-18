# MACT-LeRobot-PushT：基于 LeRobot 的记忆增强型动作分块模型

## 1. 项目简介

本项目基于 LeRobot 框架，围绕长时序机器人操作任务中的历史信息利用不足问题，对 ACT（Action Chunking Transformer）算法进行改进，设计并实现了一种记忆增强型动作分块模型 MACT（Memory-Enhanced Action Chunking Transformer）。

在机器人操作任务中，传统基于单帧观测的模仿学习策略往往难以充分利用历史状态信息。当任务中存在接触变化、目标滑动、姿态调整和连续反馈时，仅依靠当前帧图像可能导致模型无法准确判断任务阶段和物体运动趋势，从而出现动作不连续、状态判断不稳定和误差累积等问题。

针对上述问题，本项目在 ACT 动作块预测机制的基础上，引入短时历史信息建模方法。MACT 通过特征缓存机制（Feature Cache）保存近期视觉特征，并利用时间编码（Time Encoding / Time Embedding）显式区分不同历史帧的时间顺序，使模型在当前动作预测过程中能够同时参考当前观测和短时历史信息。

本项目以 PushT-v0 任务为实验平台，对 ACT 基线模型和不同观测窗口设置下的 MACT 模型进行训练与评测，并从成功率、平均累计奖励和平均最大奖励等指标分析模型性能。

---

## 2. 项目特点

本项目主要包含以下几个特点：

1. **基于 ACT 的轻量化改进**
   MACT 保留了 ACT 一次预测未来动作块的优势，仅在输入侧引入短时历史信息建模机制，改动成本较低。

2. **特征级历史信息组织**
   项目没有直接拼接多帧原始图像，而是在 ResNet-18 提取视觉特征后，通过 Feature Cache 组织短时历史特征，降低输入冗余。

3. **显式时间顺序建模**
   通过 Time Embedding 为不同时间步的特征加入时间位置信息，使模型能够区分当前帧和历史帧。

4. **多组对比实验**
   项目完成了 ACT、MACT-1、MACT-2 和 MACT-3 等多组实验，对不同观测窗口下的模型表现进行对比分析。

5. **完整训练与评测流程**
   项目包含训练、保存 checkpoint、批量评测、结果整理和曲线绘制等完整实验流程。

---

## 3. 方法概述

### 3.1 ACT 基线模型

ACT 是一种动作分块策略，其核心思想是：模型在每个决策时刻不只预测当前一步动作，而是一次性输出未来一段动作序列，即 action chunk。相比逐时刻预测单步动作，动作块预测能够减少频繁重规划带来的误差传播，在一定程度上缓解长时序任务中的误差累积问题。

ACT 的基本流程可以概括为：

```text
当前图像观测 / 状态输入
        ↓
视觉编码器提取特征
        ↓
Transformer 建模
        ↓
输出未来动作块
        ↓
滚动执行部分动作
```

虽然 ACT 在长时序任务中具有一定优势，但其输入侧对历史信息的显式利用仍然不足，因此在具有明显动态变化的任务中可能难以准确判断物体运动趋势和任务阶段。

---

### 3.2 MACT 改进思路

MACT 的设计目标是在保留 ACT 动作块预测机制的基础上，引入短时历史上下文，使模型在当前时刻进行动作预测时能够同时参考当前观测和近期历史信息。

MACT 的整体流程可以概括为：

```text
多帧图像观测
        ↓
ResNet-18 视觉特征提取
        ↓
Feature Cache 缓存近期历史特征
        ↓
Time Embedding 注入时间顺序信息
        ↓
Transformer 融合时序特征
        ↓
输出未来动作块
```

与直接拼接原始图像不同，MACT 在视觉特征层面进行历史信息组织。这样既能够保留与任务相关的视觉语义信息，又可以减少多帧原始图像输入带来的计算冗余。

---

### 3.3 Feature Cache 机制

Feature Cache 用于缓存最近若干时刻的视觉特征。设当前时刻为 `t`，视觉编码器为 `f(·)`，则当前图像观测 `I_t` 的视觉特征可以表示为：

```text
z_t = f(I_t)
```

在当前决策时刻，模型不仅使用当前特征 `z_t`，还保留最近若干步的历史特征，形成短时历史特征序列：

```text
Z_t = [z_{t-n+1}, ..., z_{t-1}, z_t]
```

其中，`n` 由参数 `n_obs_steps` 控制。

Feature Cache 的滚动更新逻辑如下：

```text
新特征进入缓存 → 超出窗口长度时移除最旧特征 → 形成新的历史特征序列
```

需要注意的是，Feature Cache 缓存的是经过视觉编码器提取后的特征，而不是原始 RGB 图像。

---

### 3.4 时间编码机制

如果只是将多个历史特征直接输入 Transformer，模型并不能天然知道这些特征之间的时间先后顺序。因此，MACT 引入时间编码机制，为不同历史位置的特征加入显式时间标识。

加入时间编码后，历史特征序列可以表示为：

```text
z'_k = z_k + e_k
```

其中，`z_k` 表示第 `k` 个时间位置的视觉特征，`e_k` 表示对应的时间嵌入。

时间编码的作用是让模型能够区分：

```text
当前帧
前一帧
更早的历史帧
```

从而帮助 Transformer 更好地建模短时状态变化过程。

---

## 4. 项目结构

本项目文件组织如下：

```text
MACT-LeRobot-PushT/
├── README.md
├── .gitignore
├── commands/
│   ├── train_act.sh
│   ├── train_mact_obs1.sh
│   ├── train_mact_obs2.sh
│   ├── train_mact_obs3.sh
│   └── eval.sh
├── configs/
│   └── my_aug_config.json
├── docs/
│   └── code_changes.md
├── figures/
│   ├── mact_architecture.png
│   ├── feature_cache.png
│   ├── time_encoding.png
│   ├── training_inference_flow.png
│   ├── pusht_task.png
│   ├── success_rate_curve.png
│   ├── avg_sum_reward_curve.png
│   └── avg_max_reward_curve.png
├── results/
│   ├── act_results.csv
│   ├── mact_obs1_results.csv
│   ├── mact_obs2_results.csv
│   ├── mact_obs3_results.csv
│   └── summary_results.csv
├── src/
│   └── lerobot/
│       ├── __init__.py
│       └── policies/
│           ├── __init__.py
│           └── act/
│               ├── configuration_mact.py
│               ├── modeling_mact.py
│               └── processor_mact.py
├── tools/
│   ├── gen_config.py
│   └── gen_config_root.py
└── videos/
    ├── pusht_success_example.mp4
    └── pusht_failure_example.mp4
说明：
如果当前项目是在原始 LeRobot 源码基础上修改完成，可以只整理 `README.md`、`commands/`、`figures/` 和 `results/`，并在 README 中说明核心代码修改位置。

---

## 5. 实验环境

本项目实验在 AutoDL 云端服务器上完成，主要环境如下：

| 项目     | 配置                  |
| ------ | ------------------- |
| 操作系统   | Ubuntu 22.04        |
| 编程语言   | Python 3.10         |
| 深度学习框架 | PyTorch             |
| CUDA   | 请根据实际环境填写           |
| GPU    | NVIDIA RTX 4090     |
| 实验框架   | LeRobot             |
| 任务环境   | PushT-v0            |
| 数据集    | lerobot/pusht_image |

环境安装示例：

```bash
conda create -n lerobot python=3.10
conda activate lerobot

git clone https://github.com/huggingface/lerobot.git
cd lerobot

pip install -e .
```

如果项目依赖特定版本的 PyTorch、CUDA 或 LeRobot，请在此处补充：

```bash
# 示例：请根据实际环境修改
pip install torch torchvision torchaudio
```

---

## 6. 数据集与任务

本项目使用 LeRobot 提供的 PushT 图像数据集：

```text
lerobot/pusht_image
```

实验任务为：

```text
PushT-v0
```

PushT 任务要求智能体通过连续推动操作，将 T 型物体移动到目标区域并尽量匹配目标姿态。该任务虽然场景相对简洁，但执行过程中包含接触、滑动、方向调整和连续反馈等动态因素，因此适合作为验证短时记忆机制的实验任务。

---
## 7. 训练与评测命令

训练与评测脚本整理在 `commands/` 文件夹中：

```text
commands/
├── train_act.sh
├── train_mact_obs1.sh
├── train_mact_obs2.sh
├── train_mact_obs3.sh
└── eval.sh
---

## 8. 评测命令

示例：

```bash
python src/lerobot/scripts/lerobot_eval.py \
  --policy.path outputs/train/mact-pusht-obs2/checkpoints/last/pretrained_model \
  --env.type pusht \
  --env.task PushT-v0 \
  --eval.n_episodes 200 \
  --policy.device cuda
```

如果需要评测指定 checkpoint，请将 `policy.path` 修改为对应路径。

---

## 9. 实验结果

本项目主要比较 ACT 基线模型与 MACT 不同观测窗口设置下的表现。主要指标包括：

* 成功率（Success Rate）
* 平均累计奖励（Average Sum Reward）
* 平均最大奖励（Average Max Reward）

### 9.1 关键结果对比

| 模型         | 首次达到 40% 成功率 | 峰值成功率 |             峰值对应步数 | 200K 成功率 | 主要特点           |
| ---------- | -----------: | ----: | -----------------: | -------: | -------------- |
| MACT-1（一帧） |         110K |   50% |               200K |      50% | 单帧条件下最终表现最好    |
| MACT-2（两帧） |          90K |   48% |               140K |      36% | 中期收敛最快，但后期波动较大 |
| MACT-3（三帧） |          未达到 |   36% | 160K / 190K / 200K |      36% | 历史信息继续增多后性能下降  |
| ACT 基线     |         160K |   48% |               200K |      48% | 基线稳定，但前期收敛较慢   |

### 9.2 成功率变化曲线

建议在此处插入图片：

```markdown
![Success Rate](figures/success_rate_curve.png)
```

### 9.3 平均累计奖励变化曲线

```markdown
![Average Sum Reward](figures/avg_sum_reward_curve.png)
```

### 9.4 平均最大奖励变化曲线

```markdown
![Average Max Reward](figures/avg_max_reward_curve.png)
```

---

## 10. 实验结论

根据实验结果，可以得到以下结论：

1. **MACT 在部分配置下具有更快的收敛速度**
   MACT-1 在 110K 时达到 40% 成功率，而 ACT 到 160K 才进入相近成功区间，说明 MACT 在训练前中期具有一定优势。

2. **MACT-1 在最终成功率上略优于 ACT**
   在相同 200K 训练预算下，MACT-1 最终成功率为 50%，ACT 最终成功率为 48%。

3. **适度历史信息有助于中期收敛**
   MACT-2 在 90K 时达到 44%，140K 达到 48%，说明两帧历史信息能够帮助模型更快利用短时动态线索。

4. **历史信息不是越多越好**
   MACT-3 的最高成功率仅为 36%，说明继续增加历史窗口可能引入冗余信息或增加优化难度，反而削弱性能。

5. **奖励指标与成功率不完全一致**
   平均累计奖励和平均最大奖励可以反映策略过程表现，但成功率更能直接体现任务是否真正完成。

---

## 11. 不足与改进方向

当前项目仍存在以下不足：

1. 实验主要集中在 PushT 单一任务上，尚未扩展到更复杂的机器人操作任务。
2. 当前记忆机制属于短时历史增强，不是真正意义上的长期外部记忆结构。
3. 实验主要基于单次训练结果，后续需要加入多随机种子重复实验。
4. MACT-2 和 MACT-3 在后期存在一定性能波动，说明历史信息组织方式仍需优化。
5. 当前实验仍停留在仿真环境，尚未部署到真实机器人平台。

后续可以从以下方向继续改进：

* 引入自适应历史帧选择机制；
* 调整历史帧间隔，避免相邻帧信息过于相似；
* 尝试注意力权重可视化，分析模型是否真正利用历史帧；
* 在 ALOHA、Isaac Sim、MuJoCo 等更复杂任务中验证；
* 探索与 Diffusion Policy、OpenVLA 等方法结合。

---

## 12. 项目收获

通过本项目，主要完成了以下工作：

1. 学习并理解 ACT、Transformer、ResNet-18、模仿学习和动作块预测等关键技术；
2. 基于 LeRobot 框架完成 PushT 任务训练环境搭建；
3. 在 ACT 基础上实现 MACT 记忆增强模型；
4. 完成 Feature Cache 和 Time Embedding 等核心模块设计；
5. 完成 ACT、MACT-1、MACT-2、MACT-3 多组训练与评测；
6. 对成功率、平均累计奖励和平均最大奖励进行统计分析；
7. 总结不同观测窗口对模型性能的影响。

---

## 13. 注意事项

本项目为本科毕业设计阶段完成的探索性工作，主要用于验证短时记忆增强机制在 PushT 任务中的可行性。当前代码和实验结果仍有进一步完善空间，暂不代表 MACT 在所有机器人操作任务中均优于 ACT。

---

## 14. 参考方向

本项目涉及的主要研究方向包括：

* Robot Learning
* Imitation Learning
* Action Chunking Transformer
* Vision-Action Policy
* Transformer
* ResNet-18
* Feature Cache
* Time Encoding
* LeRobot
* PushT-v0

## 15. 说明

本项目为本科毕业设计阶段完成的探索性研究，主要用于验证短时记忆增强机制在 PushT-v0 任务中的可行性。当前结果仅代表本实验设置下的阶段性表现，不代表 MACT 在所有机器人操作任务中均优于 ACT。

本仓库不包含完整 LeRobot 框架、原始数据集和模型权重。如需复现实验，请先安装官方 LeRobot 框架，并根据 `docs/code_changes.md` 将 MACT 相关代码放入对应路径。