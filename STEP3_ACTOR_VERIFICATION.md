# 步骤3：冻结 Actor 的逐段前向复现

状态更新：步骤3已在固定运行`outputs/actor_verification/20261030_105109_547956`通过全部280条/25,326步核对，后续纯SAE36次训练也已完成。当前下一项为[状态辅助配对方案](../../research/embodied_sae/STATE_ASSISTED_SAE_PLAN_2026-11-01.md)；新窗口见[主交接](../../项目交接文档_具身策略决策解释_2026-11-01.md)。下面是步骤3自身的复现说明，接手无需先重复全量校验。

直接运行项目根目录的 **3_verify_actor_forward.py**。在 Visual Studio 中选择这个文件运行即可，继续使用 MachineLearning 环境；无需启动原来的轨迹窗口、服务器、SAPIEN 或旧版 Stable-Baselines3。

默认程序先完成数值校验、导出报告和图表，然后打开原生结果窗口。计算期间控制台每完成25条轨迹会显示进度。只想查看上次结果时，使用 `--view-only`，不用重复计算。

## 计算范围

恢复 `320 → Linear64 → ReLU → Linear64 → ReLU → Linear22` 的 actor 后半部分，严格加载原检查点中的6个参数张量，共26,134个参数。参数保持 float32、eval 模式、requires_grad=False，前向使用 no_grad；不创建优化器。

默认权重来源为本机 `D:\homework\dexart_assets.zip` 内的 `assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip`。先核对内层检查点SHA256与采集清单一致，再通过 `torch.load(weights_only=True)` 读取张量；SB3配置中的序列化Python对象不反序列化。不会解压整个资源包或导入仿真栈。

当前数据是确定性策略采集，动作均值后直接按 [-1,1] 裁剪。融合特征保存在原始数值轨迹的 features_fused 中。此实现不恢复 PointNet，不验证观测到融合特征的计算，也不重放物理状态。

| 比较入口 | 逐段前向计算 | 比较项 |
| --- | --- | --- |
| A | 已保存 h2 → 动作头 | A_mu、A_action |
| B | 已保存 h1 → 第二层线性/ReLU → 动作头 | B_h2、B_mu、B_action |
| C | 已保存融合特征 → 两层线性/ReLU → 动作头 | C_h1、C_h2、C_mu、C_action |
| D | 已保存 μ → clip | D_action |

检查范围从靠近输出的一段逐步扩大，但每次执行的方向始终是前向；没有反向传播或训练。钩子挂在原模块名对应的两个ReLU和动作头，复制输出并返回None。每批检查挂钩调用次数和形状；退出后移除全部句柄。另做一次无钩子前向，确认输出逐值相同，并比较所有权重在运行前后的哈希。

## 判定规则

运行前把判定参数写入 configuration.json，默认逐元素要求：

```text
abs(replayed - reference) <= 1e-5 + 1e-4 * abs(reference)
```

误差使用 float64 计算，网络前向仍为 float32。每项保存 MAE、RMSE、最大绝对误差、超差数值和超差时间步数量，并保留最大误差的轨迹、动作行、维度、记录值和重算值。总体数值均值与逐轨迹宏平均分别保存，split 不合并解释为泛化结论。

原始动作均值和裁剪动作分别比较：裁剪可能掩盖原始均值误差。10项比较都需通过，且权重未变化、钩子中立检查通过，最终状态才是 passed。程序不会因发现误差自动调宽阈值。超差值明细最多导出5000行，完整数值统计不截断。

所有比较都针对真实动作行0…T−1。额外的终止状态s_T没有新激活或动作，在 terminal_states.csv 中记为“不适用”，不填零计算误差。step_errors.csv 可用 `(trajectory_id, action_index)` 与信息总表的 `(trajectory_id, state_index)` 对应；主表中终止状态对应的误差应保持缺失。

## 在结果窗口检查

1. **全局误差**：10个比较项的 MAE、最大误差及超差数值数量；上图使用 symlog 以容纳真实零值。
2. **重算与记录散点**：C入口的两层激活、原始动作、裁剪动作；若一致应贴近 y=x。仅绘图确定性抽样，每项最多5000点，统计没有抽样。
3. **逐时刻误差**：查看某条轨迹每步的最大/平均误差、超差维数，并对照开启量。
4. **指定维度对照**：选择神经元或动作维度，查看保存值与重算值的曲线以及带正负容差的残差。
5. **误差热图**：同时查看整条轨迹的所有对应维度，色条标明误差数量级。
6. **原始动作与裁剪**：选择动作行，对照全部22维原始均值和实际动作。
7. **来源与判定说明**：显示当前运行的设备、精度、批大小、阈值及原权重SHA256。

默认定位某比较的最大误差行/维度，仅用于排查计算一致性，不用于挑选语义特征。窗口“保存当前图”和Matplotlib工具栏保存按钮均写入当前结果目录的 user_exports/。本次不代表神经元语义、干预效果或闭环开盖能力已验证。

## 独立脚本用法

```bat
conda activate MachineLearning
python 3_verify_actor_forward.py
python 3_verify_actor_forward.py --view-only
python 3_verify_actor_forward.py --no-gui
python 3_verify_actor_forward.py --trajectory-id object_9996__episode_00004 --no-gui
python 3_verify_actor_forward.py --checkpoint "D:\其他目录\laptop_nopretrain_0.zip"
python 3_verify_actor_forward.py --device cuda --no-gui
```

默认用CPU、1个线程、批大小256，降低小网络多线程开销。`--device cuda` 是可选项；如果CUDA不可用会明确报错，不暗中切换设备。跨设备结果应保留不同运行记录，不能为了通过而反复改变阈值。`--atol` / `--rtol` 可显式设置，但每次运行都会记录。

## 输出组织

```text
outputs/actor_verification/
├─ README.md / latest.json          # 最近完成的结果入口
├─ logs/<时间戳>.log
└─ <北京时间时间戳>/
   ├─ configuration.json           # 在前向前固定的参数
   ├─ summary.json / README.md
   ├─ comparison_summary.csv       # 全部比较的汇总
   ├─ by_split_summary.csv
   ├─ trajectory_errors.csv        # 每轨迹、每比较项
   ├─ step_errors.csv              # 每动作步
   ├─ dimension_errors.csv         # 每神经元或动作维度
   ├─ mismatches.csv               # 超差位置，无超差时只有表头
   ├─ terminal_states.csv
   ├─ hook_checks.json / sources.json
   ├─ model/                      # 冻结actor副本及来源说明
   ├─ replayed/<轨迹ID>.npz        # 保存值、重算值、动作索引和数值状态
   ├─ figures/                    # 全局图和开发集代表轨迹图
   ├─ user_exports/
   └─ source_snapshot/
```

源数据、现有信息总表和原来的轨迹窗口不被修改。若模型/文件/形状不匹配则停止并记录错误；若数值比较超差则保留失败报告和图形，命令行返回非零退出码，不能将失败运行当成通过。
