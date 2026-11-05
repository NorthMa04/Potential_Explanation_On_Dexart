# 步骤4：首次持续开启窗口的原始激活 baseline

已与用户确认：**先只分析首次持续开启附近；接触目标采用任一组接触**。真实开发集拟合已完成，固定结果为`outputs/activation_baseline/20261030_163827_787919`。后续纯SAE36次训练也已完成，当前下一项为[状态辅助配对方案](../../research/embodied_sae/STATE_ASSISTED_SAE_PLAN_2026-11-01.md)，不再把本步骤描述为代码交付待跑。

公共实现模块已集中到 `temp_scripts/`，编号入口仍位于项目根目录。运行入口的方法和数据/输出位置不变；`temp_scripts/` 是运行依赖，应随项目一起保留。

## 直接运行

在 Visual Studio 中右键 `4_run_activation_baseline.py`，选择“设为启动文件”，继续使用已有的 **MachineLearning** 环境，然后 Ctrl+F5。计算结束后打开独立的原生结果窗口。项目默认启动文件仍为步骤1的轨迹审阅入口。

也可以在 PowerShell 中执行（任意当前目录均可）：

```powershell
& 'C:\Users\17764\anaconda3\envs\MachineLearning\python.exe' -B 'D:\homework\DeepLearning\PythonApplication\Potential_Explanation_On_Dexart\4_run_activation_baseline.py'
```

下面的短命令在本项目目录、MachineLearning 环境下运行：

```powershell
python 4_run_activation_baseline.py --prepare-only --no-gui
python 4_run_activation_baseline.py --skip-mlp
python 4_run_activation_baseline.py
python 4_run_activation_baseline.py --view-only
python 4_run_activation_baseline.py --no-gui
```

第一条只导出窗口、标签和覆盖情况，不拟合模型。第二条运行线性及PCA等对照；第三条额外运行三个种子的小MLP。`--view-only` 复看最近一次完整结果，不重复训练。`--quick-check --no-gui` 缩小迭代与调参预算，仅用于检查流程，不更新最近正式结果指针。

已核实本机环境包含 NumPy、Matplotlib、scikit-learn、SciPy；无需安装服务器仿真依赖，也不使用GPU。MLP的sample_weight需要scikit-learn 1.7或更新版本，本机为1.9.1。

## 数据如何进入实验

1. 只加载原有 train/validation 的数值轨迹，不加载三个测试划分的数值归档。
2. 每条轨迹找到第一个 `status=observed` 的 `sustained_opening` 事件。沿用v1规则：连续3步开启增量达到阈值后确认，事件起点与确认时刻分别记录。
3. 沿用已有片段的下标约定：事件在 `s_j` 开始，触发动作是 `a_(j-1)`，以 `max(0,j-1)` 为中心，前后各5个动作，共至多11个动作前状态。终止状态没有新激活，不补造训练行。
4. **窗口内所有可用步骤都保留、都参与拟合，不随机抽帧。** 同一步的 `h_t` 预测 `distance_t / opening_t / contact_t`，不使用post状态。
5. 没有该事件或只有终止处未确认短段的轨迹，记录在 `window_coverage.csv`，不制造锚点。这一实验只描述发生过首次持续开启的局部窗口。

默认每条纳入轨迹总损失权重相同：窗口内每步权重为 `1/窗口步数`，然后将所有训练权重归一化到均值1。完整窗口长度都为11时，这与逐帧等权相同；截断窗口也完整保留。`--weighting frame` 可改为逐帧等权。

物理目标：距离以m计；开启量是相对任务固定初始位置的rad，不按最终开启量归一化；接触目标为四指/手掌5组中**任一组接触handle_link**，不把它当作人工“抓稳”标签。上游严格接触判定另存于逐帧表，不作为本轮分类目标。事件v1尚待人工审阅。

真实窗口预检：train纳入94/96条轨迹、1029步；validation纳入31/32条、336步。严格接触正例均为0；任一组接触正例分别为584/190步。因此用户确认改用任一组接触作为本轮分类目标，事件窗口不扩大。这些是覆盖检查，不是探针预测结果。

## 对照与选择规则

| 名称 | 输入与模型 | 用途 |
| --- | --- | --- |
| constant | 训练集加权均值 / 接触先验 | 最简单的参照 |
| time_spline | 从轨迹开始的实际秒数，5节点三次样条，线性/逻辑读出 | 检查时间关联，避免仅用线性时间构造弱对照 |
| h1_linear / h2_linear | 一层完整64维激活；岭回归 / 逻辑回归 | 整层的当前状态可解码性 |
| h1_top1/2/4、h2_top1/2/4 | 训练集绝对加权相关性前1/2/4个坐标；同类读出 | 少量原始神经元的参照 |
| h1_pca1/2/4/8、h2_pca1/2/4/8 | 训练集加权PCA；同类读出 | 线性组合方向的参照 |
| h1_time_linear / h2_time_linear | 时间样条+整层激活；同类读出 | 在同样线性读出中检查增加激活的效果 |
| h1_mlp / h2_mlp | 完整64维激活；32个tanh隐藏单元 | 对照读出器容量，默认种子0/1/2分别保存 |

**少量神经元使用相关性排序这一启发式，不是穷举或最优组合搜索。** 相关坐标可能冗余；结果不应表述为“任何1/2/4个神经元的最好表现”。当前步骤5已采用相同的坐标筛选/调参规则，后续辅助SAE继续保持一致。

原始输入按训练集加权均值和标准差标准化。PCA对同一标准化激活的加权协方差分解；只报告少量方向，不把全维旋转当压缩。样条、标准化、PCA和坐标筛选仅在train拟合；validation只作正则强度选择。相邻步骤不会被拆到新的训练/验证集合。

岭回归alpha与逻辑回归C候选均为0.1/1/10；MLP的alpha为0.0001/0.01，各种子使用相同预算。回归目标在训练集标准化以便拟合，输出再还原m/rad。验证集按轨迹等权的MAE或log loss选参。MLP不再从训练帧中切随机验证集，收敛警告会保存和计数，不自动当作成功收敛。

输入不包含物体编号、事件名、事件相对步数、确认时刻、轨迹结局、轨迹最终长度或未来状态。事件相对步数只用于图上对齐。

## 怎样读结果

- `metrics.csv`：回归MAE/RMSE/R²；接触AP、precision、recall、F1、log loss、Brier及正例占比。分类阈值固定0.5，不调阈值。
- 同时报告轨迹等权、自然帧等权、物体等权结果；分轨迹、分物体明细另存。物体等权表示物体之间等权、物体内轨迹等权。
- **AP是scikit-learn的average precision，不是梯形积分PR-AUC。** 某分组只有单类时AP记为空，状态为undefined_single_class；precision或recall分母为0时留空。不能将这些缺失值当作0再盲目求平均。
- 如果训练窗口接触标签只有一个类别，探针回退为训练先验，模型产物会注明single_class_train；无法据此评价接触可解码性。
- `validation` 已被用来选参数和查看候选，不是独立测试成绩；目前不生成测试分数、显著性或置信区间。MLP种子波动分别保存，汇总图的散点不是数据不确定性的区间。
- 第一层为主分析，第二层同标准展示。神经元编号、相关性、预测精度都不能直接证明动作贡献。

所有产物保存在 `outputs/activation_baseline/<北京时间时间戳>/`：配置、窗口覆盖、逐步标签、两层激活、逐步预测、指标、坐标排序、完整调参日志、已拟合模型、来源哈希、代码快照及PNG。`failure.json`保存失败原因；失败或快速检查不会替换正式结果指针。既有源数据和历史运行目录不改写。

## 资料与仍需讨论的边界

- [Hewitt & Liang, 2019](https://aclanthology.org/D19-1275/)讨论了探针本身学习任务造成的解释问题。本项目借鉴控制读出器容量的原则，不宣称复现了该文的语言控制任务；线性读出是主参照，小MLP作为补充。
- [scikit-learn：防止数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)要求预处理和特征选择仅从训练集学习；本实现遵守原有整轨迹划分。
- [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)说明AP的计算方式；这里以AP及log loss等评价不平衡接触标签。

目前选“首次持续开启窗口”是用户明确的分析范围。它可以建立这个局部条件下的baseline，尚不能说明未能开始持续开启的轨迹为何失败。是否以后加入停滞/回退等对照，应在这轮结果和人工窗口审阅后讨论；当前不自动扩展。
