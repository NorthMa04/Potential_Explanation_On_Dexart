# 步骤5：纯 TopK SAE 试运行

编号入口为 `5_run_pure_sae.py`，公共模块为 `temp_scripts/sae_model.py`、`sae_training.py`、`sae_plots.py`、`sae_viewer.py`。沿用步骤1—4的命名和目录规则；所有新模型、日志、缓存、代码快照和分析结果均写入本项目 `outputs/`，源数据只读。

## 由用户启动

在 Visual Studio 中选择 **5_run_pure_sae.py** 作为启动文件，使用已有 **MachineLearning** 环境，Ctrl+F5。默认只训练一组：第一层64维激活，宽度128，TopK=8，种子0。计算结束后打开原生结果窗口。项目原来的默认启动文件仍保留步骤1。

以下命令在项目目录、MachineLearning 环境执行：

```powershell
# 只检查窗口、原权重和动作对应，不训练、不更新正式结果指针
python -B 5_run_pure_sae.py --prepare-only --no-gui

# 首次正式试运行：一组配置、一个种子，结束后开窗口
python -B 5_run_pure_sae.py

# 复看最近完成结果，不重新训练
python -B 5_run_pure_sae.py --view-only

# 重看某次运行（目录路径也可使用绝对路径）
python -B 5_run_pure_sae.py --view-only outputs/pure_sae/<运行目录>

# 后续可选：四组参数，每组一个种子
python -B 5_run_pure_sae.py --compare-configs

# 后续可选：给确定的配置补齐三个种子
python -B 5_run_pure_sae.py --width 128 --top-k 8 --seeds 0 1 2

# 第二层按同标准单独运行，或只查看重建/动作而不训练状态探针
python -B 5_run_pure_sae.py --layer h2
python -B 5_run_pure_sae.py --skip-probes
```

`--no-gui` 只训练和导出。`--device cuda` 可显式选择已有GPU，默认CPU/1线程，适合此小网络；不同设备不保证逐位相同。`--quick-check` 将训练限制在最多5轮，只检查流程，不更新正式结果指针，不能当作充分训练结果。不要把prepare-only或自动化测试当成用户正式试运行。

## 默认配置与固定的方法

| 项目 | 默认值 |
| --- | --- |
| 输入 / 范围 | h1，首次已确认 sustained_opening，触发动作 j−1 前后各5步的全部可用动作前快照 |
| 数据 | 仅原有train/validation；不读取测试数值归档，不补造终止激活 |
| 网络 | 64 → 128 → 64，TopK后ReLU，线性解码器及偏置；没有输出ReLU |
| TopK | 最多8个正激活；若候选为负，实际非零数可能少于8 |
| 预处理 | train拟合轨迹加权逐维均值/标准差；标准差<1e−6的维度改用尺度1，不删除坐标；解码后逆变换 |
| 损失 | 标准化空间64维MSE，全部训练窗口帧保留、逐轮打乱，轨迹总权重相同 |
| 优化器 | Adam，lr=0.001，默认betas/eps，weight_decay=0，batch=128，保留最后不足一批的样本 |
| 停止 | 最多500轮；validation轨迹等权MSE连续50轮未创新低则停止，恢复最低MSE检查点；初始化第0轮也参与比较 |
| 字典 | 解码器列单位范数，移除平行梯度分量，每次更新后重新归一化；编码器转置初始化后独立训练 |
| 其他目标 | L1、状态辅助、动作辅助均为0；无Dropout、无物理仿真 |
| 种子 | 默认0；支持显式0/1/2，各自报告，不挑最好种子作主结论 |

参数均可通过 `--help` 查看。`--compare-configs` 固定为(128,8)/(128,4)/(128,16)/(256,8)，替代`--width`和`--top-k`；其余训练参数保持相同。每次新建运行目录，不覆盖历史。失败、手动中断、prepare-only和quick-check不会替换最近正式成功结果；检查点和定期训练历史留在当前运行目录。

## 冻结 actor 与动作误差

从原资源ZIP严格恢复同一检查点，只读当前层之后的actor。训练前先核对当前窗口原激活的读出与记录均值/动作，逐元素容差`1e-5 + 1e-4*abs(reference)`。不通过即停止。

重建的标准化激活逆变换后接回actor；负坐标不额外裁成非负，以免引入第二种未说明的重建过程，并报告其比例。激活MSE、原始动作均值和裁剪后动作误差分别保存，原actor始终eval/冻结、不进入优化器。运行前后核对actor参数哈希。

`reconstruction_metrics.csv`含原尺度/标准化激活误差和两类动作误差，按轨迹、自然帧、物体三种权重分别报告。NMSE分母是原激活相对**训练加权均值**的MSE；分母为0时留空。`p95_row_rmse_frame_weighted`始终为自然帧的逐行RMSE分位数，不能误读为轨迹等权分位数。`dimension_errors.csv`保留全部64/22维误差，避免平均值掩盖局部偏差。

不预设未经校准的“动作保真通过阈值”：完成状态仅表示流程完成，是否足以开展解释需结合试运行误差讨论。动作接近不直接证明盖子运动接近。

## 状态预测与公平比较

SAE最佳检查点冻结后，默认另拟合新线性/逻辑回归探针。目标沿用步骤4的同一步距离(m)、开启量(rad)、任一组接触；严格接触仅保留为展示，不视为抓稳标签。

同批重新计算常数、时间样条、当前层完整激活、训练相关性前1/2/4个神经元、1/2/4/8维PCA、时间+原激活，以及z全部、前1/2/4个特征、时间+z。所有预处理、PCA及坐标排序仅拟合train；正则参数按validation轨迹等权MAE/log loss从0.1/1/10选择；接触阈值固定0.5。AP是average precision，单类别分组留空；不重新训练MLP。

这些探针的状态标签不进入SAE训练、梯度或检查点选择。少量坐标按训练相关性选择，是启发式，不是最优子集。全z比原h有更多候选坐标，不能只靠全z预测更好就宣称语义更集中。跨种子的相同编号不自动代表同一特征；种子波动不是数据置信区间。

## 查看与交回结果

窗口可选择实验、开发集轨迹和22维动作编号，包含训练/动作保持、特征使用、逐步激活误差及z热图、原/重建动作、当前状态预测和指标明细。图中保存按钮统一导出到当前实验的`figures/review_*.png`。

结果目录：`outputs/pure_sae/<北京时间戳>/`。讨论时先查看：

- 根目录的`configuration.json`、`summary.json`和`actor_reference_check.json`。
- 实验子目录的`training_history.csv`、`summary.json`、`reconstruction_metrics.csv`、`feature_usage.csv`、`probe_metrics.csv`及`figures/`。
- 有疑问时查`reconstruction_by_trajectory.csv`、`reconstruction_by_object.csv`、`probe_by_object.csv`及逐维误差。

`best_model.pt`同时包含模型、结构、预处理和最佳轮数，可通过`sae_model.load_checkpoint`重载；重载结果不依赖训练期间内存。`selected_arrays.npz`与每个实验的`reconstruction.npz`按同一行对应，并由`selected_frames.csv`保留轨迹/状态身份。原actor副本、输入及代码哈希也随运行保存。

当前仅是事件条件下的开发集实验。事件v1尚待人工语义验收，validation已参与选参；尚未做独立测试、特征跨种子匹配、干预或闭环仿真。物理阈值、数据划分和既有源数据不被步骤5修改。
