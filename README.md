# DexArt 激活、纯SAE与解释实验工程

## 2026-11-05最新接续入口

先读[11月5日主交接](../../项目交接文档_具身策略决策解释_2026-11-05.md)和[整体思路与下一步流程](../../research/embodied_sae/INTERPRETABILITY_CHAIN_AND_NEXT_STEPS_2026-11-05.md)。在线补充实验、冻结读出/时序/组分析、[7起点持续接触协调](outputs/contact_coordination_20261105/CONTACT_COORDINATION_2026-11-05.md)、[近似快照匹配](outputs/snapshot_matching_20261105/SNAPSHOT_RESULTS_2026-11-05.md)及[单状态动作反查](outputs/action_backtrace_20261105/ACTION_BACKTRACE_2026-11-05.md)已有保存结果。下一窗口先以这些病例对齐关键窗口的表示—命令—物理过程，形成可检验预测；新受控快照、候选参考替换、闭环及动作路径恢复尚未完成。SAE有用性不要求胜过Ridge，未证明独立贴掌/开盖模块。服务器最后记录为平台关机，本次仅更新文档，接手不必重训或启动服务器。

## 2026-11-04历史接续记录

先读[11月4日主交接](../../项目交接文档_具身策略决策解释_2026-11-04.md)、[本轮最终报告](outputs/local_interpretability/20261104_mainline_v3/LOCAL_RESULTS_2026-11-04.md)及[步骤14](STEP14_LOCAL_INTERPRETABILITY.md)。入口`14_run_local_interpretability.py`和3个公共模块已完成本地信息/编辑/机制分析与独立核验；已有结果可读取，接手不必重跑。重要修正：119条历史物理失败均曾达到目标进度且近距离，但未同时满足严格接触，不能解释为打不开盖子。新的物理增强、自然失败救援与新起点评价仍待开展。下方为历史阶段记录。

## 2026-11-03历史运行状态

组合物理扩展7216条已全部完成，8起点各902条，运行报错0。用户本地步骤13已完成1365状态与409500变体。完整回传、校验及服务器关闭状态见[最新交接](../../项目交接文档_具身策略决策解释_2026-11-03.md)。下文保留各阶段历史记录。

本项目包含轨迹审阅、总表、冻结actor前向复现、原始基线、纯/状态辅助SAE、独立评价及离线动作干预。原策略保持冻结。下一窗口先读[最新主交接](../../项目交接文档_具身策略决策解释_2026-11-05.md)。下方涉及“尚未开展”的旧状态按对应日期理解，当前进度以上方最新入口为准。

2026-11-02完成75配置×3种子的225次训练；冻结7配置/21模型后完成三类独立测试，新增时间控制和监督Ridge旋转强参照，完成58,110个固定输入干预/匹配对照变体。最终89项回归通过。见[完整报告](outputs/state_assisted/20261102_122123_738477/FINAL_REPORT_2026-11-02.md)及[步骤7方法](STEP7_STATE_ASSISTED_SAE.md)。受限辅助头有集中收益，相对强监督旋转的额外证据主要在开启量；没有SAE物理运动/成功率结论，不需要重跑225次训练。

随后按用户授权从原CPU模式实例复制到P4，基础CUDA/EGL离屏渲染及原策略32步开盖试运行均通过，原数据341文件SHA256一致，见[迁移记录](../../research/embodied_sae/SERVER_CHECK_AND_MIGRATION_2026-11-02.md)。最新辅助SAE尚未部署；接入前需跨Python/PyTorch版本数值导出/前向核验，特征干预还需处理合法域约束。

用户随后授权开始物理仿真并明确批准上传模型包。步骤9已导出并在P4的CPU/CUDA核验6个纯/辅助SAE及冻结Top4探针；初始scene.pack/unpack续跑重复门限失败，因此改为每条件独立新进程重放完整前缀并逐值核对。首批104条完整轨迹（4自然+4独立重复+96局部替换）全部完成并独立审计通过，原策略4/4与替换96/96均成功，没有成功率提升证据；见[物理报告](outputs/physical_simulation/20261102_pilot_v1/PHYSICAL_REPORT_2026-11-02.md)和[步骤9](STEP9_PHYSICAL_SIMULATION.md)。

随后用户授权全权推进，步骤10完成330条受约束开启量Top1物理轨迹并独立审计：8预设起点中7触发，315方向编辑/匹配对照均非负、范数匹配；原策略4/8成功，全部条件成败与各自基线相同。目标减随机均值的5步开启差约+0.255度（抑制）/−0.038度（增强），跨种子符号不一致，没有稳定语义因果证据。[步骤10](STEP10_CONSTRAINED_FEATURES.md)及[完整特征干预报告](outputs/constrained_physical/20261102_opening_v1/FEATURE_REPORT_2026-11-02.md)包含代码、原始证据和补充恢复诊断。

## 按流程选择入口

| 步骤 | 运行入口 | 工作内容 |
| --- | --- | --- |
| 1 | `1_review_trajectories.py` | 读取、校验、物理事件和连续轨迹窗口审阅 |
| 2 | `2_build_information_table.py` / `2_query_information_table.py` | 构建及查询逐状态总表 |
| 3 | `3_verify_actor_forward.py` | 冻结actor后半网络的前向复现 |
| 4 | `4_run_activation_baseline.py` | 首次持续开启窗口的原始激活状态探针与对照 |
| 5 | `5_run_pure_sae.py` | 纯TopK SAE训练、冻结actor动作保持、新状态探针及结果窗口 |
| 6 | `6_compare_sae_results.py --runs <运行目录...>` | 跨运行/配置/种子汇总，重复运行核对与PCA动作保持对照 |
| 7 | `7_run_state_assisted_sae.py` | 分阶段辅助实验、复看汇总和检查点独立核验 |
| 8 | `8_evaluate_frozen_sae.py --study-dir <目录>` | 按冻结清单独立前向评价，可选离线动作干预；没有仿真 |
| 9 | `9_export_physical_bundle.py --out <新目录>` | 导出冻结数值模型并核验；P4使用`temp_scripts/physical_campaign.py`运行独立前缀匹配的局部替换试验 |
| 10 | `10_prepare_constrained_interventions.py --out <新目录>` | 开发集定义/审计受约束开启量方向及投影后匹配对照；P4的`feature_campaign.py`执行有界物理pilot，见[步骤10](STEP10_CONSTRAINED_FEATURES.md) |
| 13 | `13_run_combined_offline.py --run <运行目录>` | 本地全量 train/validation 固定输入组合测试，逐状态编辑、冻结探针及动作比较，见[组合实验](STEP11_COMBINED_PHYSICAL.md) |

编号用于可独立运行的实验入口；全部未编号公共模块集中在 `temp_scripts/`，通过 `temp_scripts.data_access` 等包路径导入。该目录是正式运行依赖，不是可删除的临时文件。采集脚本、源数据和历史结果快照保留原名。Visual Studio中按所需步骤设置启动文件；默认仍打开步骤1。

本次组合实验的本地运行入口为步骤13。准备冻结包和物理结果审计工具分别位于 `temp_scripts/prepare_combined_interventions.py` 和 `temp_scripts/analyze_combined_physical.py`。

步骤5默认`64→128→64 / TopK=8 / seed=0`，该默认值保留为历史起点，只用当前首次持续开启窗口的train/validation。原actor冻结，状态和动作只作后续评价，不进入纯SAE训练损失。详细配置和产物说明见[步骤5](STEP5_PURE_SAE.md)。在Visual Studio中选`5_run_pure_sae.py`作为启动文件；运行结束打开结果窗口，`--view-only`复看，所有结果保存在`outputs/pure_sae/`。正式开发集结果已经生成，无需为接手再训练。

2026-11-01已在用户授权下完成12配置×3种子的36次正式开发集训练，包含全局缩放对照、训练预算/学习率及K24/K32扩展。[参数探索与候选复看](outputs/sae_study_20261101/README.md)提供完整证据与命令。默认参数保留为历史起点，未自动替换为某个“最优”配置；三个独立测试划分尚未评价。

步骤4使用现有MachineLearning环境，默认取首次持续开启窗口内的**全部步骤**，只使用train/validation，运行完成后打开结果窗口。详见[完整运行与方法说明](STEP4_ACTIVATION_BASELINE.md)。

```bat
python 4_run_activation_baseline.py --prepare-only --no-gui
python 4_run_activation_baseline.py
python 4_run_activation_baseline.py --view-only
```

第一条只检查窗口，不拟合；第二条拟合baseline；第三条复看上次结果。结果统一放在`outputs/activation_baseline/`。步骤4实际拟合已完成，固定结果为`20261030_163827_787919`；接手优先复看，不为验证文档而重训。

已经生成 [首批图形核对与示例入口](outputs/examples/INITIAL_REVIEW.md)，包含实际成功/失败轨迹图和完整数值导出。

## 在 Visual Studio 运行

打开本目录的 `Potential_Explanation_On_Dexart.slnx`，使用现有 `MachineLearning` 环境。启动文件为 **1_review_trajectories.py**，按 **Ctrl+F5** 或 **F5** 即可打开原生窗口。无需启动服务器或安装网页框架。

默认数据路径根据项目位置自动定位到 `../../research/embodied_sae/runs/laptop_physical_index_v1`，不依赖 Visual Studio 的当前工作目录。如项目被移动，可在项目的脚本参数中填写 `--index-dir "索引目录的完整路径"`。

依赖为 NumPy、Matplotlib 和 Python 自带的 Tkinter；已安装的 pandas 等库可供后续实验使用，当前界面不强制依赖它们。

## 步骤3：独立运行冻结 Actor 前向复现

在 Visual Studio 中选择 **3_verify_actor_forward.py** 作为当前要运行的文件。默认读取本机 `D:\homework\dexart_assets.zip` 内的原策略检查点，恢复同一份权重，检查全部 280 条轨迹，再打开独立的结果窗口。原来的轨迹可视化入口不变。

窗口提供全局误差、重算/记录散点、逐时刻误差、指定维度对照、误差热图和动作裁剪对照，可切换轨迹、比较入口、神经元/动作维度及动作行。所有报告、图表、重算数组、模型副本、日志和代码快照均在 `outputs/actor_verification/`。

- [步骤3的运行和结果说明](STEP3_ACTOR_VERIFICATION.md)
- [最近完成的结果入口](outputs/actor_verification/README.md)

```bat
python 3_verify_actor_forward.py
python 3_verify_actor_forward.py --view-only
python 3_verify_actor_forward.py --no-gui
```

## 独立实验脚本与逐状态信息总表

后续每个实验可以使用独立入口脚本，共用数据读取模块；无需全部并入窗口主文件。当前窗口入口保持原状，在 Visual Studio 中选择所需脚本运行即可。

- **2_build_information_table.py**：直接运行，生成全部轨迹的逐状态总表、摘要表、SQLite数据库、字段字典和逐值检查；不启动窗口。
- **2_query_information_table.py**：直接运行，查询最近成功生成的总表，默认显示开发集前20行，并保存结果及查询条件。
- [当前总表入口](outputs/information_table/README.md)：每行是 `(trajectory_id, state_index)`，同时记录物理状态、两层各64维原始激活、22维原始动作与22维实际动作，以及事件/片段关联。终止状态保留，缺失激活与动作留空。

完整表保留全部坐标；摘要表额外列出非零神经元数量、激活范数、Top5编号及数值。Top5 只表示原始强度排序，不是语义或因果贡献排名；严格大于0才计为非零，真实全零向量与终止缺失值明确区分。

SQLite中的 `states` 是逐时刻总表，`trajectories` 保存轨迹身份和来源，`events` / `segments` 保存原始自动标签；`event_state_links` 区分事件状态、动作前、响应和未来确认时刻，`segment_memberships` 保留重叠关系。使用关联表直接连接会重复原始状态行；查询脚本采用 EXISTS 保持一状态一行。`development_states` 默认仅含train/validation且有策略响应的状态。

查询示例（在项目目录执行）：

```bat
python 2_query_information_table.py --trajectory-id object_9996__episode_00004 --state-index 31 --neuron h1_12
python 2_query_information_table.py --object-id 9996 --contact 1 --limit 50
python 2_query_information_table.py --event-type first_clear_opening --trigger-before --limit 20
python 2_query_information_table.py --trajectory-id object_9996__episode_00004 --state-index 64 --neuron h1_12
```

最后一个示例明确查看终止状态，神经元值返回 NULL。默认查询只取有策略响应的行；显式指定状态索引或事件时，会保留对应的真实终止状态。若需查看测试数据，显式使用 `--split test_trajectory`、其他测试划分名或 `--split all`；导出全量资料不代表允许用测试集筛选解释或调整阈值。当前状态、动作后状态与离线标签的字段角色见 field_dictionary.csv，后续 h-only 编码器仍只接收一层当前激活。

总表按版本保存在 `outputs/information_table/<时间戳>/`，查询结果保存在 `outputs/information_table/queries/<时间戳>/`。`latest.json` 只在整批生成和核对成功后更新；每版保存来源哈希与代码快照。所有结果和中间文件仍位于本项目根目录内。

## 窗口里可以做什么

1. **筛选轨迹**：按 split、物体、轨迹结局筛选。默认仅 train + validation；测试集需要主动切换，不据其修改阈值。
2. **查看物理状态**：同一时间轴显示手掌—把手距离、开启量、执行动作后的开启变化、七行接触状态、环境进度与成功标记。接触组、任一组接触与严格接触分开。
3. **查看完整激活/动作热图**：两层各 64 个神经元、22 维原始动作均值与 22 维实际裁剪动作均可见。两层热图共用色标，原始动作与实际动作分别标明数值范围。
4. **逐步浏览**：拖动底部滑块、点击时间曲线、输入状态索引或使用左右键；“播放状态”逐步浏览记录，不运行仿真。当前步向量页展示全部激活坐标和原始/裁剪动作的差异。
5. **检查空间关系**：三维显示世界坐标下的手掌、把手、四指尖和路径，可以用鼠标旋转。只画已记录的位置点，不伪造物体实体外观或 RGB 视频。
6. **跳转事件**：点击左侧事件定位到 `s_j`，按钮可跳到触发动作前的 `s_(j-1)`，或只显示事件片段。双击事件也会聚焦片段。紫色线为选中事件，浅紫色区间为其离线确认范围。
7. **比较轨迹**：“同物体成功/失败示例”从当前划分、物体中找示例；也可逐条设置 A、B。比较时支持按首次接触、首次开启等事件对齐。没有该事件时显示缺失，不制造对齐点。示例没有做相似状态匹配，不能直接当作成功/失败原因证据。
8. **保存审阅意见**：选择事件，填写判断和备注，点击保存。意见与自动标签独立，保存动作不会修改 v1 阈值或原数据。终止处不足 3 步的连续段明确标记为未确认，未发生事件不提供虚构片段。
9. **导出图表与数值**：“导出当前轨迹/片段”保存四类 PNG、两层完整激活/动作的 CSV、状态表、事件表、NPZ 和来源说明。“导出对照图”保存对齐图与比较元数据。底部 Matplotlib 工具栏也可缩放、平移；其保存按钮同样转入程序的导出流程，统一写入项目 outputs。

左侧整体支持滚动；图表下方工具栏可缩放。标签页按需绘制，切换轨迹不预先读取全部点云或生成大量图片。

## 所有结果都在项目根目录内

“根目录”按本 Visual Studio 项目目录解释。原始数据在项目之外只读，程序新产生的结果和中间文件均收纳在：

```text
Potential_Explanation_On_Dexart/
├─ 1_review_trajectories.py   # Visual Studio 启动入口
├─ 2_build_information_table.py / 2_query_information_table.py
├─ 3_verify_actor_forward.py
├─ 4_run_activation_baseline.py
├─ temp_scripts/                       # 数据、绘图、模型、导出与窗口实现
│  ├─ __init__.py
│  ├─ project_paths.py                  # 项目根目录仍为上一级
│  ├─ data_access.py
│  └─ ...
├─ tests/test_workflow.py              # 数据与边界验证
└─ outputs/
   ├─ latest_run.txt                   # 最近一次运行目录
   ├─ review_notes.json                # 跨运行累积的人工审阅历史
   ├─ _cache/                          # Matplotlib 等缓存
   ├─ _test/                           # 测试中间文件
   └─ runs/<北京时间时间戳>_<模式>/
      ├─ run_manifest.json
      ├─ application.log
      ├─ validation/data_validation.json
      ├─ trajectories/<轨迹ID>/<范围和当前步>/
      │  ├─ 01_physical_states.png
      │  ├─ 02_activations_actions.png
      │  ├─ 03_selected_state_vectors.png
      │  ├─ 04_world_positions.png
      │  ├─ steps.csv / states.csv / events.csv
      │  ├─ aligned_numeric.npz
      │  └─ source_and_alignment.json
      ├─ comparisons/<A与B>/
      └─ review/
```

程序不会在源数据目录写文件；Python 字节码、Matplotlib 字体缓存也被限制在项目 `outputs/` 内或禁用。每次运行新建目录，当前运行中重复导出相同轨迹、范围和当前步会更新对应导出。人工意见按历史追加保存。

## 数值与下标

```text
s_t → h1_t → h2_t → μ_t → clip(μ_t)=a_t → s_(t+1)
```

长度 T 的轨迹有 T 个动作和两层激活，另有 T+1 个状态。事件状态为 `s_j`，触发动作前激活/动作行是 `j-1`，变化后的策略响应行是 `j`。终止 `s_T` 没有新激活或动作，窗口会明确留空，不用零向量代替。

`DataRepository.load(trajectory_id)` 返回 `Trajectory`：`arrays` 保存完整原始数值，`state(key)` 返回 T+1 状态链，`relation(j)` 给出当前状态和策略响应；当 j=T 时 `policy=None`。读取时校验形状、有限激活/动作、状态时间链、动作裁剪和事件/片段下标。

事件属于固定 v1 物理标注初稿。人工审阅后如需修订阈值，应在 train/validation 上另建版本；当前程序只记录意见，不自动冻结新规则。未来 h-only 编码器只输入所选一层当前激活，不能把结局、post_*、事件确认或片段归属混作输入。

## 命令行复现与检查

在本目录的 Anaconda Prompt 中：

```bat
conda activate MachineLearning
python 1_review_trajectories.py
python 1_review_trajectories.py --validate-all --export-demo
python -B -m unittest discover -s tests -v
```

第二条处理命令检查完整数据，并导出开发集同物体成功/失败示例，不开窗口。图表、检查报告与导出数据仍全部写入项目 outputs。第三条运行边界测试；测试中间文件同样在 outputs 内。
