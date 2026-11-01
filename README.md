# DexArt 轨迹读取与可视化审阅

本项目包含轨迹审阅、总表、冻结 Actor 前向复现，以及可由用户运行的原始激活 baseline。原策略保持冻结。

## 按流程选择入口

| 步骤 | 运行入口 | 工作内容 |
| --- | --- | --- |
| 1 | `1_review_trajectories.py` | 读取、校验、物理事件和连续轨迹窗口审阅 |
| 2 | `2_build_information_table.py` / `2_query_information_table.py` | 构建及查询逐状态总表 |
| 3 | `3_verify_actor_forward.py` | 冻结actor后半网络的前向复现 |
| 4 | `4_run_activation_baseline.py` | 首次持续开启窗口的原始激活状态探针与对照 |
| 5 | `5_run_pure_sae.py` | 纯TopK SAE训练、冻结actor动作保持、新状态探针及结果窗口 |

编号用于可独立运行的实验入口；全部未编号公共模块集中在 `temp_scripts/`，通过 `temp_scripts.data_access` 等包路径导入。该目录是正式运行依赖，不是可删除的临时文件。采集脚本、源数据和历史结果快照保留原名。Visual Studio中按所需步骤设置启动文件；默认仍打开步骤1。

步骤5供用户亲自试运行，默认`64→128→64 / TopK=8 / seed=0`，只用当前首次持续开启窗口的train/validation。原actor冻结，状态和动作只作后续评价，不进入纯SAE训练损失。详细配置和产物说明见[步骤5](STEP5_PURE_SAE.md)。在Visual Studio中选`5_run_pure_sae.py`作为启动文件；运行结束打开结果窗口，`--view-only`复看，所有结果保存在`outputs/pure_sae/`。代码和合成验证不代表正式SAE实验已完成。

步骤4使用现有MachineLearning环境，默认取首次持续开启窗口内的**全部步骤**，只使用train/validation，运行完成后打开结果窗口。详见[完整运行与方法说明](STEP4_ACTIVATION_BASELINE.md)。

```bat
python 4_run_activation_baseline.py --prepare-only --no-gui
python 4_run_activation_baseline.py
python 4_run_activation_baseline.py --view-only
```

第一条只检查窗口，不拟合；第二条拟合baseline；第三条复看上次结果。结果统一放在`outputs/activation_baseline/`。代码交付和自动化检查不代表正式探针实验已完成。

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
