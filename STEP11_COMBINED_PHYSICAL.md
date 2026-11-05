# 组合物理实验与步骤13本地离线测试

## 本批完成状态

2026-11-03约23时40分，正式8起点的7216条完整轨迹全部完成，调度失败0。用户步骤13已完成1365状态、409500变体。完整物理回传位于 `outputs/combined_physical/20261103_140606_955714/remote/`，校验状态见该运行目录的 `retrieval_verification.json`，关机状态见 `server_shutdown.json`。后续本地数值审计按下方辅助工具命令执行。

2026-11-03，亚洲上海时间。依据组合实验交接文稿实施，用户选择同时比较单步、5步和从首次开启触发持续至结束。服务器运行由助手负责，本地离线脚本交给用户运行。

## 实验参数

- 冻结纯 SAE `dense_l0p0` 和辅助 SAE `slots_s0.3_a1.0`，各使用模型种子 0、1、2。
- 各模型独立使用既有 train 开启量绝对相关性排名的前6坐标。默认比较 Top1、Top2、Top3、Top4、Top6。这里的组合大小和编码器 TopK 分开记录。
- 开始时刻采用自然分支首次动作前开启量达到 0.05 rad。该条件对两类模型对称。所有条件使用同一触发时刻，保留原策略失败、无触发和候选未激活。
- 每个条件从独立 Python/PhysX 进程初始化，先通过完整原策略重复门限，再核对本条件触发前的状态、完整含噪观测、h1与动作。运行至环境终止。
- 处理持续1步、5步、或从触发点持续至终止。每一步使用该分支当前 h1 重新计算编辑，保存全部目标码和其他码。
- 名义全抑制保留重建残差后投影到非负域，实际扰动上限为当前 h1 范数的10%。投影和限幅之后重新编码。
- 第二套剂量在同一当前 h 上，对六模型和所有预定组合的正可行预算取最小值，沿投影射线求解。未激活候选单独保留为空操作。闭环分支后续状态不同，因此后续预算和累计暴露需要分别审计。
- 默认对照包含3个固定随机方向和共享 Ridge 前N有向坐标和。随机组合可通过 `--random-groups` 追加，会从该模型语义前6之外抽取同大小组合并匹配预算。
- 默认4个开发物体11395、11405、11406、11477，各使用两个新环境种子2026110300、2026110301，共8起点。最大7216条完整轨迹。采用该规模作为首轮扩展的工程预算。
- `--group-mode factorial` 补齐 B、C、AC、BC，A、AB、ABC 由已有 Top2、Top4、Top6 实现。联合投影与限幅可能产生输入非可加性，因此因子条件的差异需要进一步核查，当前脚本输出描述性结果。

## 本地入口

在本项目目录的 MachineLearning 环境运行步骤13。冻结包已经由助手准备完成。该入口使用相同冻结协议，对全部1365个原train/validation状态进行409500个固定输入组合变体检查。运行与物理服务器分开，结果放在该冻结运行目录下的 `offline/<时间戳>/`。该脚本保存逐状态结果与轨迹等权汇总，完整记录未激活与匹配失败。`--max-states 2` 可先做少量状态检查。

```bat
python 13_run_combined_offline.py --run outputs/combined_physical/20261103_140606_955714
```

在 Visual Studio 中将 `13_run_combined_offline.py` 设为启动文件即可运行本次本地离线测试。

## 辅助工具

准备和审计脚本统一收纳在 `temp_scripts/`。维护实验协议时，可在项目目录以模块方式调用准备工具，每次生成新目录。

```bat
python -m temp_scripts.prepare_combined_interventions
```

可选扩展为32起点、补充因子分组与随机组合。

```bat
python -m temp_scripts.prepare_combined_interventions --origins-per-object 8 --group-mode factorial --random-groups
```

生成物位于 `outputs/combined_physical/<北京时间时间戳>/`，包括六模型、训练排名、物理协议、开发参考、代码快照、SHA256清单和上传包。`latest.json` 只在完整预检成功后更新。

服务器证据下载完成后，可调用审计工具。显式指定运行目录能够避免后续准备工具改变默认指向。

```bat
python -m temp_scripts.analyze_combined_physical --run outputs/combined_physical/<运行目录>
```

该工具读取 `remote/campaign/`，审计全部成功分支的原始NPZ、完整策略哈希、前缀、非负约束、逐步编辑、64维码和22维动作。失败重试保留全部尝试，统计使用各条件最新一次记录。物理证据尚未下载时会显示所需路径。

## 服务器入口

冻结包中的命令用于助手在服务器上执行。

```bash
/hy-tmp/embodied_sae/.venv/bin/python -B code/combined_campaign.py --root "$RUN" --verify-only
VK_ICD_FILENAMES=/hy-tmp/embodied_sae/userspace/nvidia_icd.json /hy-tmp/embodied_sae/.venv/bin/python -B -u code/combined_campaign.py --root "$RUN" --resume
```

协议与跨端参考须通过后才能进入物理运行。调度器对每个起点使用独立子进程，并行两个起点。`campaign/campaign.json` 原子记录进度，`campaign.lock` 阻止重复运行。重启使用 `--resume`，成功轨迹与报告哈希一致时跳过，失败或中断分支另建重试目录。可用空间低于4GiB时停止调度并保留证据。

用户随后要求根据资源占用增加并行支线。`temp_scripts/run_combined_parallel.py` 通过运行时执行器设置并发数，并复用冻结包中的完整调度与核验流程。运行并发及启动脚本哈希单独记录在运行目录的 `runtime/execution_<时间戳>.json`。切换调度器时，先让已启动的仿真子进程完成并核对其报告，再恢复统一的进度清单。

## 离线结果口径

`physical_results.csv` 保存逐条件完整任务结局、触发后1/5/20/50步配对开启差、距离/接触变化和实际累计扰动。任一分支提前终止时，固定时刻指标留空。

`paired_dictionary_effects.csv` 对同起点同模型条件，比较目标与全部3个随机方向的5步开启差。严格对照要求各分支全部计划生效步骤有效。`paired_origin_effects.csv` 只在三个模型种子均有效时汇总到物理起点。覆盖统计与失败分母保留在 `origin_coverage.csv`。`independent_audit.json` 记录审计状态与最大数值误差。

默认8起点属于开发扩展。组合排名和超参数在读取本批结果之前冻结，后续独立验证另用新起点。现有程序提供可审计的干预和配对结果，科学解释仍须结合剂量、覆盖和跨目标变化。


本批回传于2026-11-04凌晨完成，21816个清单文件通过大小与SHA256核验。Windows长文件路径由回传工具和物理审计入口统一兼容。
