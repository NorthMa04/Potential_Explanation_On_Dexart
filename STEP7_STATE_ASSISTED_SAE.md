# 步骤7：状态辅助SAE的自适应离线验证

2026-11-02，北京时间。用户已明确授权自主实现、运行、多方案探索并按结果调整；动作改变后物体实际运动留待后续仿真。本步骤不更新原策略，不启动物理仿真。

实际已完成75配置×3种子=225次训练、冻结21模型的三类独立评价、8个监督旋转参照和58,110个离线干预/匹配变体；最终89项回归通过。[完整结果](outputs/state_assisted/20261102_122123_738477/FINAL_REPORT_2026-11-02.md)与[冻结计划](outputs/state_assisted/20261102_122123_738477/frozen_evaluation_plan.json)为当前入口，不需要重新训练。

主配置为`slots_s0.3_a1.0`：64/K16/global，每目标固定4槽，state_weight=.3、action_weight=1，500/50预算。其额外损失/结构是在初始配对之后经开发集证据扩展，保留无动作、无状态、稠密头等配对对照。相对强监督Ridge旋转的额外独立测试证据主要在开启量；特征没有完全单语义解缠。

干预域审计未通过：所有SAE方向干预变体至少产生一个负h1坐标，抑制/增强的分组负坐标均值约8%–25%。数值范数匹配正确不代表原ReLU域合法；当前只能作固定输入读出诊断。先在开发集设计并冻结合法域约束，再进入物理仿真，不用当前测试反复调新干预规则。

## 问题与证据

检验在相近稀疏性与原策略动作保持程度下，当前物理状态的辅助监督是否让信息集中到更少的固定特征。当前状态可预测、少量特征可读、当前动作贡献、闭环物理效果分开论证。监督引导结果不称为无监督语义发现。

主数据为首次已确认sustained_opening，以触发动作j-1为中心前后5步，保留所有可用动作前帧。train94条/1029步、validation31条/336步。保持原整轨迹划分及轨迹等权。严格接触在此窗口无正例，分类目标为任一指组/手掌组接触，不代表抓稳。

## 方法与事先定义

编码器仅输入h1；z重建h1并通过线性头预测当前手掌距离、相对固定初始角开启量、任一接触。回归标签均值/标准差只用train轨迹权重拟合。默认状态损失为所选目标损失的均值（两个标准化MSE和一个BCE），总损失rec+λ_state*state。TopK不另加L1。字典列单位范数，原actor冻结。

首轮64/K16/global，Adam lr=.001，batch128，500轮/patience50，seed0/1/2。λ=0/.03/.1/.3/1。默认按validation轨迹等权重建MSE选检查点，含第0轮。辅助头独立随机流；SAE初始化/批次序列按seed配对。λ0必须与历史纯SAE逐位复现。

后续按开发集证据扩展：Huber(delta=1)、train类别权重BCE、train熵归一化BCE、单目标辅助、每目标4连接硬掩码头、每目标预分配4坐标头、K8/K32、逐维缩放、动作保持辅助以及联合损失选模。每批运行前保存完整配置计划和源码快照。不会把不同损失数值、选模规则或K当作完全相同约束。

top4头每行按当前权重绝对值选择4个连接；支持受初始权重及硬掩码优化影响，不代表所有稀疏头。slots头固定目标0/1/2分别使用z[0:4]/z[4:8]/z[8:12]；其他坐标参与重建。可选动作损失为原尺度重建送入冻结actor后未裁剪均值MSE，除以train动作总体方差；actor只传输入梯度，权重始终冻结。

默认rec选模与可选joint选模是分别定义的实验，早停遵循各自预定规则。每次同时保存rec-best/joint-best，不事后挑另一检查点冒充既定主成绩。实际训练轮数/更新数均报告。

## 评价和适应规则

所有表示冻结后重新拟合同容量线性/逻辑回归探针，正则候选与原实验相同。Top1/2/4按train绝对加权相关选固定坐标，每目标可不同；全z、时间、时间+z、原神经元、PCA对照一并保留。稠密辅助头得分不替代新探针，TopK不等于语义只占K个固定坐标。

同时比较原尺度激活NMSE、未裁剪/裁剪动作误差、逐维/轨迹分布、实际L0、失活覆盖，以及距离/开启量MAE和接触AP/log loss。三种子的样本SD仅代表训练波动。监督线性旋转作为更强有监督非SAE对照：仅train求激活-状态加权互协方差的SVD方向，用正交补PCA补齐，可逆稠密64维，明确不同于TopK容量。

开发集用于全部自适应决策；初期尝试不承诺辅助必胜。优先保留语义集中与动作保持的Pareto候选。若监督收益主要在接触或仅在全z，完整报告，不改成全面语义成功。若动作损失帮助保持动作，则补无状态但相同动作损失的对照。

跨种子匹配只使用train，综合解码方向和train激活行为，不把相同编号当同一语义。重载检查点独立核验z/重建/actor动作与保存数组。

只有开发集选型结束、候选ID/评价规则写入frozen_evaluation_plan.json后，才一次性读取三个独立测试划分。测试不再选配置、正则、特征或阈值；按轨迹配对bootstrap与种子SD分开报告。无事件轨迹保留覆盖记录，不造窗口。此为事件条件解释实验，不能扩大为所有操作或物体总体结论。

## 运行

在MachineLearning环境/本项目目录执行：

```powershell
python -B 7_run_state_assisted_sae.py --stage core
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage losses
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage structure
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage fidelity
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage refine
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage refine_slots
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage concentration
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --stage negative
python -B 7_run_state_assisted_sae.py --study-dir outputs/state_assisted/<运行ID> --analyze-only --verify
```

默认离线CPU/1线程。可用--only限制该阶段的预定配置。运行可跳过已完成且配置完全相同的实验；不覆盖不完整/失败目录，不改写历史纯SAE。所有结果/模型/日志/快照位于本项目outputs内，公共模块为正式依赖。

独立评价入口为`8_evaluate_frozen_sae.py --study-dir <目录>`，先验证原冻结清单/哈希，再前向预测。`--offline-interventions`仅增加固定输入actor动作干预，没有物理仿真。本轮已有完整独立结果，位置由研究目录`evaluation_location.json`/`intervention_location.json`记录；没有根据测试结果重选任何方法。
