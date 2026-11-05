# 步骤14：可解释性主线的本地后续分析

冻结SAE与原策略，分析开发数据的特征数量、补集/残差、双向编辑及已有物理成败翻转；不启动服务器或新物理实验。

最新完成目录：`outputs/local_interpretability/20261104_mainline_v3`。

主结果见该目录的`LOCAL_RESULTS_2026-11-04.md`，独立检查见`independent_verification.json`。v1/v2是中间开发版本；v3统一了探针float64预处理，并包含完整补充诊断。

**接手时优先读取已有结果，不自动重跑。** 新窗口接续设计见[主交接第6节](../../项目交接文档_具身策略决策解释_2026-11-04.md)。

只复核已有结果，不重新拟合：

```powershell
python -B -m temp_scripts.validate_local_interpretability --run outputs/local_interpretability/20261104_mainline_v3
```

以下完整重做命令仅在确有需要时执行，目录名替换为尚不存在的新名称：

```powershell
python -B 14_run_local_interpretability.py --out outputs/local_interpretability/new_run_name
```

输出目录要求不存在，避免覆盖旧结果。新探针使用train/validation，不能称独立测试；原始模型不重新训练。入口会完成严格接触补充分析和独立数值检查。

最重要的结果修正：物理任务成功要求目标进度、近距离与严格接触同一步满足。全部119条失败轨迹都曾达到目标进度且近距离，但当时无严格接触。不能将“干预后失败”直接解释为打不开盖子。

新增163800个固定输入变体及8190组A/B/AB检查均不产生新的物理轨迹；失败救援仍待闭环验证。
