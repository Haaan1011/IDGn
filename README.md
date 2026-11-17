# IDGn 复现说明

本项目从两份原始医院导出表开始，执行数据处理、模型训练和数值评估。代码不生成论文图。原始数据、患者级结果、划分文件和模型权重不包含在代码包中。

## 1. 准备环境和数据

- 建议使用 Linux、Python 3.12 和可用的 NVIDIA CUDA 环境。架构比较包含 16 种架构、5 个随机种子和 3 个药物任务，训练时间较长。
- 准备临床资料表和监测记录表两份原始 Excel；字段映射见 `pipeline/configs/schemas/`。文件应为原研究对应的原始数据，否则不能保证得到相同队列或论文指标。
- 在有权访问医院数据的环境中解压代码，进入本目录，并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖文件固定了 Python 包版本；GPU 驱动和 CUDA 环境需与 PyTorch 兼容。该依赖组合尚未在全新环境中完成安装验证。

## 2. 开始复现

将以下示例路径替换为实际路径。运行目录必须是**不存在**的新目录，并且对当前用户可写：

```bash
python reproduce.py \
  --clinical /授权目录/临床资料.xlsx \
  --monitoring /授权目录/监测记录.xlsx \
  --run /授权目录/IDGn运行结果
```

包含空格的路径请用引号包围。程序首先整理数据，然后生成划分、训练模型、计算指标与补充分析。默认按原分层方法和种子生成训练、验证、测试集（2849、611、611 个周期）；原数据验证中，生成的划分与原划分一致。也可指定经授权的原始划分文件：在命令末尾增加 `--split /授权目录/split.csv`，文件需包含 `cycle_uid` 和 `split` 列。

## 3. 进度、中断与结果

终端显示 `Running gn27` 等阶段名称；各阶段详细日志写入运行目录的 `logs/`，预处理日志为 `preprocessing.log`。失败时查看终端提示的日志。已经完成准备阶段的运行中断后，可使用相同输入和运行目录，并在原命令末尾增加 `--resume`；完成的阶段会跳过。如果预处理阶段尚未成功完成，需另选新的运行目录重新开始。

运行结束后用 Excel 等软件打开运行目录中 `outputs/` 下的 CSV：

| 内容 | 相对于运行目录的路径 |
| --- | --- |
| Gn 测试集 Accuracy、Weighted-F1、ROC-AUC、AP 和各次监测结果 | `outputs/gn_metrics/` |
| 测试集混淆矩阵数值 | `outputs/gn_metrics/source_data/gn_confusion_tables.csv` |
| 本次结果与论文数值对照 | `outputs/manuscript_metric_comparison.csv` |
| 受试者特征 | `outputs/table1/` |
| 架构比较 | `outputs/architecture/tables/` |
| 全变量与精简模型比较 | `outputs/table3_paired_cycle_bootstrap.csv` |
| 获卵数和 OHSS 结局 | `outputs/outcome27_model_family_comparison_20261016/tables/` |
| OHSS 校准 | `outputs/ohss_calibration/` |
| SHAP 特征贡献 | `outputs/full_shap62/` 和 `outputs/supplementary_shap27/` |
| 输入缺失敏感性分析 | `outputs/monitor_level_shared_multitask_gru_addgate_27features_input_availability_sensitivity_20261016/tables/` |

训练完成后，可在运行目录运行 `python -m streamlit run scripts/scenario_demo.py --server.address 127.0.0.1` 启动研究演示，输入模拟数据查看预测表格。演示仅用于研究，不是经批准的临床系统。

## 4. 解读结果时需要注意

这是依据论文和现存实验代码整理的重跑流程，并非所有历史实验都能逐项还原。27 个变量按论文表格固定；全变量的 62 项是在既往 55 项配置基础上按论文补建。Gn 分类的 FSH 剂量上限、优化器设置与部分旧实验不同；候选方案结局模型及阈值规则也有按照现有资料作出的实现选择。完整说明见 `README.md` 的 Reproduction decisions。测试集 OHSS AUC 在现有环境的重跑结果为 0.742，与论文报告的 0.731 不一致，请核对对照表，勿将两者视作完全一致。

---

# IDGn manuscript reproduction

This package implements the 11.12 manuscript analysis and the companion Supplementary Methods. It starts from the two original hospital exports and trains the models from scratch. Outputs are numerical results, tables, and run logs. The package contains source code and configuration only. The workbooks, cycle-level outputs, split manifest, and model checkpoints stay in the authorized computing environment.

## Run

Use Python 3.12. An NVIDIA CUDA environment is recommended for the repeated GRU training and SHAP stages. The architecture grid trains 16 variants × 5 seeds × 3 drugs.

```shell
python -m pip install -r requirements.txt
python reproduce.py --clinical "/authorized/clinical.xlsx" --monitoring "/authorized/monitoring.xlsx" --run "/authorized/idgn-run"
```

The clinical export supplies baseline characteristics, cycle eligibility and outcomes. The monitoring export supplies visits, medication and follicle measurements. The original worksheet columns and field mappings are in `pipeline/configs/schemas/`. The run directory must be new and writable.

With the original workbooks, the original split procedure (seed 20260424, stratified by recorded live birth and OHSS status) recovers the frozen 2,849/611/611 assignments. In the hospital verification run, all 4,071 regenerated assignments agreed with the archived split manifest. To use an authorized copy of that manifest explicitly, pass `--split /authorized/split.csv`. Add `--resume` with the same inputs to continue a stopped run. Each stage records its own log.

## Outputs

| Manuscript item | Under the run's `outputs/` directory |
| --- | --- |
| Table 1 cohort characteristics | `table1/` |
| Table 2 and Supplementary Tables S1A–S1B | `architecture/tables/` with all five seeds and mean/SD by task |
| Table 3 full-feature/parsimonious comparison | `table3_paired_cycle_bootstrap.csv` and the 1,000 paired replicates per task |
| Global SHAP rankings | `full_shap62/` (62-feature neural components) and `supplementary_shap27/` (27-feature neural classifiers) |
| Supplementary Table S2 | `pipeline/configs/features/parsimonious_feature_dictionary.tsv` in the source package |
| Gn test metrics and Supplementary Table S4 | `gn_metrics/` with Accuracy, Weighted-F1, macro one-vs-rest ROC-AUC, macro AP and monitor-level accuracy through Monitor 9 |
| Table 4 input-availability sensitivity | `monitor_level_shared_multitask_gru_addgate_27features_input_availability_sensitivity_20261016/tables/` |
| Gn confusion counts and proportions | `gn_metrics/source_data/gn_confusion_tables.csv` |
| Clinical outcome performance | `outcome27_model_family_comparison_20261016/tables/` |
| OHSS calibration | `ohss_calibration/` with slope, intercept, Brier score and five-group calibration statistics |
| Candidate regimens and research demonstration | `scenarios/`, with validation-fitted temperature/threshold settings and a synthetic scenario |
| Cross-check against manuscript point estimates | `manuscript_metric_comparison.csv` |

Open the tabular research demonstration locally after training:

```shell
cd /authorized/idgn-run
python -m streamlit run scripts/scenario_demo.py --server.address 127.0.0.1
```

The demonstration runs actual model inference from editable synthetic baseline and monitoring rows. It shows the retained dose combinations, predicted outcomes, dose-stratified scenario estimates, and individual neural-logit SHAP as tables. It is a research interface, and its candidate comparisons are predictions under entered scenarios.

## Reproduction decisions

- Dose labels are the recorded post-visit decisions and inputs stop at the current visit. Monitor order and stimulation day are separate. Continuous scaling and median imputation use training data. Hormone/endometrial measurements forward-fill within cycles. Models train under the fixed split. Gn metrics use uncalibrated probabilities with 1,000 treatment-cycle bootstrap resamples for reported intervals.
- The original 55-variable Table 3 archive and the 11.12 manuscript's 62-variable current-monitor specification differ. The code restores the seven eligible candidate fields to the 55-variable configuration. Three pre-decision drug doses replace three legacy fields from the daily-expanded dataset. This manifest is written at run time.
- The upper dose boundary in the manuscript is 300 IU/day. In the rerun, 4,289 FSH records and 4,291 LH and HMG records were evaluable on the same 611 test cycles. The manuscript's shared count of 4,291 records and the older training artifact used a different FSH upper-bound convention.
- The final Gn retraining uses the written Supplementary S2 AdamW settings, the archived two neural architectures, the task-specific tree experts and the published mixture weights. The archive used different optimizer settings for the two neural members. Recalculated values are compared directly against the manuscript rather than adjusted to match.
- The 27 variables are fixed from Supplementary Table S2. The package recalculates global SHAP rankings for 62 and 27 inputs. Clinical review is part of the published variable selection, and the text provides no reproducible rule that assigns all 27 features to a deterministic SHAP cutoff.
- The manuscript does not give a numerical threshold-fitting criterion. The demonstration uses per-drug validation temperature scaling with multiclass log loss and a pooled one-vs-rest F1 threshold, keeping the highest-probability class as a fallback. Candidate outcome models are trained on frozen GRU representations plus the three candidate doses. The final-visit OHSS performance retains the exploratory comparison described in Supplementary Methods S2. The web demonstration and final-visit outcome evaluation are distinct analyses.

Results in this package reflect the published analysis specification as implemented here. Several published point estimates differ from the source-data retraining, including OHSS ROC-AUC. Use the generated comparison table for any manuscript update.
