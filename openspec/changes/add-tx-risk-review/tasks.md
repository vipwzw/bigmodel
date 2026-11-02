# Tasks

## 1. 骨架、依赖与 CLI 契约

- [x] 1.1 新建 `src/bigmodel_lab/risk/` 包（`__init__.py`、`fixtures/`）与 `docs/risk-review.md` 骨架，验证 `.venv/bin/python -c "import bigmodel_lab.risk"` 成功且目录结构符合 design.md D9
- [x] 1.2 在 `pyproject.toml` 新增可选 dependency-group `risk`（`eth-abi`、`eth-utils`）并更新锁文件，验证 `uv sync --group risk` 成功、`.venv/bin/python -c "import eth_abi"` 成功，且默认 `uv sync` 不安装该组时核心测试仍可运行
- [x] 1.3 实现 `risk/config.py` 的 pydantic 配置模型（dataset/rules/model/train/eval/review），校验 `configs/risk.yaml` 与 `configs/risk-rules.yaml`，验证合法配置可加载、非法字段给出字段级错误（`pytest tests/test_risk_config.py`）
- [x] 1.4 在 `cli.py` 注册 `lab risk data|review|train|eval` 子命令骨架与参数（`--request/--batch/--mode/--out/--json` 等）及退出码约定，验证 `lab risk --help` 输出四组子命令、`lab stages` 仍输出 8 项（`pytest tests/test_risk_cli.py`）
- [x] 1.5 更新 `README.md` 快速开始与 `docs/risk-review.md` 的"运行入口、退出码、产物目录"章节，验证文档中的命令与 `lab risk --help` 输出一致且可执行

## 2. 数据模型与场景生成

- [x] 2.1 实现 `risk/schema.py`：`RiskSample`、`ReviewRequest`、`WorldState`、`Label`、`ReviewReport` 模型与 `schema_version`，验证合法样本通过、缺字段/类型错误/未知版本被拒绝并给出字段级错误（`pytest tests/test_risk_schema.py`）
- [x] 2.2 实现确定性场景生成器，覆盖普通转账、批量提现、approve/permit、EIP-7702 委托、权限变更、代理升级、批量 nonce、跨链组合与正常运营基线，验证相同种子生成结果逐字节一致、场景覆盖检查能发现缺失类别（`pytest tests/test_risk_dataset_generator.py`）
- [x] 2.3 新增币安 7702 回归用例 fixture 及变体（地址、金额、nonce 数量、目标合约不同），标注 `expected_decision: BLOCK`，验证 fixture 通过 schema 校验且变体与原始样本不共享同一实例（`pytest tests/test_risk_binance_case.py`）
- [x] 2.4 实现数据卡、按种子/场景族的 train/val/test 划分与泄漏检查，验证相同种子划分一致、跨集实例为零、数据卡包含来源许可/场景构成/标签分布/已知限制（`pytest tests/test_risk_dataset_split.py`）
- [x] 2.5 实现 `lab risk data generate|validate|show`，产物写入 `runs/risk/<run-id>/`（沿用 `RunContext`/manifest）并包含统计报告与数据卡，验证生成退出码 0、非法样本校验非零退出（`pytest tests/test_risk_data_cli.py`）
- [x] 2.6 在 `docs/risk-review.md` 补充数据章节（样本 schema、八类场景、来源与许可、局限、如何新增场景），验证文档中的生成/校验命令可执行且输出与测试统计一致

## 3. 解码、模拟与特征

- [x] 3.1 实现 `risk/decode.py`：常见 selector 内置解码、基于 `eth-abi` 的 calldata 解码、EIP-712 结构化解析、EIP-7702 授权元组解码与结构化输入降级，验证已知操作、未知 calldata、依赖缺失三类用例（`pytest tests/test_risk_decode.py`）
- [x] 3.2 实现 `risk/simulate.py`：基于 world snapshot 的转账、批量提现、approve/permit、7702 委托、owner 变更、代理升级、批量 nonce、跨链组合状态转移与资产/授权/权限/代码差分，验证每类操作的差分结果（`pytest tests/test_risk_simulate.py`）
- [x] 3.3 实现 `risk/features.py`：capability delta、最坏情况资产暴露比例、滚动窗口累计暴露、跨链汇总，验证 7702 用例暴露比例为 1.0、普通转账按金额计算、窗口累计正确（`pytest tests/test_risk_features.py`）
- [x] 3.4 在 `docs/risk-review.md` 补充"解码 → 模拟 → 能力/暴露"章节与教学模拟器 vs 生产 fork 模拟器的差异说明，验证文档示例请求可复现测试中的暴露数值

## 4. 规则、决策与审查报告

- [x] 4.1 实现 `configs/risk-rules.yaml`（规则版本、不可逆操作、资产比例/窗口阈值、白名单、批量 nonce、新对手方）与 `risk/rules.py` 规则引擎，验证规则命中携带 ID/严重度/证据/版本、重复 ID 或非法配置拒绝启动（`pytest tests/test_risk_rules.py`）
- [x] 4.2 实现 `risk/decision.py` 四档决策融合与"不可降级"约束（含模型意见占位），验证 BLOCK 不会被降级为 ALLOW/STEP_UP、STEP_UP/ESCALATE 与退出码映射正确（`pytest tests/test_risk_decision.py`）
- [x] 4.3 实现 `risk/pipeline.py` 与 `lab risk review` 单笔/JSONL 批量审查，输出稳定 JSON 报告（请求摘要、解码、模拟差分、能力变化、暴露、规则命中、模型意见、最终决策、证据、版本、耗时），验证单笔/批量报告与退出码（`pytest tests/test_risk_review_cli.py`）
- [x] 4.4 实现安全与失败路径：外部文本注入标记、解码/模拟/规则加载失败时 fail-closed，验证注入样本结论不受影响、失败样本输出 ESCALATE 或 BLOCK 且退出码非零（`pytest tests/test_risk_safety.py`）
- [x] 4.5 在 `docs/risk-review.md` 补充规则/决策/报告字段章节与币安用例逐层命中表，验证文档中的 `lab risk review` 命令可执行且报告字段与实现一致

## 5. 审查训练数据构建

- [x] 5.1 实现 SFT 数据转换：system rubric + schema + 安全约束，user 为规范化请求/上下文/模拟结果 JSON，assistant 为 `ReviewReport` JSON，验证输出通过 schema 校验、user 内容不含 label 字段（`pytest tests/test_risk_model_data.py`）
- [x] 5.2 实现确定性 DPO 偏好对生成（完整有证据报告为 chosen，删除能力变化/降低等级/建议 ALLOW 的劣化报告为 rejected），验证 rejected 至少缺少一类关键证据、prompt 不含标签（`pytest tests/test_risk_dpo_data.py`）
- [x] 5.3 实现训练样本预算（sanity/standard 上限、按场景族分层抽样）与统计报告，验证样本数符合预算配置、类别分布可复现（`pytest tests/test_risk_train_data_budget.py`）
- [x] 5.4 在 `docs/risk-review.md` 补充训练数据格式、证据引用约定与标签泄漏防护说明，验证文档中的样例 JSON 可被转换脚本解析

## 6. 模型训练、推理与集成

- [x] 6.1 实现 `lab risk train`：复用 `SFTConfig`/`SFTTrainer` 完成 sanity/standard 档训练，记录配置、manifest、指标与检查点，验证 sanity 档在预算内完成并产出摘要（`pytest tests/test_risk_train.py` + 实跑一次）
- [x] 6.2 实现 `risk/infer.py`：复用 `build_generator`/`ChatSession` 加载合并模型或基座，提取并校验 JSON 报告，处理超时/异常/非法输出，验证合法输出、非法输出、超时回退三类用例（`pytest tests/test_risk_infer.py`）
- [x] 6.3 将模型意见接入 `review --mode rules_plus_model`：模型只能维持或提升决策，失败自动回退 `rules_only` 并在报告标记降级，验证 BLOCK 不被降低、模型超时仍完成审查（`pytest tests/test_risk_review_model.py`）
- [x] 6.4 实现 `lab risk train --export`（复用 `merge_and_export`）与可选 `--quantize`（`mlx_lm.convert`），验证导出模型可被 `infer` 加载并产出合法报告、量化产物可加载（`pytest tests/test_risk_export.py`）
- [x] 6.5 在 `docs/risk-review.md` 补充训练/推理/导出章节（无模型环境的降级说明、模型输出约束与失败回退），验证文档中的训练与审查命令可执行

## 7. 评测与门禁

- [x] 7.1 实现评测集加载、场景族/时间切分与训练集泄漏检测，验证统计报告、严格模式下泄漏以非零退出（`pytest tests/test_risk_eval_dataset.py`）
- [x] 7.2 实现核心指标：分场景族召回、固定阈值 FPR、资产加权召回（被拦截暴露/总暴露）、决策分布、P50/P95 延迟、模型非法输出率，验证手工构造样本的指标计算（`pytest tests/test_risk_eval_metrics.py`）
- [x] 7.3 实现 `rules_only` vs `rules_plus_model` 消融与币安回归门禁（全部 BLOCK），验证未拦截任一变体时评测失败且报告列出失败用例与决策依据（`pytest tests/test_risk_eval_gate.py`）
- [x] 7.4 实现 `lab risk eval`：输出 JSON + Markdown 报告、manifest 与版本信息，验证重复评测指标在容差内一致、历史报告对比可用（`pytest tests/test_risk_eval_cli.py`）
- [x] 7.5 在 `docs/risk-review.md` 补充评测指标、门禁阈值、消融结论解读章节，验证文档命令可生成报告并解释各项指标含义

## 8. 集成验收与文档收尾

- [x] 8.1 端到端 sanity 全链路（`lab risk data generate` → `train` → `eval` → `review`）在 10 分钟内完成，币安用例与变体均 BLOCK，验证一次实跑产物清单与 manifest 完整（可用脚本或命令序列复核）
- [x] 8.2 全量质量门禁：`pytest -q` 与 `ruff check .` 通过，验证新增测试覆盖 schema、解码/模拟、规则/决策、模型数据、评测与 CLI 关键路径（同时修复了 `docs/deep-dives/figures/make_figures.py` 的历史 lint 问题）
- [x] 8.3 完成 `docs/risk-review.md` 的局限、生产架构对照与迁移路径章节，更新 `README.md` 与 `docs/README.md` 导航，验证链接有效、文档命令可执行
- [x] 8.4 回归检查：`lab stages` 仍为 8 项、既有阶段测试全部通过、`git diff` 仅包含本变更预期文件、`openspec validate add-tx-risk-review` 通过（提交范围已核对：59 个文件均为本变更产物；工作区另有并行会话的未提交改动，未纳入本次提交）

## Workflow follow-up

- 经项目评审后归档本变更（`openspec archive add-tx-risk-review`），并验证主 spec 中生成 `tx-risk-*` 能力与 Purpose。
- 是否接入真实 fork 模拟器、公开攻击数据集或 HTTP 审查服务，留待后续独立变更。
