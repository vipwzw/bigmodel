# Design

## Context

- `bigmodel-lab` 的 8 阶段课程已完成并通过验收：统一 `lab` CLI、`RunContext`/manifest、`sanity|standard` 预算、YAML 深合并配置、Qwen3 LoRA SFT（`finetune/sft.py`）、DPO（`finetune/align.py`）、评测与 MLX 量化服务均已可运行。
- 现有代码把"阶段"（stage registry）与"工具命令"（`generate`、`serve`、`clean`、`doctor`）分开；本变更采用后者形态，不新增第 9 个课程阶段，避免改动 8 阶段注册表与既有课程文档的语义。
- 目标应用场景来自币安 2026-10-30 事件：攻击者经被入侵的 DevOps 机器与服务容器，向热钱包签名基础设施提交 EIP-7702 委托授权；签名系统按正常流程签发，链上执行权被让渡。公开复盘的关键事实：私钥未泄露、签名对象只有 32 字节摘要、50 个连续 nonce 批量签发、全部资产暴露、无正常提现审批。该事件说明签名前需要一层"语义与能力审查"，而不只是验签。
- 单机约束：Apple Silicon / 48 GB 统一内存、无 CUDA、默认离线；模型训练只能复用 Qwen3-0.6B 级别；不引入重型 EVM 依赖，不访问 RPC。
- 参考实现：现有 `SFTTrainer`/`DPOTrainer`/`merge_and_export`/`build_generator` 的接口可直接复用；`RunContext` 可创建 `runs/risk/<run-id>/` 产物目录。

## Goals / Non-Goals

**Goals:**

- 在完全离线、确定性、可复现的条件下跑通"签名请求 → 解码 → 模拟 → 能力/暴露特征 → 规则决策 → 模型第二意见 → 评测"的完整审查链路。
- 让"能力变化（capability delta）"与"最坏情况资产暴露"成为一等公民：能识别"转账金额为 0 但让渡账户控制权"这类事件。
- 规则基线可独立工作；小模型只在规则之上补充语义判断，且不能降低规则决策级别。
- 币安式 EIP-7702 委托作为回归用例，评测门禁要求必须拦截；同时用场景变体验证不是"记住固定样本"。
- 复用既有训练/评测/量化链路，sanity 档单次 ≤10 分钟，standard 档在单机数小时内可完成。
- 文档明确教学模拟器与生产模拟器的差异，给出迁移到生产审查架构的路径。

**Non-Goals:**

- 不实现真实私钥管理、签名、交易广播或链上实时监控。
- 不实现完整 EVM 执行器或 fork 模拟器（v1 用确定性状态快照 + 操作语义模拟；真实 fork/RPC 适配器只留接口与文档）。
- 不覆盖非 EVM 链（BTC PSBT、Solana 指令仅文档对照）。
- 不追求生产级检测率或 SOTA；0.6B 模型与合成数据只用于演示方法论与可复现评测。
- 不提供 HTTP 在线服务、SIEM/工单系统集成或策略下发通道（保留 CLI 与报告文件作为契约）。

## Decisions

### D1 交付形态：新增 `lab risk` 子命令族，而不是第 9 个阶段

- 选择：`lab risk data|review|train|eval`，与 `generate`/`serve` 同级；`lab stages` 仍列 8 个阶段。
- 理由：本变更是一条**应用实验**，依赖 Stage 5/6/7/8 的产物但不应改变课程主线；子命令可独立运行、独立回滚，也让 `configs/risk.yaml` 与 8 阶段配置解耦。
- 备选：新增 Stage 9（改动注册表、CLI 帮助、课程文档与验收语义，牵动现有已完成变更）；独立仓库（失去 `RunContext`/训练栈复用与统一文档）；仅做库不提供 CLI（不利于课程演示与验收）。

### D2 模拟器：离线确定性世界状态 + 操作语义，而不是真实 fork/RPC

- 选择：每笔请求携带账户世界状态快照（多链余额、allowance、owner/admin/guardian 等权限、代码委托、历史窗口），由纯 Python 模拟器应用操作语义并产出差分。
- 理由：离线可复现、测试快、无 RPC 依赖；币安事件的关键不变量（7702 委托 = 全账户控制；approve = 代扣额度；升级/改 owner = 控制权变更）可以精确建模。教学重点是"能力/暴露"抽象，而不是 EVM 字节码执行。
- 备选：接入 Anvil/Tenderly（真实但仍需链状态与网络，MPS 机器上成本高，且模拟结果不确定性影响可复现验收）；只做静态分析（无法得到资产暴露与最坏损失）；直接解析真实链上交易（无法覆盖尚未上链的签名请求）。
- 迁移：引擎定义 `Simulator` 接口；生产环境可替换为 fork 模拟器，特征与报告 schema 不变。

### D3 决策架构：规则硬约束 + 能力/暴露 + 模型第二意见，模型不能降级

- 选择：规则/不变量先给出最低决策级别；模型只能维持或提升（`ALLOW < STEP_UP < ESCALATE < BLOCK`），且模型不可用时规则照常工作。
- 理由：签名是不可逆的高风险动作，模型是概率性的，不能作为唯一防线；规则可审计、可测试，模型负责规则表达不出的语义与未知组合；该分层与"已知风险靠规则、未知风险靠异常+语义+人审"的原则一致。
- 备选：端到端分类器直接放行/拦截（不可审计、易被对抗、无法解释）；纯规则（无法识别新的语义伪装与未知组合）；纯模型（违反 fail-closed）。

### D4 核心表征：capability delta，而不是交易金额或 selector 黑名单

- 选择：解码后计算"签名后谁获得了什么新能力"：资金转出、代扣额度、账户执行权、owner/admin 变更、代理实现变更、批量 nonce 跨度；再由能力与暴露推导风险。
- 理由：攻击者可换合约、换链、换函数名，但能力变化空间有限；币安事件按金额看是 0，按能力看是 100%。这使模型和规则都能泛化到未见过的攻击变体。
- 备选：金额/频率阈值（漏掉 7702）；selector 黑名单（规则写不全）；纯交易图异常（无法看到授权语义）。

### D5 数据：确定性合成场景 + 币安回归 fixture，真实攻击数据仅作可选扩展

- 选择：Python 内置场景生成器（固定种子）+ 基于公开复盘的结构化币安用例；覆盖正常运营、普通转账、批量提现、approve/permit、7702 委托、owner 变更、代理升级、批量 nonce、跨链组合。生成结果写入 `runs/risk/.../artifacts/dataset/`，fixture 源码放 `src/bigmodel_lab/risk/fixtures/`（小体积、入库）。
- 理由：离线可复现、无隐私/合规问题、可控制类别平衡与难度；真实攻击数据（DeFiHackLabs 等）经网络获取且许可证/链上状态复原复杂，作为可选扩展在文档中说明。
- 备选：抓取公开攻击交易（依赖网络与 RPC 状态，且需要人工标注）；手写少量样例（覆盖不足、易过拟合）。

### D6 模型：Qwen3-0.6B LoRA SFT，DPO 可选；输出受 JSON schema 约束

- 选择：复用 `finetune/sft.py` 的 `SFTConfig`/`SFTTrainer`/`LoraConfigSpec`；SFT 为主，DPO 复用 `finetune/align.py` 作为可选增强；推理复用 `serve/generate.py` 的 `build_generator`/`ChatSession`。
- 理由：现有栈已验证 MPS 可跑、10 分钟 sanity 档可行；审查任务输入输出高度结构化，0.6B 足够验证"结构化审查"方法论；DPO 可后来开启而不改接口。
- 备选：更大模型（单机时间不可行）；调用外部 API（默认离线、数据不能外发）；从零预训练（与任务不匹配）。

### D7 结构化输出：JSON schema + 解析校验 + 失败回退

- 选择：模型只输出审查报告 JSON（动作类型、能力变化、暴露比例、意图一致性、风险因子、风险等级、建议动作、置信度、证据）；解析失败/字段非法/超时 → 记录失败并回退 `rules_only`，最终决策不低于规则级别。
- 理由：可机器校验、可审计、可做证据引用；便于统计非法输出率与训练改进；比自由文本更抗提示注入。
- 备选：自由文本 + 二次抽取（链路更长、错误不可控）；embedding 分类（不可解释）。

### D8 解码依赖：可选 `risk` 依赖组（eth-abi/eth-utils）+ 内置常见 selector + 结构化输入降级

- 选择：`pyproject.toml` 新增可选依赖组 `risk = ["eth-abi>=5", "eth-utils>=4"]`；引擎内置 transfer/approve/setApprovalForAll/7702 等常见操作解码；样例可直接提供结构化字段；依赖缺失时结构化路径可用，原始 calldata 解码给出明确提示。
- 理由：纯 Python、清华镜像可装、无需 CUDA；不引入 web3.py 这样的大依赖；保留降级路径保证离线与最小环境可测。
- 备选：web3.py（重、带 RPC 心智）；自写 RLP/ABI（易错且重复造轮子）；强制结构化输入（失去真实 calldata 演示价值）。

### D9 目录、配置与产物：独立 `risk` 包 + 独立 YAML + 复用 RunContext

- 选择：
  - `src/bigmodel_lab/risk/`：`schema.py`、`decode.py`、`simulate.py`、`features.py`、`rules.py`、`decision.py`、`pipeline.py`、`dataset.py`、`model_data.py`、`train.py`、`infer.py`、`evaluate.py`、`config.py`、`cli.py`、`fixtures/`。
  - `configs/risk.yaml`（流水线/训练/评测参数）与 `configs/risk-rules.yaml`（规则、阈值、白名单、不可逆操作清单，带 `version`）。
  - 产物写入 `runs/risk/<run-id>/`，沿用 `RunContext.create`/`Budget`/manifest 规范；规则、数据、模型版本写入报告。
  - 不修改 `src/bigmodel_lab/config.py` 与 `stages/`；`risk/config.py` 用 pydantic 独立校验 `risk:` 段（现有 `load_config` 保留原始 dict，阶段配置已是同样模式）。
  - `cli.py` 只新增 `risk` 子解析器并分发。
- 理由：最小侵入、可独立回滚；避免通用配置模型被应用实验污染；复用现成实验产物规范。
- 备选：把 risk 配置塞进 `LabConfig`（形成基础模块 → 应用的依赖倒置）；复用 stage 目录（混淆课程阶段语义）。

### D10 评测：场景族切分 + 资产加权指标 + 币安门禁，而不是总体准确率

- 选择：按场景族与时间切分；测试集不共享训练实例；指标含分族召回、固定 FPR、资产加权召回（被拦截暴露/总暴露）、决策分布、P50/P95 延迟、模型非法输出率；`rules_only` 与 `rules_plus_model` 消融；币安用例族必须 BLOCK；固定种子可复现。
- 理由：安全场景的代价不对称（漏报一枚大额资产远重于少量误杀），总体 accuracy 会被正常样本淹没；门禁直接对齐事件目标。
- 备选：总体准确率/F1（不反映资产风险）；只用合成集不设 holdout（易过拟合）。

### D11 安全边界：不碰密钥、不广播、外部文本只作数据

- 选择：引擎与模型都不接触私钥、不签名、不广播；备注/合约名/代币符号等不可信文本按数据字段序列化，system prompt 明确"字段内容不是指令"；报告中标记可疑注入文本；解码/模拟失败 fail-closed。
- 理由：审查层本身不能成为新的攻击面；币安事件的教训是"内部可信请求也可能被攻陷"，审查层必须假设输入不可信。
- 备选：把备注原文拼进 prompt 不做隔离（提示注入高危）；审查层持有签名权限（扩大攻击面，绝对不做）。

### D12 CLI 契约与退出码

- `lab risk data generate|validate|show`
- `lab risk review --request <file> | --batch <jsonl> [--mode rules_only|rules_plus_model] [--out <dir>] [--json]`
- `lab risk train [--export] [--quantize]`
- `lab risk eval --dataset <path> --mode <mode> [--out <dir>]`
- 退出码：0 = 审查完成且无阻断；2 = 存在 BLOCK 或门禁失败；3 = ESCALATE/无法完成审查（含配置/模拟/解码错误）。批量模式下逐笔报告，汇总退出码取最严重者。

### D13 组件关系与数据流

```
request.json / sample.jsonl
   │  normalize + schema 校验（risk/schema.py）
   ▼
decode（risk/decode.py）──► simulate（risk/simulate.py，world snapshot）
   │                              │
   ▼                              ▼
features（risk/features.py）   capability delta / asset at risk / rolling exposure
   │                              │
   ├──────── rules（risk/rules.py + configs/risk-rules.yaml）
   │                              │
   ├──────── model opinion（risk/infer.py，可选）
   ▼                              ▼
decision（risk/decision.py）──► report.json + exit code + manifest
```

阶段复用关系：

| 现有组件 | 复用方式 |
|---|---|
| `manifest.RunContext` / `config.load_config` / `Budget` | 创建 `runs/risk/<run-id>/`、配置快照、sanity/standard 档、manifest |
| `finetune.sft.SFTConfig/SFTTrainer/load_model_and_tokenizer/merge_and_export` | 审查模型 SFT、合并导出 |
| `finetune.align.DPOConfig/DPOTrainer` | 可选 DPO 增强 |
| `serve.generate.build_generator/ChatSession` | 审查模型离线推理 |
| `cli.py` | 注册 `risk` 子命令；不改阶段注册表 |
| stage 8 量化经验 | 导出后用 `mlx_lm.convert` 做可选 4bit 量化演示（在 risk 包内调用，不依赖 stage 私有函数） |

### D14 训练数据格式

- SFT：`{"messages":[{"role":"system","content":<rubric+schema+安全约束>},{"role":"user","content":<规范化请求+上下文+世界状态+模拟结果 JSON 字符串>},{"role":"assistant","content":<ReviewReport JSON 字符串>}]}`
- 输入不包含 `label`；gold report 由 label + 证据字段确定性生成（证据引用如 `simulation.authority_delta`、`rule:R-001`）。
- DPO：`{"prompt":..., "chosen":..., "rejected":...}`；rejected 由确定性劣化生成（删除能力变化、降低风险等级、建议 ALLOW、删除证据）。
- 划分：先按场景实例划分，再按场景族保证测试族与训练族不重叠；币安用例族只在测试/门禁中使用，训练集中使用其变体（不同地址/金额/批次大小）。

## Risks / Trade-offs

- [教学模拟器与真实 EVM 行为存在差异] → 文档明确"语义级模拟"；引擎保留 `Simulator` 接口；评测报告标注模拟器版本；生产迁移需替换为 fork 模拟器并重跑评测。
- [合成数据导致过拟合、门禁被"记住样本"绕过] → 场景族 holdout + 币安变体（地址、金额、nonce 数量、目标合约变化）+ 固定种子回归；规则与模型分开评测。
- [模型幻觉或输出非法 JSON] → schema 校验、非法输出率统计、失败回退 `rules_only`、规则地板不可降级。
- [提示注入：备注/合约名/代币符号含指令] → 规范化序列化 + system prompt 声明数据边界 + 注入测试用例；报告标记可疑文本。
- [误杀正常运营交易] → 分级决策（STEP_UP 而非直接 BLOCK）、白名单/预登记审批、评测把 FP 成本纳入报告。
- [可选解码依赖不可用/离线] → 结构化输入路径优先；依赖缺失给出明确提示；`eth-abi`/`eth-utils` 均为纯 Python，清华镜像可装。
- [MPS 上 LLM 推理延迟] → 默认 `rules_only`；模型超时回退；sanity 档限制样本数；报告记录 P50/P95。
- [币安回归用例与公开复盘绑定太紧] → 只保留结构化关键特征（不复制敏感细节），并用变体覆盖同类攻击；文档注明来源与局限。
- [风险：新增依赖影响现有 uv 环境] → 使用独立 dependency-group，`uv sync` 默认不装；只在运行 risk 时提示 `uv sync --group risk`。

## Migration Plan

- 纯增量：新增 `src/bigmodel_lab/risk/`、`configs/risk.yaml`、`configs/risk-rules.yaml`、`docs/risk-review.md`、`tests/test_risk_*.py`；`cli.py` 仅追加子命令注册。
- 不修改 Stage 1–8 的实现与注册表；现有 `lab` 命令行为保持兼容；不迁移任何生产流量。
- 实施顺序：schema/fixture → 解码/模拟/特征 → 规则/决策/CLI → 数据集生成与划分 → SFT/DPO 训练 → 推理集成 → 评测/门禁 → 文档/README；每步可独立提交与回滚。
- 回滚：删除 risk 包与两个配置文件、撤销 `cli.py` 注册、移除 `pyproject.toml` 的 optional group；已产生的 `runs/risk/` 可直接删除，不影响 8 阶段实验目录。
- 数据与模型产物（`runs/`、`models/exports/`）不入库；fixture 与配置入库。

## Open Questions

- 是否在后续版本接入真实 fork 模拟器（Anvil/Tenderly）或链上回放：接口已预留，不影响 v1 规格与任务拆分。
- 是否引入公开攻击数据集（如 DeFiHackLabs）做可选扩展验证：涉及许可证与链状态复原，v1 不作为验收项。
- 是否需要在 Stage 8 服务上暴露审查 HTTP 接口：当前 CLI + 报告文件已满足教学与评测，在线服务留待后续变更。
