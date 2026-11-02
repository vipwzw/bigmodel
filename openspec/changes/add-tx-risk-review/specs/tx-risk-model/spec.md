# Spec Delta

## Purpose

把结构化审查样本转换为训练数据，基于既有 Qwen3 LoRA 链路训练交易审查小模型，并让模型以受约束的风险报告形式作为规则引擎的第二意见，同时保留规则基线可用。

## ADDED Requirements

### Requirement: 审查训练数据构建
系统 SHALL 将审查样本转换为 SFT 对话样本（输入结构化请求与上下文，输出受 schema 约束的审查报告），并 SHALL 支持构造 DPO 偏好对（完整有证据的报告为 chosen，漏报或弱化风险的报告为 rejected）；转换 MUST 确定性可复现且不把标签泄漏进模型输入。

#### Scenario: SFT 样本生成
- **WHEN** 用户从标注数据集生成训练数据
- **THEN** 输出包含 system/user/assistant 字段的 JSONL，并通过 schema 与标签泄漏检查

#### Scenario: 偏好对生成
- **WHEN** 用户生成 DPO 数据
- **THEN** 每条样本包含 chosen/rejected 两份报告，rejected 至少缺少一类关键风险证据

### Requirement: 审查模型训练
系统 SHALL 提供 `lab risk train` 子命令，复用现有 SFT（可选 DPO）训练栈与 sanity/standard 档，记录配置、manifest、指标与检查点；训练 MUST 支持离线运行与断点续训。

#### Scenario: sanity 训练
- **WHEN** 用户以 sanity 档运行 `lab risk train`
- **THEN** 在声明时间内完成训练并产出检查点、指标与训练摘要

#### Scenario: 非法输出统计
- **WHEN** 模型输出不符合审查报告 JSON schema
- **THEN** 评测将该项计为失败样本并在指标中单独统计

### Requirement: 结构化输出约束
系统 SHALL 约束模型输出为审查报告 JSON schema（动作类型、能力变化、暴露比例、意图一致性、风险因子、风险等级、建议动作、置信度、证据），信息不足时 MUST 允许并鼓励输出 uncertain/ESCALATE。

#### Scenario: 合法输出
- **WHEN** 模型返回通过 schema 校验的报告
- **THEN** 集成层接受并记录该报告

#### Scenario: 非法或信息不足
- **WHEN** 模型输出缺字段、超范围或明确表示信息不足
- **THEN** 集成层将非法输出标记失败并回退规则基线；信息不足时决策不得为 ALLOW

### Requirement: 规则与模型集成
系统 SHALL 在审查中支持 rules_only 与 rules_plus_model 两种模式；模型不可用、超时或输出非法时 MUST 自动回退 rules_only 并标记降级；模型 MUST NOT 将规则要求的最低决策级别降低（例如把 BLOCK 改为 ALLOW）。

#### Scenario: 模型超时回退
- **WHEN** 模型推理超时或推理依赖缺失
- **THEN** 审查以 rules_only 完成、报告标记降级，最终决策不低于规则基线

#### Scenario: 模型加重风险
- **WHEN** 规则给出 STEP_UP 而模型识别出新的高风险能力变化
- **THEN** 最终决策可提升为 BLOCK，并在报告中同时保留规则与模型证据

### Requirement: 模型导出与复用
系统 SHALL 支持将训练后的审查模型合并/导出，并可复用既有量化与本地推理服务链路加载；导出模型 MUST 能被审查集成层加载并产出合法 JSON 报告。

#### Scenario: 导出后可加载
- **WHEN** 用户导出审查模型并以其运行单笔审查
- **THEN** 模型加载成功并输出通过 schema 校验的报告

#### Scenario: 无模型环境
- **WHEN** 环境未训练或未下载审查模型
- **THEN** `lab risk review` 仍以 rules_only 模式完成，不因缺模型而失败
