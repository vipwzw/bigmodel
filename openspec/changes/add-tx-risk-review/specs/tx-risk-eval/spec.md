# Spec Delta

## Purpose

对审查引擎与审查模型进行可复现、可解释的量化评测：在场景化标注集上统计攻击召回、误拦截、资产加权召回与延迟，比较规则基线与规则+模型组合，并把币安事件回归作为必过门禁。

## ADDED Requirements

### Requirement: 评测集与切分
系统 SHALL 维护与审查报告字段对齐的标注评测集，覆盖各场景族与币安回归用例；SHALL 支持按场景族与时间切分，并保证测试集与训练集、评测集与训练集之间不共享同一场景实例。

#### Scenario: 评测集加载
- **WHEN** 用户加载评测集
- **THEN** 系统校验 schema、标签完整性与场景族分布，并报告样本统计

#### Scenario: 泄漏检测
- **WHEN** 评测集与训练集存在重复场景实例
- **THEN** 系统报告泄漏并在严格模式下以非零退出码结束

### Requirement: 核心指标
系统 SHALL 报告按场景族的召回率、固定阈值下的误报率、资产加权召回（被拦截暴露/总暴露）、决策分布、P50/P95 延迟与规则命中分布；指标 MUST 记录规则、模型与数据版本。

#### Scenario: 全量评测
- **WHEN** 用户对规则基线运行评测
- **THEN** 输出 JSON 与 Markdown 报告，包含上述指标及版本信息

#### Scenario: 延迟统计
- **WHEN** 评测批量样本
- **THEN** 报告给出 P50/P95 延迟与最慢样本的耗时分解

### Requirement: 消融与币安回归门禁
系统 SHALL 对比 rules_only 与 rules_plus_model 两种模式并报告逐场景族差异；币安事件回归用例 MUST 全部被拦截（BLOCK），存在未拦截时 MUST 以非零退出码结束。

#### Scenario: 消融对比
- **WHEN** 用户运行两种模式的对比评测
- **THEN** 报告给出每类场景的指标差异、模型净收益与误杀代价

#### Scenario: 回归失败
- **WHEN** 任一币安回归用例未被判定为 BLOCK
- **THEN** 评测失败、报告列出失败用例与决策依据，并以非零退出码结束

### Requirement: 可复现与报告
系统 SHALL 支持固定种子与确定性输出，重复评测的指标 MUST 在声明容差内一致；报告 MUST 记录数据、规则、模型与代码版本，并支持与历史评测对比。

#### Scenario: 重复评测
- **WHEN** 用户以相同输入与配置重复运行评测
- **THEN** 指标差异在声明容差内，报告标注两次运行关系

#### Scenario: 历史对比
- **WHEN** 用户选择两个历史评测报告进行对比
- **THEN** 输出指标差异表与回归项列表

### Requirement: 评测 CLI
系统 SHALL 提供 `lab risk eval` 子命令，支持指定数据集、规则配置、模型与模式，输出 JSON/Markdown 报告到规范化实验目录，并在回归失败或指标低于声明门限时返回非零退出码。

#### Scenario: 运行评测
- **WHEN** 用户执行 `lab risk eval --dataset <path> --mode rules_plus_model`
- **THEN** 生成报告与 manifest，退出码反映是否通过门禁

#### Scenario: 门限未达标
- **WHEN** 召回或误报率未达到配置门限
- **THEN** 命令以非零退出码结束并指出未达标项
