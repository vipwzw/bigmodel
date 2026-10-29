# bigmodel-lab

单机 Apple Silicon 大模型训练全流程实验课程（OpenSpec 规格驱动）。

> 课程文档见 `docs/README.md`；实现计划见 `openspec/changes/add-llm-training-lab/`。

## 快速开始

```bash
uv sync --all-groups
.venv/bin/lab doctor
.venv/bin/lab stages
```

## 目录结构

```
configs/     各阶段默认配置
src/bigmodel_lab/  课程实现（CLI、基座、各阶段模块）
docs/stages/ 8 个阶段课程文档
tests/       测试
data/ models/ runs/  资源与实验产物（不入库）
```
