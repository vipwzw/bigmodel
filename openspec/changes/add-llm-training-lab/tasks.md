# Tasks

## 1. 项目骨架与环境自检（lab-runtime）

- [x] 1.1 初始化 git 仓库（无远程）与 `.gitignore`（排除 `.venv/`、`data/`、`models/`、`runs/`、`__pycache__/`、`*.safetensors` 等），验证 `git status` 不出现大文件与缓存目录
- [x] 1.2 用 uv 创建 Python 3.12 `.venv` 并配置清华 PyPI 镜像与 `uv.lock`，验证 `uv sync` 成功且 `.venv/bin/python -V` 输出 3.12
- [x] 1.3 编写 `pyproject.toml`（core/train/mlx/eval/serve/dev 依赖分组），验证 `uv sync --all-groups` 成功且能 import `torch`、`mlx`、`transformers`、`peft`、`trl`
- [x] 1.4 实现 YAML 配置加载与 pydantic 校验（`configs/` 下默认配置，含数据/模型来源优先级与可选代理设置），验证非法配置报错、合法配置可解析（`pytest tests/test_config.py`）
- [x] 1.5 实现 `lab` CLI 骨架与 8 阶段注册表（`lab stages`、`lab run <stage>`），验证未知阶段返回明确错误且不建产物、列表输出 8 项（`pytest tests/test_cli.py`）
- [x] 1.6 实现 `lab doctor`：依赖版本、MPS/MLX 可用性、内存/磁盘、ModelScope/hf-mirror/HuggingFace/PyPI 源与本地代理探测，PASS/WARN/FAIL 与退出码，验证正常、代理关闭、断网三种情况下输出符合预期（`pytest tests/test_doctor.py` + 实跑一次）
- [x] 1.7 实现实验目录与 manifest（配置快照、环境、种子、代码哈希、耗时、退出码、metrics、产物清单），验证一次运行生成完整目录结构（`pytest tests/test_run_manifest.py`）
- [x] 1.8 实现 `--profile sanity|standard`、`--max-minutes`、`--dry-run` 与下载前磁盘预检，验证超时任务优雅停止且已产出产物保留（`pytest tests/test_budget.py`）
- [x] 1.9 完成 MPS/MLX 性能基线（矩阵运算与 tiny 训练步吞吐）并输出基准报告，验证报告数据来自实跑且据此校准 design 的 token 预算（报告文件 + 复算命令）
- [x] 1.10 编写阶段 1 课程文档（学习目标、运行命令、产物说明、MPS↔CUDA/统一内存↔HBM 对照），验证文档命令逐条可执行

## 2. 数据工程（data-pipeline）

- [x] 2.1 实现多源下载封装（ModelScope / hf-mirror / 官方 Hub，可配本地代理；缓存复用、校验和、续传、磁盘预检），验证重复执行命中缓存、主源失败时按优先级回落、错误信息可操作
- [x] 2.2 实现中文维基子集与 TinyStories 获取脚本及获取清单，验证清单中的文件大小与校验和与实际一致，并验证切换 hf-mirror 备用源可用
- [x] 2.3 实现清洗管线（规范化、长度过滤、字符集/语言过滤、控制字符处理）与统计报告，验证单测覆盖各过滤规则与统计数值（`pytest tests/test_clean.py`）
- [x] 2.4 实现精确 + 近似（MinHash）去重并输出报告，验证单测对已知重复样本去重生效
- [x] 2.5 生成数据卡（来源、许可证、规模、字段、抽样样例、已知偏差），验证数据卡内容与统计报告一致
- [x] 2.6 编写阶段 2 文档（与 Spark/Ray Data、数据质量门禁等生产做法的对照），验证文档命令可执行
- [x] 2.7 数据管线集成测试：从原始子集到清洗统计全流程在 sanity 档 ≤2 分钟完成，验证 `pytest tests/test_data_pipeline.py` 通过

## 3. 分词器与数据打包（data-pipeline）

- [x] 3.1 实现 BPE 分词器训练（词表 16k/32k，`tokenizers`）与评估报告（压缩率、未知字符比例、样本编码），验证 `pytest tests/test_tokenizer.py` 通过
- [x] 3.2 实现中文指令数据到 chat 格式的转换与 schema 校验（`AI-ModelScope/alpaca-data-gpt4-chinese`），验证有效/无效样本统计单测通过
- [x] 3.3 实现偏好数据格式化（`hh-rlhf` 测试分片 + self-play 工具），验证 chosen/rejected 结构与残缺过滤统计的测试通过
- [x] 3.4 实现序列 packing 与 attention/loss 掩码及抽样校验，验证打包前后 token 计数一致、掩码单测通过
- [x] 3.5 实现可复现数据划分（比例 + 种子）与划分清单，验证相同种子重复划分结果完全一致
- [x] 3.6 编写阶段 3 文档（分词器训练、数据打包的生产实践对照），验证文档命令可执行
- [x] 3.7 集成测试：从本地语料到 packed 训练数据集全流程通过，验证 `pytest tests/test_pack.py` 与产物清单完整

## 4. 从零预训练（pretraining）

- [x] 4.1 实现 GPT 模型定义（RoPE、RMSNorm、SwiGLU、可配 GQA）与参数量/FLOPs/显存估算报告，验证前向形状与参数计数单测通过
- [x] 4.2 实现训练循环（低精度混合训练、warmup+cosine、梯度裁剪、梯度累积、每步指标 JSONL），验证 sanity 档 loss 明显下降且指标文件有据可查
- [x] 4.3 实现显存控制（梯度检查点、batch/累积配置）与 OOM 降级建议，验证开启检查点后峰值内存低于关闭时的实测值
- [x] 4.4 实现检查点保存/保留策略与断点续训（模型+优化器+调度器+数据位置+随机状态），验证中断后续训 loss 与步数连续（`pytest tests/test_resume.py`）
- [x] 4.5 实现训练监控与摘要（loss/吞吐曲线、NaN/爆炸检测、token 数/耗时/峰值内存），验证摘要数值与日志末条一致
- [x] 4.6 实现 MLX 等价训练循环并做首步数值对齐（loss 与梯度范数在声明容差内），验证对齐报告生成（`pytest tests/test_mlx_parity.py`）
- [x] 4.7 产出 MPS vs MLX 吞吐/内存对比报告，验证报告数据来自两种后端的真实运行
- [x] 4.8 实现 S/M/L 三档规模扫描与 loss-vs-compute 曲线，验证三段训练产物齐全且曲线图可生成
- [x] 4.9 编写阶段 4 文档（FSDP/ZeRO/Megatron 与 scaling law 的对照，说明本机哪些只能概念对照），验证文档命令可执行

## 5. 指令微调（supervised-finetuning）

- [ ] 5.1 实现 Qwen3-0.6B/1.7B 的 ModelScope 获取与本地加载（离线），验证断网状态下加载成功并输出参数量与内存报告
- [ ] 5.2 实现 LoRA 与 QLoRA 配置封装（秩、缩放、丢弃率、目标模块）与量化可用性探测，验证配置校验单测通过、不可用时给出替代方案
- [ ] 5.3 实现 chat template 与 loss mask 自检（打印样本解码与计损位置），验证填充/用户输入被屏蔽、自检失败会中止训练（`pytest tests/test_mask.py`）
- [ ] 5.4 实现 SFT 训练（自研训练循环 + peft，参考 TRL SFTTrainer 行为）并保存适配器与训练摘要，验证 sanity 档训练完成、摘要字段齐全
- [ ] 5.5 实现 SFT 检查点保存与断点续训，验证中断后从检查点续训成功
- [ ] 5.6 跑两组超参对比（如不同秩/学习率）并生成损失曲线、吞吐、峰值内存与产物体积对比报告，验证报告生成且配置记录在 manifest
- [ ] 5.7 实现微调前后固定提示集对比生成与适配器合并导出，验证导出模型可被推理命令加载并生成回复
- [ ] 5.8 编写阶段 5 文档（LoRA/QLoRA/全参微调与生产微调流水线的对照），验证文档命令可执行

## 6. 偏好优化（preference-optimization）

- [x] 6.1 完成偏好数据构造与校验（公开数据 + self-play 自造），验证统计报告与无效样本过滤单测通过
- [x] 6.2 实现 DPO 训练（自研训练循环、冻结参考模型、可配 beta），验证 sanity 档训练完成并输出奖励间隔/偏好准确率曲线
- [x] 6.3 实现对齐稳定性监控（近似 KL、奖励间隔、生成质量退化检测与告警/中止），验证模拟异常可触发告警（`pytest tests/test_dpo_monitor.py`）
- [x] 6.4 完成 SFT vs DPO 并排对比、评审表与固定种子复现，验证两次生成的样例逐条一致
- [x] 6.5 编写阶段 6 文档（DPO/RLHF/GRPO 的生产实践与本机取舍对照），验证文档命令可执行

## 7. 评测与实验追踪（evaluation）

- [x] 7.1 实现困惑度评测与训练/评测数据重叠检查，验证 `pytest tests/test_ppl.py` 通过且报告含 token 数与重叠比例
- [x] 7.2 实现本地任务基准（中文选择题 + 指令遵循规则任务）与评分规则版本记录，验证规则版本写入报告
- [x] 7.3 实现固定提示集批量生成评测与规则评分（含空输出/超长输出统计），验证报告包含逐条输出与汇总
- [x] 7.4 实现实验追踪（JSONL + 可选 TensorBoard + 可选 wandb offline）与本地对比报告，验证断网时报告正常生成且 manifest 标记离线模式
- [x] 7.5 实现跨实验对比与预置场景（规模对比、SFT 前后、DPO 前后、后端对比），验证每个场景报告可复现生成
- [x] 7.6 实现评测可复现验收（固定种子与采样参数、容差断言），验证重复评测指标在容差内一致
- [x] 7.7 编写阶段 7 文档（lm-evaluation-harness 经 hf-mirror/代理补充运行、LLM-as-judge 等生产评测做法对照），验证文档命令可执行

## 8. 量化推理与服务（inference-serving）

- [x] 8.1 实现 MLX 4bit/8bit 量化转换与体积/精度报告，验证转换后模型可加载并生成样例
- [x] 8.2 实现推理基准（吞吐、首 token 延迟、峰值内存、可配 batch/上下文），验证报告可复算且参数完整
- [x] 8.3 实现 CLI 生成接口（采样参数、流式输出、多轮会话），验证非法温度/top-p 在生成前被拒绝（`pytest tests/test_generate_cli.py`）
- [x] 8.4 实现 OpenAI 兼容服务（`mlx_lm.server` 或 FastAPI 封装）与契约测试，验证 `/v1/chat/completions` 正常响应、并发与非法请求错误处理（`pytest tests/test_server.py`）
- [x] 8.5 完成 transformers+MPS 基线推理对照与量化前后对比报告，验证报告数据来自实跑
- [x] 8.6 编写阶段 8 文档（vLLM/SGLang/TGI 能力对照、迁移路径、GGUF/llama.cpp 在代理可用时的可选构建说明），验证文档命令可执行

## 9. 课程整合与端到端验收

- [ ] 9.1 完成课程总目录与学习路径（`docs/README.md` + 8 阶段导航 + 时间预算表），验证所有文档链接有效
- [ ] 9.2 运行 `lab run all --profile sanity` 端到端流程（≤30 分钟），验证退出码为 0 且 8 阶段产物齐全
- [ ] 9.3 复现性验收：同配置重复运行预训练、SFT、评测关键实验，验证指标在声明容差内
- [ ] 9.4 实现并验证 `lab clean` 与磁盘预算检查，验证清理后仅保留代码、缓存与指定产物
- [ ] 9.5 完成总复盘文档（本机做法↔生产栈总结、迁移 CUDA 环境建议、后续扩展方向），验证文档评审通过
- [ ] 9.6 全量测试与静态检查通过（`pytest`、`ruff check`），验证命令输出全绿

## Workflow follow-up

- 所有任务与验收完成后，归档变更：`openspec archive add-llm-training-lab`
- 归档后检查 `openspec/specs/` 下 7 个 capability 主 spec 已正确生成
