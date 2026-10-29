# Stage 8：量化推理与服务

> 对应 capability：`inference-serving`（量化、基准、生成接口、OpenAI 兼容服务、生产对照）
> 前置：Stage 5 的 SFT 模型。产物：MLX 4bit/8bit 量化模型、推理基准、服务契约报告、CLI 验证。

## 学习目标

1. 理解**量化**的收益与代价：模型体积、显存、吞吐、精度损失。
2. 学会测量推理性能的三个关键指标：**吞吐（tokens/s）**、**首 token 延迟（TTFT）**、**峰值显存**。
3. 亲手把模型暴露成 **CLI 生成接口**和 **OpenAI 兼容 HTTP 服务**，并写契约测试。
4. 对照生产推理栈（vLLM/SGLang/TGI）理解单机实现缺了什么。

## 运行命令

```bash
.venv/bin/lab run 8 --profile sanity          # 量化 + 基准 + 服务契约，约 15 秒（量化缓存后）

# CLI 生成接口（流式、多轮）
.venv/bin/lab generate --model-path models/exports/mlx-qwen3-0.6b-sft-lora-merged-4bit \
  --prompt "用一句话介绍你自己。" --max-new-tokens 32 --stream

# 启动 OpenAI 兼容服务
.venv/bin/lab serve --model-path models/exports/mlx-qwen3-0.6b-sft-lora-merged-4bit --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 产物说明

| 路径 | 内容 |
|---|---|
| `models/exports/mlx-qwen3-0.6b-sft-lora-merged-4bit/` | 4bit 量化 MLX 模型 |
| `models/exports/mlx-qwen3-0.6b-sft-lora-merged-8bit/` | 8bit 量化 MLX 模型 |
| `runs/stage-8/<run>/artifacts/quant_report.json` | 量化体积/比例/可加载性/采样 |
| `runs/stage-8/<run>/artifacts/bench_report.json` | transformers/MPS vs MLX 各精度基准 |
| `runs/stage-8/<run>/artifacts/server_contract.json` | OpenAI 兼容服务契约测试结果 |
| `runs/stage-8/<run>/artifacts/cli_demo.txt` | CLI 生成接口实测记录 |
| `runs/stage-8/<run>/artifacts/serve_report.md` | 可读报告 + 生产栈对照 |

## 实测结果（sanity 档，2026-10-29，Qwen3-0.6B SFT 合并模型）

量化体积：

| 配置 | 体积 | 相对源模型 |
|---|---|---|
| 源模型（bf16 safetensors） | ~2.2 GB | 1.00 |
| MLX 4bit | 0.358 GB | 0.16 |
| MLX 8bit | 0.635 GB | 0.285 |

推理基准（同一提示、max_tokens=48）：

| 配置 | tokens/s | TTFT | 峰值显存 |
|---|---|---|---|
| transformers + MPS bf16 | 130.7 | 0.018 s | 1.85 GB |
| MLX bf16 | 129.7 | 0.091 s | 2.53 GB |
| **MLX 4bit** | **465.8** | 0.029 s | **0.58 GB** |
| MLX 8bit | 392.9 | 0.028 s | 0.90 GB |

> 解读：4bit 量化的收益非常明显——体积降到 16%、显存降到约 1/3、吞吐约为 bf16 的 3.6 倍。
> 这是内存受限的单机场景下最划算的优化；代价是精度损失（需通过评测/SFT 质量对比来确认是否可接受）。

OpenAI 兼容服务契约测试（全部通过）：

| 检查 | 结果 |
|---|---|
| `GET /v1/models` | ✅ HTTP 200 |
| 非流式 `POST /v1/chat/completions` | ✅ 返回 `chat.completion` 与内容 |
| 流式（SSE） | ✅ 返回 `chat.completion.chunk` 与 `[DONE]` |
| 非法请求（messages 为空） | ✅ HTTP 400 + `invalid_request_error` |
| 并发请求 | ✅ 串行化保护下全部 200 |

CLI 生成接口：`lab generate --stream` 退出码 0，输出非空（见 `cli_demo.txt`）。

## 机制说明

1. **量化**：`mlx_lm.convert(..., quantize=True, q_bits=4/8)` 做 affine 量化，直接在 Apple Silicon 上加载。
2. **基准**：每次加载一个生成器，测 48 token 的生成吞吐、首 token 时间与 MLX/Torch 显存峰值。
3. **CLI/编程 API**：`serve/generate.py` 提供 `MLXGenerator` / `TransformersGenerator` / `ChatSession`，
   支持流式、采样参数（temperature/top_p/重复惩罚）与多轮会话；`lab generate` 是 CLI 封装。
4. **服务**：FastAPI 实现 OpenAI 兼容的 `/v1/models` 与 `/v1/chat/completions`（流式 SSE），
   内部用锁保护单模型串行生成；契约测试覆盖正常、流式、非法请求、并发。
5. **契约**：`server_contract.json` 可直接作为回归基线。

## 本机做法 ↔ 生产推理栈对照

| 本机 | 生产栈（vLLM/SGLang/TGI） | 差异 |
|---|---|---|
| MLX 4bit 单请求串行 | PagedAttention + 连续批处理 + 张量并行 | 大规模并发调度能力缺失 |
| FastAPI 单进程 + 文件锁 | 多副本 + 负载均衡 + 指标/限流/鉴权 | 仅验证接口契约 |
| MLX affine 4bit/8bit | GPTQ/AWQ/FP8 + 融合内核 + KV cache 量化 | 量化思想一致，实现不同 |
| 无投机解码 | 投机解码/前缀缓存/分块预填充 | 本机未实现，仅文档说明 |
| 单机 ~466 tokens/s | 数千 tokens/s（A100/H100 集群） | 硬件差异 |

## 可选扩展（未进入验收）

- **GGUF/llama.cpp**：本机无 cmake，且需要 GitHub 源码构建；在代理可用时可按官方文档尝试
  （`llama.cpp` + `llama-quantize`），对比 MLX 与 GGUF 的量化差异。
- **mlx-lm 自带 server**：`python -m mlx_lm.server` 提供 OpenAI 兼容接口；本课用 FastAPI 封装是为了
  可控地演示契约与并发行为。
- **投机解码**：可加载一个更小的 draft 模型（如 Qwen3-0.6B 量化为 draft）做 speculative decoding 实验。

## 常见问题

- **量化很慢**：首次转换需要遍历权重（0.6B 约 10–20 秒）；之后会复用 `models/exports/` 下的产物。
- **生成含 `<think>`**：Qwen3 思考模式；可在模板中关闭（`enable_thinking=False`）或后处理截断。
- **服务端口占用**：`lab serve --port` 指定其他端口；契约测试会自动选空闲端口。
- **MLX 与 PyTorch 结果不同**：量化和采样都有随机性；用固定种子与同一提示对比才有意义。
