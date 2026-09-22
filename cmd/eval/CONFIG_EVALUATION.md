# 通用 checkpoint 评测

统一入口：`scripts/run_eval.py`。它既支持原来的单个 JSONL 评测，也支持按顺序混合运行
AIME JSONL 与 GEM/ACEBench 等离线 suite。`scripts/run_benchmark_eval.py` 现在只是兼容旧命令的
转发层，新命令统一使用 `run_eval.py`。
宿主机只需要 Python 3.10+ 和可使用 GPU 的 Docker；模型推理依赖来自统一的
`envduels-unified:cu124` 镜像，无需手动启动推理服务。

```bash
cd /data1/shibo517/envduels/spade

# 只检查配置、模型目录、数据格式和启动计划，不占用 GPU
python scripts/run_eval.py --config configs/evaluation.json --dry-run

# 先验证链路：第一道题，每题 8 次采样
python scripts/run_eval.py --config configs/evaluation.json --max-problems 1

# 完整 AIME26 pass@8
python scripts/run_eval.py --config configs/evaluation.json

# 换成 AIME25
python scripts/run_eval.py --config configs/evaluation.json \
  --data data/aime25/aime2025.jsonl

# 换模型和数据，也可以直接修改配置中的 checkpoint、data
python scripts/run_eval.py --config configs/evaluation.json \
  --checkpoint /absolute/path/to/hf_checkpoint \
  --data /absolute/path/to/test.jsonl \
  --output-dir outputs/my_evaluation
```

## 顺序运行多个评测

`configs/qwen38_all_evals.json` 已按以下顺序配置：

1. AIME 2025 Avg@8
2. AIME 2026 Avg@8
3. GEM（Reasoning-Gym、LCB-v6、GPQA-D）

完整运行配置中的全部项目：

```bash
python scripts/run_eval.py --config configs/qwen38_all_evals.json
```

也可以用一个 `--evals` 后跟多个名字来选择项目并明确执行顺序。例如只运行 AIME25、
AIME26 和 GEM，并保证前一个结束后才开始下一个：

```bash
python scripts/run_eval.py \
  --config configs/qwen38_all_evals.json \
  --evals aime25 aime26 gem
```

只验证一题、一次采样，不启动完整 suite：

```bash
python scripts/run_eval.py \
  --config configs/qwen38_all_evals.json \
  --evals aime25 \
  --max-problems 1 \
  --samples-per-problem 1
```

先做完整的宿主机配置、模型和数据校验但不占 GPU：

```bash
python scripts/run_eval.py \
  --config configs/qwen38_all_evals.json \
  --evals aime25 aime26 gem \
  --dry-run
```

`evaluations` 数组中的每个元素必须有唯一的 `name`。JSONL 项使用 `type: "jsonl"`
和 `data`；离线 suite 使用 `type: "suite"`、`config` 和 `suites`。数组顺序是默认运行顺序，
`--evals` 的参数顺序会覆盖它。相邻且 `context_length` 相同的项目复用一个 vLLM；配置变化时
自动重启服务。当前 AIME25/26 共用 40960 上下文服务，随后只重启一次，以 32768 上下文运行
GEM。GEM 配置只包含 Reasoning-Gym 的 math、algorithmic、cognition、logic 四类，以及
GPQA-D 和 LCB-v6，不再运行 ACEBench。任何一步失败时立即停止，不会把后续评测误报为已完成。

当前统一配置沿用已完成 AIME run 的生成协议，但把 `samples_per_problem` 和 `avg_at` 都设为
8，只进行 Avg@8，不再生成 Avg@32 所需的额外样本。输出改到
`outputs/evaluation/qwen38-all`，GPU 改为 8、9。AIME 数据仍使用已验证的本地文件：
`/home/yzo/home/spade/workspace/aime-2025/aime-2025.jsonl` 和
`/home/yzo/home/spade/workspace/aime-2026/aime-2026.jsonl`。

直接运行即开始评测；只有 `--dry-run` 不会启动模型。
`--max-problems 1 --samples-per-problem 1` 可进一步缩短链路验证，但此时只计算 pass@1。
每题最多生成 `max_tokens` 个 token；一道题也可能需要较长时间。

## 修改配置

主要修改 `evaluation` 中这些字段；命令行同名参数优先：

| 配置项 | 含义 |
| --- | --- |
| `checkpoint` | 完整本地 Hugging Face 基础模型目录，含权重、config、tokenizer |
| `lora` | 可选 PEFT adapter 目录；为 `null` 时评测基础模型 |
| `data` | 本地测试集 JSONL 文件 |
| `output_dir` | 输出父目录，每次自动创建时间戳子目录 |
| `samples_per_problem` | 每题采样数 k，任意正整数 |
| `max_problems` | `null` 测全部，正整数取前 N 题 |
| `benchmark` | `boxed_integer` 整数比较，或 `boxed_exact_match` 字符串精确比较 |
| `prompt_key` / `answer_key` / `id_key` | 数据字段名；`id_key: null` 时自动使用行号 |
| `prompt_suffix` | 追加到问题末尾的输出格式要求；空字符串表示不追加 |
| `gpu_ids` / `tensor_parallel` | 可见 GPU 编号及张量并行数，两者数量需一致 |
| `chat_template_kwargs` | 模型聊天模板参数；当前 Qwen 使用非 thinking 模式 |
| `max_tokens` / `context_length` | 单条回答上限 / 输入加输出的总上下文长度 |
| `evaluations` | 多评测有序列表；不设置时保持单个 `data` 的旧行为 |
| `max_concurrent` | `eval_offline` suite 的客户端并发上限 |
| `port` / `served_model_name` | 共享 vLLM 服务端口及请求中的模型名 |
| `wandb_enabled` | 是否把整个有序评测作为一个 W&B run 记录 |

配置中的 checkpoint、data、output_dir 以及相应命令行覆盖值，相对路径均以 **spade 根目录**
为基准，与执行命令时所在目录无关；也支持绝对路径。`--config` 本身的相对路径按当前工作目录解析。

配置只接受 JSON。这样训练与评测使用同一种格式，并且宿主机不需要安装 PyYAML。
JSON 不支持注释，字段说明以本文档为准；未知字段会被拒绝，避免拼写错误被静默忽略。

当前设置沿用本节点已验证过的 GPU 4、5、6、7，TP=4、BF16、eager 模式。
更换模型时，应确认镜像中的 vLLM 支持该模型，并调整显存、上下文和模板参数。
纯文本模型通常可将 `limit_mm_per_prompt` 和 `chat_template_kwargs` 设为 `{}`。
checkpoint 始终是完整 HF 基础模型。LoRA 目录通过 `lora` 或命令行
`--lora /path/to/adapter` 单独传入，目录需包含 adapter_config 和 adapter 权重。

## 测试数据

默认每行一条，支持任意非零题数；ID 必须唯一，标准答案必填：

```json
{"id":"example-1","problem":"What is 1+1?","answer":"2"}
{"id":"example-2","problem":"What is 3+4?","answer":7}
```

也支持聊天格式：

```json
{"id":"example-1","prompt":[{"role":"user","content":"What is 1+1?"}],"label":"2"}
```

此时设 `prompt_key: "prompt"`、`answer_key: "label"`。消息必须以 user 结尾；
标准答案只用于评分，不加入请求。已有完整提示词时可设置 `prompt_suffix: ""`。
该入口用于文本问答评测，当前两种评分器都要求模型的最终答案写在 `\boxed{...}` 内；
代码执行、交互环境和语义等价评分需要增加对应评分器。

## 输出与比较

每次运行生成 `outputs/evaluation/<UTC时间戳>/`。单评测保持原目录布局；多评测会生成
`01-aime25/`、`02-aime26/`、`03-gem/` 这样的有序子目录。主要包含：

- `scores.json`：全部选中题目成功完成后才写入最终分数。
- `responses.jsonl`：每题每次采样的回答、标准答案、预测答案、是否正确、结束原因。
- `config.json` / `resolved_config.json`：实际配置、数据哈希、所选题目 ID 等。
- `launch.json` / `runtime.json`：Docker 镜像 ID 和推理依赖版本。
- `console.log` / `server.log`：进度和推理服务日志；启动后会打印日志路径。
- `status.json`：running、completed、failed 或 interrupted。
- `sequence_status.json`：当前正在运行哪一步，以及已完成的评测。
- `sequence_results.json`：每完成一步立即更新的汇总和耗时。

`pass_at_8` 是每题恰好生成 8 个回答、至少一个正确的题目比例；`avg_at_8` 是所有回答的平均正确率。
分数范围为 0 到 1。同时记录无法提取答案的比例和因长度上限截断的比例。
少于预期的返回样本数会使本次运行失败，不会缩小分母来报告最终分数。
中断时保留已完成的回答，Ctrl-C 会停止本次启动的容器。重新运行创建新目录，不自动续跑。

`max_problems` 产生子集结果，`scores.json` 会标记 `is_subset`，不能作为完整 AIME 成绩。
默认非 thinking、8192 输出上限用于当前链路和自定义协议比较，不等同于模型官方榜单设置。
比较 baseline 与 RL 模型时只更换 checkpoint，保持数据、提示词、采样数和生成参数一致。

CPU 测试（不会启动 Docker 或占用 GPU）：

```bash
python -m unittest discover -s tests -p test_run_eval.py -v
```
