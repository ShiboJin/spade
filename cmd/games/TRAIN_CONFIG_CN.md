# Qwen3.8-27B EnvDuels LoRA GRPO 配置说明

训练配置是 `configs/train_qwen38_envduels_lora.json`。JSON 不能写注释，因此参数说明集中在这里。

## 主机内存保护

2026-09-19 的三次故障是宿主机 RAM 和 swap 耗尽。除了权重加载外，
Transformers 5.12.1 还会在 FSDP worker 上用 `zeros_like` 创建完整 CPU 参数，
八个进程的内存相加超过本机容量。`fsdp_ram_loader.py` 现在同时跳过 worker
读取权重和参数清零，参数由 rank 0 广播覆盖；buffer 保留上游初始化行为。
该补丁只支持固定的 Transformers 版本，升级依赖时必须重新验证。

请使用 `python3 scripts/run_train.py` 或 `bash scripts/unified_runtime.sh train`：

- `training.memory_limit_gib` 默认 160：整个训练容器所有进程合计的 RAM 硬上限。
- `training.host_memory_reserve_gib` 默认 48：启动时必须至少有 `160 + 48 = 208 GiB`
  可用内存，低于要求直接拒绝启动。预留量不能低于 32 GiB。
- 运行时每 2 秒检查主机可用内存；低于预留量的一半（默认 24 GiB）就终止本次
  容器，并把原因写入本次运行的 `status.json`。内核硬上限不依赖这个轮询。
- 同一 checkout 的训练和评估互斥，且启动时检查已有带保护标签的容器。
- 模型加载前再次读取 cgroup 文件，确认限制确实生效；保护缺失则拒绝加载模型。

本机 cgroup v1 没有 swap limit accounting，Docker 的 `--memory-swap` 会发出警告。
因此同时设置 `--memory-swappiness=0`，并验证实际 `memory.swappiness` 为 0。
这会使容器达到内存上限时触发容器内 OOM，而非继续依赖 swap。
其语义见 [Linux memory controller 文档](https://docs.kernel.org/admin-guide/cgroup-v1/memory.html#swappiness)。
全局内存压力仍可能来自其他用户/任务，不能把这层保护理解为整机永不故障的保证。

`scripts/run_eval.py` 也使用相同保护，默认 `evaluation.memory_limit_gib=96`、
`evaluation.host_memory_reserve_gib=48`。直接运行其他旧脚本、自行启动 Docker 或
构建镜像不受这两个入口的保护。`--dry-run` 只检查配置，不启动任务也不检查实时余量。
无需重建镜像；代码和配置通过 bind mount 生效。

验证覆盖 CPU 回归、限额小容器和双 GPU 小模型的 FSDP 权重同步；
完整 27B 训练仍需另行在这些限制下验证，任务自身仍可能 OOM 退出。

容器使用宿主机数字 UID 运行，并显式设置 `USER=envduels`。不要删除这个环境变量：
镜像的 `/etc/passwd` 不一定有宿主机 UID，PyTorch Inductor 即使设置了缓存目录也会
调用 `getpass.getuser()`。用户名缺失会导致首次导入中断，后续导入可能显示
`Artifact of type=inductor already registered`，而不是最初的 UID 查询错误。
vLLM、FlashInfer、CUDA 的缓存及 XDG 配置路径也显式指向可写的 `/tmp`，
并关闭 vLLM usage stats，避免数字 UID 的默认 home 路径导致权限错误。

## 当前实验会运行多少数据

当前四个核心参数是：

```json
"fixed_pool_epochs": 3,
"num_games_per_rollout": 6,
"trajectories_per_game": 4,
"batch_size": 24
```

它们表示：

- 每个 generation batch 选择 6 个不同环境；
- 每个环境生成 4 条独立轨迹，这 4 条构成一个 GRPO reward group；
- 每个 generation batch 共生成 `6 × 4 = 24` 条 episode；
- 90 个环境每个 epoch 分成 `90 ÷ 6 = 15` 个 batch；
- 3 个 epoch 共 45 次 optimizer update；
- 每个环境每个 epoch 运行 4 条，整次实验运行 `4 × 3 = 12` 条；
- 每个 epoch 360 条 episode，整次实验 1080 条。

如果目标仍是“每个环境总共恰好 8 条”，有两种直接配置：

更接近 Inkling 的四条一组、固定池跑两遍：

```json
"fixed_pool_epochs": 2,
"num_games_per_rollout": 6,
"trajectories_per_game": 4,
"batch_size": 24
```

或者：

八条一组、固定池只跑一遍：

```json
"fixed_pool_epochs": 1,
"num_games_per_rollout": 3,
"trajectories_per_game": 8,
"batch_size": 24
```

第二种的 GRPO 组更大，组内 advantage 通常更稳定；第一种更接近 Inkling 的 `trajectories_per_game=4`，但同一个环境的两组轨迹来自两个 epoch。

## Batch 必须满足的约束

第一条关系是：

```text
batch_size = num_games_per_rollout × trajectories_per_game
```

当前使用 8 张 GPU，`per_device_train_batch_size=1`，所以全局 micro-batch 是：

```text
global_micro_batch = 8 GPU × 1 = 8
```

第二条关系是：

```text
batch_size % global_micro_batch = 0
gradient_accumulation_steps = batch_size ÷ global_micro_batch
```

因此 `5 × 4 = 20` 虽然满足第一条关系，但 20 不能被 8 整除，ms-swift 无法在八个进程上平均切分。当前 `6 × 4 = 24` 可以被 8 整除，自动得到梯度累积 3。

当前 ms-swift 的 sampler 会丢弃不完整的 tail。要保证一个 epoch 覆盖全部 90 个环境，还需要：

```text
90 % num_games_per_rollout = 0
```

launcher 会检查以上约束并在启动 GPU 前报错。

## 固定池和训练时长

- `fixed_pool_seed`：为每个环境生成唯一、可复现的 reset seed。改变它以后需要删除旧的生成数据 `data/envduels/fixed90-swift.jsonl`，再运行 `bash scripts/unified_runtime.sh prepare`；否则 launcher 会拒绝 seed 不一致的数据集。
- `fixed_pool_epochs`：完整遍历 90 个环境的次数。总轨迹数为 `90 × trajectories_per_game × fixed_pool_epochs`。
- `max_steps`：显式限制 optimizer update 数。为 `null` 时按 `fixed_pool_epochs` 运行；设为整数时优先按 update 数停止，不能保证完整覆盖全部环境。
- `dataset_shuffle`：每个 epoch 是否打乱 90 个环境的顺序。建议保持 `true`。
- `num_substeps`：同一批 rollout 被用于多少轮更新，对应 ms-swift `num_iterations`。建议先保持 1；增大会节省生成成本，但会让后续更新更 off-policy。

## GRPO rollout

- `num_games_per_rollout`：一个 generation batch 中包含多少个不同环境。
- `trajectories_per_game`：每个环境生成多少条独立轨迹，也是 GRPO group size。
- `batch_size`：一次 generation 产生的 episode 总数，必须等于前两项的乘积。
- `per_device_train_batch_size`：每张 GPU 每个 micro-step 训练几条轨迹。27B + 24GB 4090 建议保持 1。
- `remove_constant_reward_groups`：仅在关闭 SAGE 时控制原有 constant-group 重采样。启用 SAGE 时，全 1 组直接屏蔽，全 0 组使用第一个作者 hint 重采一次；最终只保留 0/1 混合组，不换环境补采。
- `max_rollout_attempts`：启用 SAGE 时必须为 `1`，表示不换环境补采；hint 重采一次独立于这个计数。关闭 SAGE 时控制原有重采样次数。
- `min_valid_groups`：SAGE 更新所需的最低有效组数，默认 `2`。每组 4 条轨迹时至少 8 条有效轨迹才更新；不足则跳过整个更新窗口，包括 optimizer 和学习率 scheduler。

## 优化器和 loss

- `learning_rate`：LoRA 学习率，当前 `1e-6` 来自 Inkling 配置。首轮不建议提高。
- `warmup_ratio`：前 3% update 用于学习率 warmup。
- `lr_scheduler_type`：warmup 后的学习率曲线；`constant` 表示保持不变。
- `adam_beta1`、`adam_beta2`、`adam_epsilon`、`weight_decay`：AdamW 参数，当前沿用 Inkling。
- `kl_penalty_coef`：参考策略 KL 惩罚系数，对应 ms-swift `beta`。`0.0` 表示不加显式 KL penalty。
- `loss_type`：必须为 `grpo`。ms-swift 在 GRPO advantage 上使用带 clipping 的 policy loss。
- `reward_normalization`：`grpo_no_std` 表示组内 reward 减去均值，但不除以标准差；对应 ms-swift `scale_rewards=none`。
- `ppo_clip_low`、`ppo_clip_high`：policy ratio 的低侧和高侧 clip，当前是 0.20/0.28。
- `seed`：训练数据 shuffle、采样器等训练随机过程的 seed。它和环境 reset 使用的 `fixed_pool_seed` 是两个概念。

## LoRA

- `lora_rank`：LoRA 秩。越大可训练容量和显存/通信开销越高；当前 32。
- `lora_alpha`：LoRA 缩放参数；当前 64，即 rank 的两倍。

基础模型参数由 FSDP2 分片，optimizer 只更新 LoRA 参数。

## Actor 生成

- `max_turns`：每个环境最多交互轮数，当前 25。
- `max_context_length`：prompt、历史消息和当前生成合计的最大上下文，当前 8192。
- `actor_max_tokens`：每一个 actor turn 最多新生成多少 token，当前 1024。它不是整个 episode 的总 token 上限。
- `enable_thinking`：是否启用 Qwen thinking，当前关闭。
- `preserve_thinking`：模板是否保留已有 thinking 内容；thinking 关闭时通常不会产生实际影响。
- `actor_temperature`：采样温度；越高，四条轨迹的差异通常越大。
- `actor_top_p`、`actor_top_k`：核采样和 top-k 截断。
- `overlong_filter`：过滤超过长度边界的生成，建议保持开启。

## vLLM 和 GPU

- `gpu_ids`：宿主机使用的 GPU 编号，当前八张卡。
- `vllm_tensor_parallel`：vLLM rollout 的 tensor parallel 数，当前 8；必须整除 GPU 数。27B BF16 按 4 卡分片时，权重占用已超过每卡 0.45 的预算，无法创建 KV cache；因此本配置使用八卡共同分片。
- `vllm_gpu_memory_utilization`：vLLM 权重与缓存的显存预算比例，当前 0.45，配合 TP=8 使用。预算过低会出现 `No available memory for the cache blocks`；不能对所有显存错误都直接降低该值，应先区分加载、权重同步和训练阶段。
- `move_model_batches`：训练模型向 vLLM 同步权重时的 decoder 分组数，当前 64（一层一组；embedding/lm_head 和视觉模块由 ms-swift 单独分组）。必须启用分组，避免 FSDP2 在每张卡上一次性重建整个 27B 模型导致显存 OOM。
- `accelerate.num_processes`：FSDP 训练进程数，必须等于 `gpu_ids` 数量。
- `mixed_precision: bf16`：训练精度。
- `fsdp_version: 2`：使用 FSDP2 分片 27B 基础模型。

`configs/ms_swift_fsdp2.json` 也会显式传给 ms-swift，让 Trainer 创建前的模型加载阶段启用 FSDP2：只有 rank 0 读取完整权重，再同步给其他 rank，避免八个进程各自复制一份 27B 模型导致宿主内存 OOM。激活显存优化使用 FSDP 原生 `activation_checkpointing`。

这些 Accelerate/FSDP 字段已经按 8×4090 配好，换节点或改变 GPU 数时才需要一起调整。

## 日志、保存和恢复

- `rollout_json_export`：记录生成内容和 reward，便于检查训练行为，但会增加磁盘写入。
- `wandb_enabled`：当前为 `true`，同时记录 TensorBoard 和 W&B。
- `wandb_mode`：`online` 实时上传；改成 `offline` 时训练容器保持断网，稍后再用 `wandb sync` 上传。
- `wandb_project`：W&B 项目名，当前配置为 `envduels`。
- `wandb_entity`：个人账号可保持 `null`；需要记录到团队时填写团队 entity。
- `wandb_run_name`：为 `null` 时自动使用本次 UTC 时间戳对应的唯一运行名。
- W&B 的 `experiment` config 会保存完整解析配置，包括固定池参数和推导出的 batch、step 数量；ms-swift 自己的 Trainer 参数也会同时保存。
- `save_every`：每多少个 optimizer update 保存一次。当前 1，最容易恢复但写盘频繁；正式长跑可以改为 5 或 10。
- `save_total_limit`：最多保留多少个最近 checkpoint，当前 3。
- `resume_from_checkpoint`：为 `null` 时新开训练；恢复时填写之前的 `checkpoint-N` 目录，会恢复 optimizer、scheduler 和 trainer state。
- `output_dir`：每次运行的时间戳目录写到这里。

## 常用命令

只检查和打印推导结果，不使用 GPU：

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --dry-run
```

跑一个 optimizer update 的完整链路测试：

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --smoke
```

按 JSON 正式训练：

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json
```

第一次使用时，只需在已经存在的基础镜像上增加一个很小的 W&B 和 Qwen processor Python 包层，不会重新编译 Torch 或 vLLM：

```bash
bash scripts/unified_runtime.sh build-wandb
read -rsp 'W&B API key: ' WANDB_API_KEY
export WANDB_API_KEY
echo
```

API key 只从宿主机环境透传，不会写进 JSON、`launch.json` 或训练日志。默认不会把模型 checkpoint 上传到 W&B。

临时覆盖 epoch，而不修改 JSON：

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --epochs 2
```
