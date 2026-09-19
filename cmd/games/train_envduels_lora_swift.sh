#!/usr/bin/env bash
# Qwen3.8-27B LoRA GRPO through ms-swift's multi-turn Gym rollout interface.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Verify real kernel limits before importing Torch or allocating model memory.
python3 scripts/memory_guard.py "${SPADE_MEMORY_LIMIT_GIB:?Use scripts/run_train.py for protected training}"

MODEL="${MODEL:-$PROJECT_ROOT/checkpoints/Qwen3.8-27B}"
DATASET="${DATASET:-$PROJECT_ROOT/data/envduels/fixed90-swift.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/qwen38-envduels-lora}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:?ACCELERATE_CONFIG must be set by scripts/run_train.py}"
FSDP_CONFIG="${FSDP_CONFIG:-$PROJECT_ROOT/configs/ms_swift_fsdp2.json}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-24}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
PRESERVE_THINKING="${PRESERVE_THINKING:-true}"
SCALE_REWARDS="${SCALE_REWARDS:-none}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"
DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-false}"
OVERLONG_FILTER="${OVERLONG_FILTER:-true}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"
MAX_TURNS="${MAX_TURNS:-25}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
VLLM_TP="${VLLM_TP:-8}"
MOVE_MODEL_BATCHES="${MOVE_MODEL_BATCHES:-64}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RUN_NAME="${RUN_NAME:-envduels-grpo}"

for name in NUM_GPUS NUM_GENERATIONS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS GENERATION_BATCH_SIZE NUM_ITERATIONS MAX_TURNS MAX_LENGTH MAX_COMPLETION_LENGTH VLLM_TP MOVE_MODEL_BATCHES LORA_RANK LORA_ALPHA; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
[[ "$DATASET_SHUFFLE" == true || "$DATASET_SHUFFLE" == false ]] || { echo 'DATASET_SHUFFLE must be true or false' >&2; exit 2; }
for name in ENABLE_THINKING PRESERVE_THINKING DYNAMIC_SAMPLE OVERLONG_FILTER LOG_COMPLETIONS; do
    value="${!name}"
    [[ "$value" == true || "$value" == false ]] || { echo "$name must be true or false" >&2; exit 2; }
done
IFS=',' read -r -a REPORT_TO_ARGS <<< "$REPORT_TO"
(( ${#REPORT_TO_ARGS[@]} > 0 )) || { echo 'REPORT_TO must not be empty' >&2; exit 2; }
for reporter in "${REPORT_TO_ARGS[@]}"; do
    [[ "$reporter" == tensorboard || "$reporter" == wandb ]] || {
        echo "Unsupported report target: $reporter" >&2
        exit 2
    }
done
[[ -n "$RUN_NAME" ]] || { echo 'RUN_NAME must not be empty' >&2; exit 2; }
TRAIN_DURATION_ARGS=()
if [[ -n "${MAX_STEPS:-}" && -z "${NUM_TRAIN_EPOCHS:-}" ]]; then
    [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_STEPS must be a positive integer' >&2; exit 2; }
    TRAIN_DURATION_ARGS=(--max_steps "$MAX_STEPS")
elif [[ -z "${MAX_STEPS:-}" && -n "${NUM_TRAIN_EPOCHS:-}" ]]; then
    TRAIN_DURATION_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
else
    echo 'Set exactly one of MAX_STEPS and NUM_TRAIN_EPOCHS' >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    [[ -d "$RESUME_FROM_CHECKPOINT" ]] || { echo "Missing resume checkpoint: $RESUME_FROM_CHECKPOINT" >&2; exit 2; }
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
(( NUM_GPUS % VLLM_TP == 0 )) || { echo 'NUM_GPUS must be divisible by VLLM_TP' >&2; exit 2; }
(( MAX_LENGTH > MAX_COMPLETION_LENGTH + 64 )) || { echo 'MAX_LENGTH is too small' >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "Missing HF checkpoint: $MODEL" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 2; }
[[ -f "$ACCELERATE_CONFIG" ]] || { echo "Missing Accelerate config: $ACCELERATE_CONFIG" >&2; exit 2; }
[[ -f "$FSDP_CONFIG" ]] || { echo "Missing ms-swift FSDP config: $FSDP_CONFIG" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR already exists: $OUTPUT_DIR" >&2; exit 2; }

exec accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    /opt/ms-swift/swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --tuner_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --fsdp "$FSDP_CONFIG" \
    --dataset "$DATASET" \
    --dataset_shuffle "$DATASET_SHUFFLE" \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --external_plugins spade/swift_backend/fsdp_ram_loader.py spade/swift_backend/envduels_gym.py spade/swift_backend/wandb_config.py \
    --callbacks envduels_wandb_config \
    --multi_turn_scheduler gym_scheduler \
    --gym_env envduels \
    --use_gym_env true \
    --max_turns "$MAX_TURNS" \
    --num_generations "$NUM_GENERATIONS" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --num_iterations "$NUM_ITERATIONS" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size "$VLLM_TP" \
    --move_model_batches "$MOVE_MODEL_BATCHES" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.45}" \
    --vllm_max_model_len "$MAX_LENGTH" \
    --vllm_enforce_eager true \
    --enable_thinking "$ENABLE_THINKING" \
    --preserve_thinking "$PRESERVE_THINKING" \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "${LEARNING_RATE:-1e-6}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant}" \
    --weight_decay "${WEIGHT_DECAY:-0.1}" \
    --adam_beta1 "${ADAM_BETA1:-0.9}" \
    --adam_beta2 "${ADAM_BETA2:-0.98}" \
    --adam_epsilon "${ADAM_EPSILON:-1e-12}" \
    --seed "${TRAIN_SEED:-42}" \
    "${TRAIN_DURATION_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --save_steps "${SAVE_STEPS:-1}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --logging_steps 1 \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --temperature "${TEMPERATURE:-0.6}" \
    --top_p "${TOP_P:-0.95}" \
    --top_k "${TOP_K:-20}" \
    --beta "${BETA:-0.0}" \
    --loss_type "$LOSS_TYPE" \
    --scale_rewards "$SCALE_REWARDS" \
    --epsilon "${PPO_CLIP_LOW:-0.2}" \
    --epsilon_high "${PPO_CLIP_HIGH:-0.28}" \
    --dynamic_sample "$DYNAMIC_SAMPLE" \
    --max_resample_times "${MAX_RESAMPLE_TIMES:-3}" \
    --overlong_filter "$OVERLONG_FILTER" \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --log_completions "$LOG_COMPLETIONS" \
    --report_to "${REPORT_TO_ARGS[@]}" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR"
