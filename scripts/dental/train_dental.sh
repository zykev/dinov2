#!/bin/sh
set -e

# DINOv2 牙科 SSL 预训练（病例级 split，计划 §9.5.2）
# 用法（先 conda activate intraoral）:
#   ./scripts/dental/train_dental.sh                                   # 官方 vitb14 权重 continued SSL 预训练
#   ./scripts/dental/train_dental.sh --no-resume                       # 忽略已有 checkpoint 从头训练
#   ./scripts/dental/train_dental.sh train.OFFICIAL_EPOCH_LENGTH=500   # 透传其他 config 参数
#
# 可覆盖环境变量: GPUS(逗号分隔GPU号) DATA_ROOT SPLIT_JSON OUTPUT_DIR INIT_CKPT TRAIN_RECORDS BATCH_SIZE_PER_GPU
#
# 注意:
#   - 官方 dinov2_vitb14_pretrain.pth 是 plain state_dict，而 SSLMetaArch 与 fvcore
#     Checkpointer 都要求 {"model": ...} 包装；脚本调用 wrap_checkpoint.py 生成包装
#     sidecar（{OUTPUT_DIR}/init_checkpoint_wrapped.pth，已存在则跳过）并注入
#     student.pretrained_weights=<wrapped>。
#   - OFFICIAL_EPOCH_LENGTH 按 train records / (batch_size_per_gpu * NUM_GPUS) 自动计算。

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 定位目录（本脚本位于 baselines/dinov2/scripts/dental/ 下）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DINO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/../.." && pwd)"

# GPU 列表（逗号分隔，如 "2,3,4,5,6,7"）
GPUS="${GPUS:-2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NUM_GPUS="$(echo "${GPUS}" | awk -F',' '{print NF}')"

DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/.datasets/intraoral}"
SPLIT_JSON="${SPLIT_JSON:-${WORKSPACE_ROOT}/.datasets/intraoral/train_test_seed0_test0p2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${DINO_ROOT}/exp/dino_dental_v1}"
INIT_CKPT="${INIT_CKPT:-${WORKSPACE_ROOT}/.checkpoints/intraoral/dinov2_vitb14_pretrain.pth}"
BATCH_SIZE_PER_GPU=32

# TRAIN_RECORDS = split_json 中 annotations.train 总数（full+tooth+sextant+group）
TRAIN_RECORDS="${TRAIN_RECORDS:-68606}"
OFFICIAL_EPOCH_LENGTH=$((TRAIN_RECORDS / (BATCH_SIZE_PER_GPU * NUM_GPUS)))
[ "${OFFICIAL_EPOCH_LENGTH}" -lt 1 ] && OFFICIAL_EPOCH_LENGTH=1

# 官方 dinov2_vitb14_pretrain.pth 是 plain state_dict，SSLMetaArch / fvcore
# Checkpointer 要求 {"model": ...} 包装；wrap_checkpoint.py 负责包装并插值 pos_embed
# （官方 518px 的 37x37 -> 224px 的 16x16，bicubic）。首次运行生成 sidecar，之后跳过。
WRAPPED_CKPT="${OUTPUT_DIR}/init_checkpoint_wrapped.pth"

# 官方权重必须存在，缺失直接报错（不做 from-scratch 兜底）
[ -f "${INIT_CKPT}" ] || { echo "[train_dental] ERROR: INIT_CKPT not found (${INIT_CKPT})" >&2; exit 1; }

if [ ! -f "${WRAPPED_CKPT}" ]; then
    mkdir -p "$(dirname "${WRAPPED_CKPT}")"
    echo "[train_dental] wrapping official checkpoint -> ${WRAPPED_CKPT}"
    python "${SCRIPT_DIR}/wrap_checkpoint.py" \
        --src "${INIT_CKPT}" --dst "${WRAPPED_CKPT}" \
        --global-crops-size "${GLOBAL_CROPS_SIZE:-224}" --patch-size "${PATCH_SIZE:-14}"
fi
PRETRAINED_OPTS="student.pretrained_weights=${WRAPPED_CKPT}"

# 额外透传参数（如 --no-resume、train.OFFICIAL_EPOCH_LENGTH=...）
EXTRA_ARGS="$@"

cd "${DINO_ROOT}"
echo "[train_dental] GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[train_dental] OUTPUT_DIR=${OUTPUT_DIR} OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH}"

PYTHONPATH=. torchrun --nproc_per_node="${NUM_GPUS}" \
    dinov2/train/train.py \
    --config-file dinov2/configs/train/dental_vits14.yaml \
    --output-dir "${OUTPUT_DIR}" \
    train.dataset_path="DentalSSLData:root=${DATA_ROOT}:split_json=${SPLIT_JSON}:split=TRAIN:categories=full,tooth,sextant,group" \
    train.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
    train.OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH}" \
    ${PRETRAINED_OPTS} \
    ${EXTRA_ARGS}
