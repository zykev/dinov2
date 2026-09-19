#!/bin/sh
set -e

# DINOv2 牙科 SSL 预训练 —— 新公开 split（自有采集者 + AIDC，排除 odontify）
#
# 本脚本是 scripts/dental/train_dental.sh 的「预设包装」：只预设下面 4 个环境变量，
# 训练超参（dental_vits14 config / GLOBAL_CROPS_SIZE 448 / batch 8 /
# categories full,tooth,sextant,group / 官方 vits14（ViT-Small）权重初始化 …）
# 全部沿用原脚本，不在两处重复声明，以免以后改超参时两边漂移。
#
#   1) SPLIT_JSON  -> <root>/.datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#                     新的病例级 split（把 AIDC 纳入训练、排除 odontify 外部集）
#   2) DATA_ROOT   -> <root>/.datasets/intraoral（intraoral -> intraoral1 回退不变）
#   3) OUTPUT_DIR  -> exp/dino_dental_small_448_no_odontify
#                     与换 split 前的两个 vits14/vitb14 产物目录都隔离：
#                     exp/dino_dental_448（vitb14）是 compare/ 各 run 的 manifest 引用对象，
#                     exp/dino_dental_small_448（原脚本默认）是同一权重的常规 split 产物，
#                     两者都不能覆盖。注意 init_checkpoint_wrapped.pth
#                     （pos_embed 插值 sidecar）也随 OUTPUT_DIR 走，新目录会重新生成一份
#   4) MASTER_PORT -> 29534（torchrun 的 --master_port 是 action=env，会读该环境变量；
#                     原脚本未显式传 --master_port，默认 29500 常被其它任务占用）
#
# TRAIN_RECORDS 无需设置：原脚本会调用 scripts/dental/count_train_records.py 从当前
# SPLIT_JSON 现算（换 split 后 OFFICIAL_EPOCH_LENGTH 自动跟随）。
#
# 用法（先 conda activate intraoral；从任意目录；参数原样透传给原脚本）:
#   bash baselines/dinov2/scripts/dental/train_dental_no_odontify.sh
#   bash baselines/dinov2/scripts/dental/train_dental_no_odontify.sh --no-resume
#
# 环境变量优先级仍然是「显式传入 > 本脚本预设」，例如:
#   GPUS=2,3 OUTPUT_DIR=/tmp/dino_try \
#       bash baselines/dinov2/scripts/dental/train_dental_no_odontify.sh
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)

# 目录定位（脚本位于 <root>/baselines/dinov2/scripts/dental/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DINO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/../.." && pwd)"
ORIGINAL_SCRIPT="${SCRIPT_DIR}/train_dental.sh"

# 新 split 的文件名（相对 .datasets/<collector-root> 的后缀）
NEW_SPLIT_SUFFIX="/train_test_seed0_test0p2_no_odontify.json"

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。与原脚本同一套规则。
resolve_dental_path() {
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

export DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
export SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "${NEW_SPLIT_SUFFIX}")}"
# 新 split 用独立输出目录，避免覆盖 compare/ manifest 还在引用的旧产物
# （exp/dino_dental_448 是 vitb14 产物，exp/dino_dental_small_448 是换 split 前的 vits14 产物）
export OUTPUT_DIR="${OUTPUT_DIR:-${DINO_ROOT}/exp/dino_dental_small_448_no_odontify}"
export MASTER_PORT="${MASTER_PORT:-29534}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[train_dental_no_odontify] ERROR: 新 split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

echo "[train_dental_no_odontify] DATA_ROOT=${DATA_ROOT}"
echo "[train_dental_no_odontify] SPLIT_JSON=${SPLIT_JSON}"
echo "[train_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train_dental_no_odontify] MASTER_PORT=${MASTER_PORT}"
echo "[train_dental_no_odontify] -> exec ${ORIGINAL_SCRIPT} $*"

exec sh "${ORIGINAL_SCRIPT}" "$@"
