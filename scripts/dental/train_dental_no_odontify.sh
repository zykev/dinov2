#!/bin/sh
set -e

# DINOv2 牙科 SSL 预训练 —— no-odontify 病例级 split
#（自有采集者 + AIDC 纳入训练；odontify 外部集整体排除）
#
# 独立脚本：自带全部训练超参与 GPU / 端口配置，直接调 torchrun。
#
# 用法（先 conda activate intraoral；从任意目录运行也可，路径自动解析）:
#   ./scripts/dental/train_dental_no_odontify.sh
#   ./scripts/dental/train_dental_no_odontify.sh --no-resume
#   EPOCHS=80 ./scripts/dental/train_dental_no_odontify.sh   # 改总训练轮数（默认 50）
#   ./scripts/dental/train_dental_no_odontify.sh train.OFFICIAL_EPOCH_LENGTH=500   # 透传其他 config 参数
#
# 可覆盖环境变量:
#   GPUS        默认 6 卡 GPU 2-7（分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀）
#               GPUS="2 3" / GPUS="2,3" / GPUS="cuda:2 cuda:3" 三者等价
#   DATA_ROOT   默认 <root>/.datasets/intraoral（挂载点缺失时回退 intraoral1）
#   SPLIT_JSON  默认 .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   OUTPUT_DIR  默认 baselines/dinov2/exp/dino_dental_small_448_no_odontify
#   INIT_CKPT   默认 .checkpoints/intraoral/dinov2_vits14_pretrain.pth
#   TRAIN_RECORDS    默认由 count_train_records.py 从当前 split 现算（失败回退 68606）
#   MASTER_PORT      默认 29534（torchrun 默认 29500 常被其它任务占用）
#   BATCH_SIZE_PER_GPU  默认 8
#   GLOBAL_CROPS_SIZE   默认 448
#   LOCAL_CROPS_SIZE    默认 98
#   EPOCHS           默认 50（注入 optim.epochs，覆盖 config dental_vits14.yaml 的 200）
#
# 注意:
#   - 骨架：官方 vits14 = ViT-Small（embed 384 / depth 12 / heads 6），与 config
#     dinov2/configs/train/dental_vits14.yaml 的 student.arch=vit_small 一致。
#     要回到 vitb14：把 INIT_CKPT/OUTPUT_DIR 换回 dinov2_vitb14_pretrain.pth /
#     exp/dino_dental_448，并把该 config 的 student.arch 改回 vit_base。
#   - 官方 dinov2_vits14_pretrain.pth 是 plain state_dict，而 SSLMetaArch 与 fvcore
#     Checkpointer 都要求 {"model": ...} 包装；脚本调用 wrap_checkpoint.py 生成包装
#     sidecar（${OUTPUT_DIR}/init_checkpoint_wrapped.pth，已存在则跳过）并注入
#     student.pretrained_weights=<wrapped>。
#   - 分辨率：默认 GLOBAL_CROPS_SIZE=448，与 ours full_canvas=448 统一，vits14 的
#     dense 特征网格为 32x32（448/14）；wrap_checkpoint.py 会把官方 37x37（518px）
#     pos_embed 双三次插值到 32x32。模型 pos_embed 网格由 cfg.crops.global_crops_size
#     决定（dinov2/models/__init__.py::build_model(img_size=cfg.crops.global_crops_size)），
#     因此 config 的 global_crops_size 必须与 wrap 目标一致。local_crops_size 保持 98
#     （14 的整数倍），local 分支在 forward 内动态插值 pos_embed，无需改动。
#   - OFFICIAL_EPOCH_LENGTH 按 train records / (batch_size_per_gpu * NUM_GPUS) 自动计算，
#     它是「每个 epoch 的 iteration 数」而非轮数；总迭代数 = optim.epochs × OFFICIAL_EPOCH_LENGTH。
#     optim.epochs 由本脚本按 EPOCHS（默认 50）注入。
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 定位目录（脚本位于 baselines/dinov2/scripts/dental/ 下）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DINO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${DINO_ROOT}/../.." && pwd)"

# 指定用于训练的 GPU：分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀
#   GPUS="2 3"   GPUS="2,3"   GPUS="cuda:2 cuda:3"   GPUS="2, 3"
GPUS="${GPUS:-2,3,4,5,6,7}"

# 解析 GPUS -> torchrun 用的 CUDA_VISIBLE_DEVICES 列表与卡数 NUM_GPUS
# （NUM_GPUS 还参与下面 OFFICIAL_EPOCH_LENGTH 的换算）。
# set -f：解析期间关闭通配符展开（GPUS 里误写 * ? [ 时不会被 glob 成文件名）。
CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
set -f
for GPU in $(printf '%s\n' "${GPUS}" | tr ',;' '  '); do
    GPU_ID=${GPU#cuda:}
    case "${GPU_ID}" in
        ''|*[!0-9]*)
            echo "[train_dental_no_odontify] ERROR: GPUS 中的 '${GPU}' 不是合法 GPU 序号" >&2
            echo "  合法写法: GPUS=\"2 3\" / GPUS=\"2,3\" / GPUS=\"cuda:2 cuda:3\"" >&2
            exit 1
            ;;
    esac
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES},${GPU_ID}
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done
set +f
if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "[train_dental_no_odontify] ERROR: GPUS 为空，应形如 GPUS=\"2 3\" 或 GPUS=\"2,3\"" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES

# 分布式 rendezvous 端口：避开常被占用的 29500
MASTER_PORT="${MASTER_PORT:-29534}"

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。显式设置 DATA_ROOT / SPLIT_JSON 时以环境变量为准，不回退。
resolve_dental_path() {
    # $1 = 相对 .datasets 层名的后缀（"" 或 "/train_test_seed0_test0p2_no_odontify.json"）
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "/train_test_seed0_test0p2_no_odontify.json")}"
# 输出目录：no-odontify 用独立目录，既不与常规 split 的 vits14 产物互相覆盖，
# 也不覆盖 vitb14 产物 exp/dino_dental_448（后者是 compare/ 三份 manifest 的引用对象）。
# 注意 init_checkpoint_wrapped.pth（pos_embed 插值 sidecar）也随 OUTPUT_DIR 走，新目录会重新生成一份。
OUTPUT_DIR="${OUTPUT_DIR:-${DINO_ROOT}/exp/dino_dental_small_448_no_odontify}"
# 初始化权重：官方 vits14（ViT-Small，plain state_dict；对应 config 的 student.arch=vit_small）
INIT_CKPT="${INIT_CKPT:-${WORKSPACE_ROOT}/.checkpoints/intraoral/dinov2_vits14_pretrain.pth}"
# 448 下 global crop 为 32x32 tokens（224 时 16x16），attention 开销约 16 倍，batch 调小
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
GLOBAL_CROPS_SIZE="${GLOBAL_CROPS_SIZE:-448}"
# 总训练轮数：注入 optim.epochs 覆盖 config dental_vits14.yaml 里的 200
# （由 100 减半为 50，与 mae / swin-mae / simclrv2 脚本保持一致）
EPOCHS="${EPOCHS:-50}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[train_dental_no_odontify] ERROR: split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

# TRAIN_RECORDS = split_json 中 train 分区展开后的 records 总数（full+tooth+sextant+group）。
# 直接由 count_train_records.py 从 (DATA_ROOT, SPLIT_JSON) 现算（复用数据集本身的 build_records，
# 因此换 split 时无需手改，也不会与 dataloader 实际遍历的样本数漂移）；显式设置 TRAIN_RECORDS
# 时以环境变量为准。若计算失败（依赖缺失等），回退到历史「仅自有采集者」split 的值 68606 并告警。
if [ -z "${TRAIN_RECORDS:-}" ]; then
    TRAIN_RECORDS="$(python "${SCRIPT_DIR}/count_train_records.py" \
        --root "${DATA_ROOT}" --split_json "${SPLIT_JSON}" \
        --split train --categories "full,tooth,sextant,group" 2>/dev/null || true)"
fi
if [ -z "${TRAIN_RECORDS:-}" ]; then
    TRAIN_RECORDS=68606
    echo "[train_dental_no_odontify] WARNING: 无法从 ${SPLIT_JSON} 推导 TRAIN_RECORDS，回退到历史值 ${TRAIN_RECORDS}。" \
         "若 split 已变，OFFICIAL_EPOCH_LENGTH 会偏小，请显式设置 TRAIN_RECORDS。" >&2
fi
echo "[train_dental_no_odontify] TRAIN_RECORDS=${TRAIN_RECORDS} (derived from $(basename "${SPLIT_JSON}"))"
OFFICIAL_EPOCH_LENGTH=$((TRAIN_RECORDS / (BATCH_SIZE_PER_GPU * NUM_GPUS)))
# 用 if 而非 `[ ... ] && ...`：后者在条件为假时会让 AND-OR list 以非零退出，
# 在 set -e 下是有风险的写法（当前 bash 不退出，但语义上不值得依赖）。
if [ "${OFFICIAL_EPOCH_LENGTH}" -lt 1 ]; then
    OFFICIAL_EPOCH_LENGTH=1
fi

# 官方 dinov2_vits14_pretrain.pth 是 plain state_dict，SSLMetaArch / fvcore
# Checkpointer 要求 {"model": ...} 包装；wrap_checkpoint.py 负责包装并插值 pos_embed
# （官方 518px 的 37x37 -> GLOBAL_CROPS_SIZE 网格，bicubic）。首次运行生成 sidecar，
# 之后跳过（若改 GLOBAL_CROPS_SIZE 需删除旧 sidecar 重新生成）。
WRAPPED_CKPT="${OUTPUT_DIR}/init_checkpoint_wrapped.pth"

# 官方权重必须存在，缺失直接报错（不做 from-scratch 兜底）
[ -f "${INIT_CKPT}" ] || { echo "[train_dental_no_odontify] ERROR: INIT_CKPT not found (${INIT_CKPT})" >&2; exit 1; }

if [ ! -f "${WRAPPED_CKPT}" ]; then
    mkdir -p "$(dirname "${WRAPPED_CKPT}")"
    echo "[train_dental_no_odontify] wrapping official checkpoint -> ${WRAPPED_CKPT}"
    python "${SCRIPT_DIR}/wrap_checkpoint.py" \
        --src "${INIT_CKPT}" --dst "${WRAPPED_CKPT}" \
        --global-crops-size "${GLOBAL_CROPS_SIZE}" --patch-size "${PATCH_SIZE:-14}"
fi
PRETRAINED_OPTS="student.pretrained_weights=${WRAPPED_CKPT}"

# 额外透传参数（如 --no-resume、train.OFFICIAL_EPOCH_LENGTH=...）
EXTRA_ARGS="$@"

cd "${DINO_ROOT}"
echo "[train_dental_no_odontify] GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS} MASTER_PORT=${MASTER_PORT}"
echo "[train_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR} EPOCHS=${EPOCHS} OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH}"

PYTHONPATH=. torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    dinov2/train/train.py \
    --config-file dinov2/configs/train/dental_vits14.yaml \
    --output-dir "${OUTPUT_DIR}" \
    train.dataset_path="DentalSSLData:root=${DATA_ROOT}:split_json=${SPLIT_JSON}:split=TRAIN:categories=full,tooth,sextant,group" \
    train.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
    train.OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH}" \
    optim.epochs="${EPOCHS}" \
    ${PRETRAINED_OPTS} \
    ${EXTRA_ARGS}
