export CUDA_VISIBLE_DEVICES=7

current_date=$(date +%Y%m%d)
MODEL_NAME='Mini-Monkey-InternVL2-26B-Full'
DATASET_NAME='MMMU_DEV_VAL'
OUTPUT_DIR='./work-dir/MiniMonkey-no-scm/'${current_date}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python run.py --data ${DATASET_NAME} \
              --model ${MODEL_NAME} \
              --verbose \
              --mode all \
              --work-dir ${OUTPUT_DIR} \
              --rerun