

TARGET_TASK_NAME="tydiqa"
PERCENTAGE=0.05
TRAIN_FILES=./data/selected_data/${TARGET_TASK_NAME}tydiqa_adam_sim_trainp${PERCENTAGE}_seed3_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 