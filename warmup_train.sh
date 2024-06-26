DATA_DIR=./data
MODEL_PATH=meta-llama/Llama-2-7b-hf
DATA_SEED=3
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

method=$1
batch_size=$2
PERCENTAGE=$3 # percentage of the full data to train, you can specify the training file you want to use in the script
NVAL=$4
subject=$5

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$method" "$batch_size" "$subject" "$NVAL"