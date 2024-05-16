CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="people"
#

python run_ner_crf.py \
  --model_name_or_path=$BERT_BASE_DIR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --task_name=$TASK_NAME \
  --max_seq_length=128 \
  --n_context=3 \
  --per_gpu_train_batch_size=12 \
  --per_gpu_eval_batch_size=12 \
  --gradient_accumulation_steps 1 \
  --learning_rate=3e-5 \
  --crf_learning_rate=1e-3 \
  --linear_learning_rate=5e-3 \
  --num_train_epochs=5.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/k_3/ \
  --overwrite_output_dir \
  --seed=42
