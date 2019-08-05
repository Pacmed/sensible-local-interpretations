export GLUE_DIR=/scratch/users/vision/data/nlp/glue
export TASK_NAME=SST-2

python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir /scratch/users/vision/chandan/pacmed/glue/$TASK_NAME-1epoch/
#     --max_steps 1000 \    
