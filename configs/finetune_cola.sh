export GLUE_DIR=data/glue/
export TASK_NAME=CoLA

DATA_DIR=data/bert_25m_roberta/
#MODEL_DIR=savedir/fairseq/bert_roberta_mlm/model/
# MODEL_DIR=savedir/fairseq/first/bert_roberta_mlm/model/
# MODEL_DIR=savedir/fairseq/bert_roberta_masked_freeze_bert/model/
# MODEL_DIR=savedir/fairseq/bert_roberta_normal_freeze_bert/model/
# MODEL_DIR=savedir/fairseq/bert_roberta_masked_freeze_bert10/model/
MODEL_DIR=savedir/fairseq/bert_roberta_masked_freeze_bert10_100k/model/

cp ${DATA_DIR}prepared/dict.en.txt.json $MODEL_DIR

if [ ! -d "$GLUE_DIR" ]; then
    python download_glue_data.py --data_dir $GLUE_DIR --tasks all
fi

python run_glue.py \
  --model_name_or_path $MODEL_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --weight_decay 0.1 \
  --warmup_steps 161 \
  --save_steps 50 \
  --eval_steps 50 \
  --logging_steps 50 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --metric_for_best_model mcc \
  --greater_is_better \
  --num_train_epochs 10.0 \
  --evaluate_during_training \
  --output_dir savedir/glue/cola3-mafr10-100k-e10-acc/ \
  --logging_dir savedir/glue/cola3-mafr10-100k-e10-acc/logs/
