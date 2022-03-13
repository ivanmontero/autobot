# export GLUE_DIR=data/glue/
# export TASK_NAME=SST-2

# DATA_DIR=data/bert_25m_roberta/
DATA_DIR=data/bert_roberta_large/
# bert_roberta_masked_freeze_bert5
MODEL_DIR=savedir/fairseq/bert_roberta_large_masked_freeze_bert_ft1_100k/model/
# MODEL_DIR=savedir/fairseq/bert_roberta_large_masked_freeze_bert10_100k/model/
# MODEL_DIR=savedir/fairseq/bert_roberta_masked_freeze_bert10/model/
# MODEL_DIR=savedir/fairseq/bert_bert_mlm/model/
# MODEL_DIR=savedir/fairseq/bert_roberta_mlm/model/
# MODEL_DIR=savedir/fairseq/first/bert_roberta_mlm/model/

cp ${DATA_DIR}prepared/dict.en.txt.json $MODEL_DIR
#
#if [ ! -d "$GLUE_DIR" ]; then
#    python download_glue_data.py --data_dir $GLUE_DIR --tasks all
#fi

python sbert/sbert_nli.py \
  --model_path $MODEL_DIR \
  --save_path savedir/sbert/nli2_roberta_large_mafr10_100k_e1_ft1/ \
  --epochs 1 \
  --seed 42 \
  --lr 2e-5 \
  --weight_decay 0.1 \
  --valid_freq 500 \
  --batch_size 16
#python run_glue.py \
#  --model_name_or_path $MODEL_DIR \
#  --task_name $TASK_NAME \
#  --do_train \
#  --do_eval \
#  --data_dir $GLUE_DIR/$TASK_NAME \
#  --max_seq_length 128 \
#  --per_device_train_batch_size 64 \
#  --learning_rate 3e-5 \
#  --weight_decay 0.1 \
#  --warmup_steps 379 \
#  --save_steps 500 \
#  --eval_steps 500 \
#  --logging_steps 500 \
#  --save_total_limit 1 \
#  --num_train_epochs 6.0 \
#  --evaluate_during_training \
#  --output_dir savedir/glue/sst23-mlm1-e6-b64/ \
#  --logging_dir savedir/glue/sst23-mlm1-e6-b64/logs/

#parser.add_argument("--model_path", type=str, default="bert-base-uncased")
#parser.add_argument("--save_path", type=str, default="savedir/sbert/test/")
#parser.add_argument("--nhead_bottleneck", type=int, default=None)
#parser.add_argument("--epochs", type=int, default=1)
#parser.add_argument("--seed", type=int, default=42)
#parser.add_argument("--batch_size", type=int, default=16)
#parser.add_argument("--mode", type=str, default="train_and_evaluate", choices=["train_and_evaluate", "train", "evaluate"])
