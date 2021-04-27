CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/Taipei_FAQ/train.tsv \
    --dev_path ./datasets/Taipei_FAQ/dev.tsv \
    --test_path ./datasets/Taipei_FAQ/test.tsv \
    --epochs_num 10 --batch_size 8 --kg_name Vacant \
    --output_model_path ./outputs/kbert_Vacant_Result.bin \
    > outputs/kbert_Vacant_Result.log &
