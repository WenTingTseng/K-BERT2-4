python3 -u Predict.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
	--train_path ./datasets/Taipei_FAQ/train.tsv \
    --dev_path ./datasets/Taipei_FAQ/dev.tsv \
    --test_path ./datasets/Taipei_FAQ/test.tsv \
    --epochs_num 10 --batch_size 8 --kg_name Vacant \
    --output_model_path ./outputs/kbert_Vacant_Result.bin \
    > outputs/kbert_Vacant_eval_Result.log &