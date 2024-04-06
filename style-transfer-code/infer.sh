#if there is more than one device in CUDA_VISIBLE_DEVICES, then set model_device to the specified gpu, else omit parameter.

# MT5
python test.py --test_path /wnc_8_languages_new/en/en_test.csv --checkpoint /genmodels/0.pt --tokenizer google/mt5-small --model google/mt5-small --save_dir /scratch/inferences --test_batch_size 2 --max_source_length 250 --max_target_length 250 --model_device cuda:0 --isTest 1 --isTrial 0 --is_mt5 1 --multilingual False --language English


#MT0
python test.py --test_path /wnc_8_languages_new/en/en_test.csv --checkpoint /genmodels/0.pt --tokenizer bigscience/mt0-small --model bigscience/mt0-small --save_dir /scratch/inferences --test_batch_size 2 --max_source_length 250 --max_target_length 250 --model_device cuda:1 --isTest 1 --isTrial 0 --is_mt5 1 --multilingual False --language English


# IndicBART
python test.py --test_path /wikibias_8_languages_new/IndicBART/bn/bn_test.csv --checkpoint /genmodels/0.pt --tokenizer ai4bharat/IndicBART --model ai4bharat/IndicBART --save_dir /scratch/inferences --test_batch_size 2 --max_source_length 250 --max_target_length 250 --model_device cuda:2 --isTest 1 --isTrial 0 --is_mt5 0 --multilingual True --language '<2bn>'
