# MT5 monolingual (english hence infoxlm)
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

torchrun \
   --nproc_per_node=$NUM_GPUS_PER_NODE \
   --nnodes=$NUM_NODES \
        train.py --train_path /wnc_8_languages_new/mT5/en/en_train.csv --val_path /wnc_8_languages_new/mT5/en/en_val.csv --tokenizer google/mt5-small --model google/mt5-small --is_mt5 1 --exp_name ddp-mT5-monolingual-baseline --save_dir /genmodels/ --num_epochs 1 --train_batch_size 2 --val_batch_size 2 --test_batch_size 2 --max_source_length 250 --max_target_length 250 --isTrial 0 --world_size $WORLD_SIZE --multilingual False --isTest 0 --wandb 1

#MT0 monolingual
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
        train.py --train_path /wnc_8_languages_new/mT5/en/en_train.csv --val_path /wnc_8_languages_new/mT5/en/en_val.csv --tokenizer bigscience/mt0-small --model bigscience/mt0-small --is_mt5 1 --exp_name ddp-mT0-monolingual-baseline --save_dir /genmodels/ --num_epochs 1 --train_batch_size 2 --val_batch_size 2 --test_batch_size 2 --max_source_length 250 --max_target_length 250 --isTrial 0 --world_size $WORLD_SIZE --multilingual False  --isTest 0 --wandb 1

#MT0 multilingual
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
        train.py --train_path /wnc_8_languages_new/mT5/all/all_train.csv --val_path /wnc_8_languages_new/mT5/all/all_val.csv --tokenizer bigscience/mt0-small --model bigscience/mt0-small --is_mt5 1 --exp_name ddp-mT0-multilingual-baseline --save_dir /genmodels/ --num_epochs 1 --train_batch_size 2 --val_batch_size 2 --test_batch_size 2 --max_source_length 250 --max_target_length 250 --isTrial 0 --world_size $WORLD_SIZE --multilingual True --isTest 0 --wandb 1


#IndicBART monolingual
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
        train.py --train_path /wnc_8_languages_new/IndicBART/en/en_train.csv --val_path /wnc_8_languages_new/IndicBART/en/en_val.csv --tokenizer ai4bharat/IndicBART --model ai4bharat/IndicBART --is_mt5 0 --exp_name ddp-bart-monolingual-baseline --save_dir /genmodels/ --num_epochs 1 --train_batch_size 2 --val_batch_size 2 --test_batch_size 2 --max_source_length 250 --max_target_length 250 --isTrial 0  --world_size $WORLD_SIZE --multilingual False --isTest 0 --wandb 1


#IndicBART multilingual
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
        train.py --train_path /wikibias_8_languages_new/IndicBART/all/all_train.csv --val_path /wikibias_8_languages_new/IndicBART/all/all_val.csv --tokenizer ai4bharat/IndicBART --model ai4bharat/IndicBART --is_mt5 0 --exp_name ddp-bart-multilingual-baseline --save_dir /genmodels/ --num_epochs 1 --train_batch_size 2 --val_batch_size 2 --test_batch_size 2 --max_source_length 250 --max_target_length 250 --isTrial 0  --world_size $WORLD_SIZE --multilingual True --isTest 0 --wandb 1