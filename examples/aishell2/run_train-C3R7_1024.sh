#! /usr/bin/env bash

cd ../.. > /dev/null

# train model
# if you wish to resume from an exists model, uncomment --init_from_pretrained_model
export FLAGS_sync_nccl_allreduce=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u train.py \
--batch_size=16 \
--num_epoch=10 \
--num_conv_layers=3 \
--num_rnn_layers=7 \
--rnn_layer_size=1024 \
--num_iter_print=100 \
--save_epoch=1 \
--num_samples=999078 \
--learning_rate=5e-4 \
--max_duration=27.0 \
--min_duration=0.0 \
--test_off=False \
--use_sortagrad=True \
--use_gru=True \
--use_gpu=True \
--is_local=True \
--share_rnn_weights=False \
--train_manifest='data/aishell2/manifest.train' \
--dev_manifest='data/aishell2/manifest.dev' \
--mean_std_path='data/aishell2/mean_std.npz' \
--vocab_path='data/aishell2/vocab.txt' \
--output_model_dir='./checkpoints/aishell2-C3R71024' \
--augment_conf_path='conf/augmentation.config' \
--specgram_type='linear' \
--shuffle_method='batch_shuffle_clipped' \
#--init_from_pretrained_model="./checkpoints/aishell2-C3R71024/epoch_6" \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
