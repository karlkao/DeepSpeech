#! /usr/bin/env bash

cd ../.. > /dev/null

# grid-search for hyper-parameters in language model
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7,8,9,10,11,12,13,14,15 \
python -u tools/tune.py \
--num_batches=-1 \
--batch_size=256 \
--beam_size=500 \
--num_proc_bsearch=12 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=1024 \
--num_alphas=20 \
--num_betas=20 \
--alpha_from=0.0 \
--alpha_to=10.0 \
--beta_from=0.0 \
--beta_to=10.0 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=True \
--share_rnn_weights=False \
--tune_manifest='data/Wang/16k/manifest.dev' \
--mean_std_path='data/aishell2/mean_std.npz' \
--vocab_path='data/aishell2/vocab.txt' \
--model_path='checkpoints/aishell2-C2R31024/step_final' \
--lang_model_path='models/lm/zhidao_giga.klm' \
--error_rate_type='cer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in tuning!"
    exit 1
fi


exit 0
