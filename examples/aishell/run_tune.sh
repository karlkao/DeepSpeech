#! /usr/bin/env bash

cd ../.. > /dev/null

# grid-search for hyper-parameters in language model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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
--mean_std_path='data/aishell/mean_std.npz' \
--vocab_path='data/aishell/vocab.txt' \
--model_path='checkpoints/aishell/step_final' \
--lang_model_path='models/lm/zh_giga.no_cna_cmn.prune01244.klm' \
--error_rate_type='cer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in tuning!"
    exit 1
fi


exit 0
