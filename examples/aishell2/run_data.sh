#! /usr/bin/env bash

cd ../.. > /dev/null

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python data/aishell2/aishell2.py \
--manifest_prefix='data/aishell2/manifest' \
--target_dir='./dataset/aishell2'

if [ $? -ne 0 ]; then
    echo "Prepare aishell2 failed. Terminated."
    exit 1
fi


# build vocabulary
python tools/build_vocab.py \
--count_threshold=0 \
--vocab_path='data/aishell2/vocab.txt' \
--manifest_paths 'data/aishell2/manifest.train' 'data/aishell2/manifest.dev'

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python tools/compute_mean_std.py \
--manifest_path='data/aishell2/manifest.train' \
--num_samples=2000 \
--specgram_type='linear' \
--output_path='data/aishell2/mean_std.npz'

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "Aishell data preparation done."
exit 0
