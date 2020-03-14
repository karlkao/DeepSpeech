#!/usr/bin/env bash

BATCH_SIZE_PER_GPU=64
MIN_DURATION=0.0
MAX_DURATION=27.0

join_by () {
	local IFS="$1"; shift; echo "$*";
}

for NUM_GPUS in 16 8 4 2 1
#for NUM_GPUS in 8 4 2 1
do
  DEVICES=$(join_by , $(seq 0 $(($NUM_GPUS-1))|sed 's/^3//g'))
  BATCH_SIZE=$(($BATCH_SIZE_PER_GPU))
  echo ${DEVICES}
  CUDA_VISIBLE_DEVICES=$DEVICES \
  python train.py \
  --batch_size=$BATCH_SIZE \
  --num_epoch=1 \
  --test_off=True \
  --train_manifest='data/aishell2/manifest.train' \
  --mean_std_path='data/aishell2/mean_std.npz' \
  --dev_manifest='data/aishell2/manifest.dev' \
  --vocab_path='data/aishell2/vocab.txt' \
  --output_model_dir='./checkpoints/profile' \
  --min_duration=$MIN_DURATION \
  --num_conv_layers=2 \
  --num_rnn_layers=3 \
  --num_samples=999078 \
  --rnn_layer_size=1024 \
  --use_gru=True \
  --use_gpu=True \
  --max_duration=$MAX_DURATION > tmp.log 2>&1

  if [ $? -ne 0 ];then
      exit 1
  fi

  cat tmp.log  | grep "Time" | awk '{print "GPU Num: " "'"$NUM_GPUS"'" "	Time: "$2}'

  rm tmp.log
done
