#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
export CUDA_VISIBLE_DEVICES=4,5,6,7
DATA_PATH=/mnt/sdb/yuran/av_hubert/datasets/lrs2/raw/mvlrs_v1/big_data   # path to train dataset dir
OUT_PATH=/mnt/sdb/yuran/av_hubert/datasets/lrs2/outs_llm6/output_finetune_keywords   # output path to save 

ROOT=/mnt/sdb/yuran/av_hubert_llm/VSP-LLM
SRC=${ROOT}/src
LLM_PATH=${ROOT}/checkpoints/Llama-2-7b-hf   # path to llama checkpoint
PRETRAINED_MODEL_PATH=${ROOT}/checkpoints/large_vox_iter5.pt   # path to pretrained avhubert

# start training
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
fairseq-hydra-train \
    --config-dir ${SRC}/conf \
    --config-name vsp-llm-433h-finetune \
        common.user_dir=${SRC} \
        task.data=${DATA_PATH} \
        task.label_dir=${DATA_PATH} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=4 \
        distributed_training.nprocs_per_node=4