#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=en    # language direction (e.g 'en' for VSR task / 'en-es' for En to Es VST task)

# set paths
ROOT=/mnt/sdb/yuran/av_hubert_llm/VSP-LLM
MODEL_SRC=${ROOT}/src
LLM_PATH=${ROOT}/checkpoints/Llama-2-7b-hf   # path to llama checkpoint
DATA_ROOT=/mnt/sdb/yuran/av_hubert/datasets/multivsr/lrs3_format/data   # path to test dataset dir

MODEL_PATH=/mnt/sdb/yuran/av_hubert/datasets/multivsr/outs_qwen_visual_no_detach/output_finetune/checkpoints/checkpoint_best.pt
OUT_PATH=/mnt/sdb/yuran/av_hubert/datasets/multivsr/outs_qwen_visual_no_detach/output_infer

# fix variables based on langauge
if [[ $LANG == *"-"* ]] ; then
    TASK="vst"
    IFS='-' read -r SRC TGT <<< ${LANG}
    USE_BLEU=true
    DATA_PATH=${DATA_ROOT}/${TASK}/${SRC}/${TGT}

else
    TASK="vsr"
    TGT=${LANG}
    USE_BLEU=false
    DATA_PATH=${DATA_ROOT}
fi

# start decoding
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="4"
python -B ${MODEL_SRC}/vsp_llm_decode.py \
    --config-dir ${MODEL_SRC}/conf \
    --config-name s2s_decode \
        common.user_dir=${MODEL_SRC} \
        dataset.gen_subset=test \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        generation.beam=20 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH}/${TASK}/${LANG}