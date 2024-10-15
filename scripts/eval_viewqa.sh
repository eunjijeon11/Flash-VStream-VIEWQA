#!/bin/bash

# set up python environment
#conda activate vstream

echo start eval VIEWQA
# define your openai info here


CUDA_VISIBLE_DEVICES=0 python -m flash_vstream.eval_video.eval_any_dataset_features \
        --model-path Flash-VStream-7b \
        --dataset viewqa \
        # --api_key $OPENAIKEY \
        # --api_base $OPENAIBASE \
        # --api_type $OPENAITYPE \
        # --api_version $OPENAIVERSION \
        >> vstream-7b-eval-viewqa.log 2>&1 

