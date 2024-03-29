#!/bin/bash
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
export DATA_DIR=$1
NOISE_DIR=$2
PATH_CHECKPOINT=$3
PLATFORM=$4

if [ $PLATFORM = "GPU" ]; then
    export CUDA_VISIBLE_DEVICES='0'
fi

python eval.py \
    --platform=$PLATFORM \
    --dataset_path=$DATA_DIR \
    --noise_path=$NOISE_DIR \
    --ckpt_path=$PATH_CHECKPOINT > log.txt 2>&1 &
