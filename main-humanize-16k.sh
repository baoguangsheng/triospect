#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

model=Qwen3-4B
maxtoks=200
echo "Model: $model"

data_path=./data/humanize-16k/${model}-${maxtoks}tokens
result_path=./exp_test/humanize-16k/${model}-${maxtoks}tokens
mkdir -p $data_path $result_path

datasets='essay.dev,essay.test,arxiv.dev,arxiv.test,writing.dev,writing.test,news.dev,news.test'

# textual transformation to produce content-preserving and expression-preserving texts
# for prompt in simplify replace; do
#   python scripts/feature_preserver.py --data_path $data_path --datasets $datasets --prompt $prompt
# done

# eval detectors
detectors='roberta,radar,log_perplexity,log_rank,lrr,fast_detect,triospect(fast_detect),binoculars,triospect(binoculars),imbd,triospect(imbd)'

python scripts/delegate_detector.py --data_path $data_path --result_path $result_path \
              --datasets $datasets --detectors $detectors --categorize detect
