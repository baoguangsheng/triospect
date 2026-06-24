# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import numpy as np
import os.path as path
from tqdm import tqdm
from model_proxy import ModelProxy, ResponseCache
from utils import load_json, save_json, full_lang
from utils import get_lang, validate_common, validate_repetition, validate_garbage


SYSTEM_PROMPT = 'You are a professional writer.'

TASK_PROMPT_EXTRACT_CONTENT_SUMMARIZE = \
'''Summarize the main ideas of the text in {lang} language in a clear and concise manner:
{text}
'''

TASK_PROMPT_EXTRACT_CONTENT_OUTLINE = \
'''Outline the main points of the text in {lang} language to get a clear and concise picture of the content:
{text}
'''

TASK_PROMPT_EXTRACT_CONTENT_SIMPLIFY = \
'''Simplify the text in {lang} language to make it clear and concise while preserving its meaning:
{text}
'''

TASK_PROMPT_EXTRACT_EXPRESSION_LIST = \
'''Identify and list the representative language expressions used in the text in {lang} language:
{text}
'''

TASK_PROMPT_EXTRACT_EXPRESSION_SUBSTITUTE = \
'''Substitute the key ideas in the text in {lang} language with placeholders, ensuring the original tone, style, and language remain intact:
{text}
'''

TASK_PROMPT_EXTRACT_LANGUAGE_REPLACE = \
'''Replace the main points of the text in {lang} language with a generic topic while preserving the language expression:
{text}
'''



class FeaturePreserver:
    def __init__(self, args, dataset):
        self.args = args
        self.cache = ResponseCache(path.join(args.cache_dir, f'{type(self).__name__}.{args.model}.{dataset}.json'))
        self.model_proxy = ModelProxy(args.max_tokens, self.cache)
        # prompts for system role
        self.sys_prompt = SYSTEM_PROMPT
        self.task_prompts = {
            'summarize': TASK_PROMPT_EXTRACT_CONTENT_SUMMARIZE,
            'outline': TASK_PROMPT_EXTRACT_CONTENT_OUTLINE,
            'simplify': TASK_PROMPT_EXTRACT_CONTENT_SIMPLIFY,
            'list': TASK_PROMPT_EXTRACT_EXPRESSION_LIST,
            'substitute': TASK_PROMPT_EXTRACT_EXPRESSION_SUBSTITUTE,
            'replace': TASK_PROMPT_EXTRACT_LANGUAGE_REPLACE,
        }
        # model settings
        self.model = args.model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.lang = get_lang(dataset)
        self.gen_lens = []
        self.res_lens = []

    def _get_validate_fn(self, item, model_name):
        id = item['id']

        def _validate_response(response, raise_exception=False):
            # # basic validation
            # valid = validate_common(model_name, id, response, self.lang, raise_exception)
            # if not valid:
            #     return False
            # # check repetition
            # valid = validate_repetition(model_name, id, response, self.lang, raise_exception)
            # if not valid:
            #     return False
            # # check garbage
            # valid = validate_garbage(model_name, id, response, self.lang, raise_exception)
            # if not valid:
            #     return False
            # check length
            length = len(item['generation'])
            real_length = len(response)
            if raise_exception:
                self.gen_lens.append(length)
                self.res_lens.append(real_length)
                summary = f'Expect {np.mean(self.gen_lens):.0f}({np.std(self.gen_lens):.0f}) vs got {np.mean(self.res_lens):.0f}({np.std(self.res_lens):.0f}) in {len(self.gen_lens)} samples.'
                if real_length < length // 4:
                    print(f'WARNING: Short response (expect {length} but got {real_length}) from {model_name}.', summary)
                elif real_length > length * 4 // 3:
                    print(f'WARNING: Long response (expect {length} but got {real_length}) from {model_name}.', summary)
                else:
                    print(f'PASS: Good response (expect {length} and got {real_length}) from {model_name}.', summary)
            # pass validation
            return True

        return _validate_response
    

    def _get_task_prompt(self, dataset, text):
        kwargs = {'lang': full_lang(self.lang), 'text': text}
        template = self.task_prompts[self.args.prompt]
        return template, template.format(**kwargs)

    def _extract_item(self, dataset, item):
        id = item['id']
        generation = item['generation']
        # call model
        model_name = self.model
        sys_prompt = self.sys_prompt
        task_template, task_prompt = self._get_task_prompt(dataset, generation)
        temperature = self.temperature
        topp = self.top_p
        validate_fn = self._get_validate_fn(item, model_name)
        feature_field = f'f_{self.args.prompt}'
        # check existence
        # if feature_field in item and validate_fn(item[feature_field]):
            # return item
        # call API
        try:
            output = self.model_proxy.prompt_generate(model_name, sys_prompt, task_prompt,
                                identifier=id, temperature=temperature, top_p=topp,
                                validate_fn=validate_fn)
            item[feature_field] = output
        except Exception as e:
            print(e)
        return item

    def preserve_feature(self, dataset):
        # load data
        data_file = path.join(self.args.data_path, f'{dataset}.json')
        data = load_json(data_file)
        # refine the language expression
        data = [self._extract_item(dataset, item) for item in tqdm(data)]
        save_json(data_file, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/raid")
    parser.add_argument('--datasets', type=str, default='raid.dev,raid.test')
    parser.add_argument('--prompt', type=str, default="simplify", choices=['summarize', 'outline', 'simplify', 'list', 'substitute', 'replace'])
    parser.add_argument('--strict', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--model', type=str, default='Qwen3-4B')
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    for dataset in args.datasets:
        print(f'Extract feature for {dataset} by {args.prompt} ...')
        extractor = FeaturePreserver(args, dataset)
        extractor.preserve_feature(dataset)

