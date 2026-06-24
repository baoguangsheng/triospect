# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import time
from collections import Counter
from utils import load_json, save_json
import hashlib

class ResponseCache:
    def __init__(self, cache_file, n_update=20):
        self.n_update = n_update
        self.cache_file = cache_file
        if os.path.exists(self.cache_file):
            self.cache = load_json(self.cache_file)
            self.n_saved = len(self.cache)
        else:
            self.cache = {}
            self.n_saved = 0
        self.counter = Counter()

    def __del__(self):
        save_json(self.cache_file, self.cache)
        self.n_saved = len(self.cache)
        if len(self.counter) > 0:
            print(f'{self.cache_file} new responses: {self.counter}')

    def cachekey(self, kwargs, identifier=None):
        key = json.dumps(kwargs) + (identifier if identifier else '')
        key = hashlib.md5(key.encode()).hexdigest()
        return key

    def update_cache(self, key, response, category):
        assert len(key) > 0 and len(response) > 0
        self.cache[key] = response
        self.counter[category] += 1
        if len(self.cache) >= self.n_saved + self.n_update:
            save_json(self.cache_file, self.cache)
            self.n_saved = len(self.cache)
            # print(f'Saved cache with {self.n_saved} items.')

    def count_exception(self):
        self.counter['exception'] += 1

    def get_cache(self, key):
        if key in self.cache:
            return self.cache[key]
        return None


class ModelProxy:
    def __init__(self, max_tokens, cache):
        self.max_tokens = max_tokens
        self.cache = cache
        # model clients
        self.clients = dict()
        self.clients.update(self.prepare_client0())  # local run
        self.clients.update(self.prepare_client1())  # remote api

    def prepare_client0(self):
        from openai import OpenAI
        clients = {
            'llama-3.3-70b-instruct': ('Llama-3.3-70B-Instruct', OpenAI(base_url='http://xxxx/v1', api_key='na')),
            'qwen2.5-72b-instruct': ('Qwen2.5-72B-Instruct', OpenAI(base_url='http://xxxx/v1', api_key='na')),
            'DeepSeek-R1-Distill-Qwen-7B': ('DeepSeek-R1-Distill-Qwen-7B',  OpenAI(base_url='http://172.16.78.10:38918/v1', api_key='na')),
            'DeepSeek-R1-Distill-Llama-8B': ('DeepSeek-R1-Distill-Llama-8B', OpenAI(base_url='http://172.16.78.10:32074/v1', api_key='na')),
            'Qwen3-8B-AWQ': ('Qwen3-8B-AWQ', OpenAI(base_url='http://172.16.78.10:36427/v1', api_key='na')),
            'Qwen3-4B': ('Qwen3-4B-Nothinking', OpenAI(base_url='http://172.16.78.10:38070/v1', api_key='na')),
        }
        return clients

    def prepare_client1(self):
        api_base = 'https://xxxx/v1'
        api_key = 'xxxx'
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key)
        clients = {
            'claude-3-5-sonnet-20241022': ('claude-3-5-sonnet-20241022', client),
            'gpt-3.5-turbo-0125': ('gpt-3.5-turbo-0125', client),
            'gpt-4o-2024-11-20': ('gpt-4o-2024-11-20', client),
            'gemini-1.5-pro-latest': ('gemini-1.5-pro-latest', client),
            'gemini-2.5-pro-exp-03-25': ('gemini-2.5-pro-exp-03-25', client),
            'deepseek-v3': ('deepseek-v3', client),
            'deepseek-r1': ('deepseek-r1', client),
        }
        return clients

    def prompt_generate(self, model_name, sys_prompt, task_prompt, identifier=None, temperature=1.0, top_p=1.0, validate_fn=None):
        kwargs = {"model": model_name, "temperature": temperature, "top_p": top_p, 'max_tokens': self.max_tokens,
                  "messages": [{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": task_prompt}],
                  }
        key = self.cache.cachekey(kwargs, identifier)
        response = self.cache.get_cache(key)
        if response is None or not validate_fn(response):
            # mapping model to client and real model name
            if model_name not in self.clients:
                raise Exception('No client for model: {}'.format(model_name))
            model_name, client = self.clients[model_name]
            kwargs['model'] = model_name
            # retry 1 time
            ntry = 2
            for idx in range(ntry):
                try:
                    response = client.chat.completions.create(**kwargs)
                    response = response.choices[0].message.content
                    parts = response.split('</think>')
                    if len(parts) == 2:
                        print(f'Thinking {len(parts[0].split())} words and response {len(parts[1].split())} words.')
                        response = parts[1].strip()
                    validate_fn(response, raise_exception=True)
                    break
                except Exception as e:
                    if idx < ntry - 1:
                        print(f'{model_name}, {identifier}, {kwargs}')
                        print(f'{model_name}: {e}. {response}. Retrying ...')
                        time.sleep(5)
                        continue
                    self.cache.count_exception()
                    raise e
            self.cache.update_cache(key, response, model_name)
        return response

