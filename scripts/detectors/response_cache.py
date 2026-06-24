# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import time
from io import open
from collections import Counter
from .utils import load_text
import hashlib
from threading import Lock

class ResponseCache:
    def __init__(self, cache_file, n_update=40):
        self.n_update = n_update
        self.cache_file = cache_file
        self.updated_keys = set()
        self.cache = {}
        self.counter = Counter()
        self._load_cache()
        self.lock = Lock()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            print('Load cache:', self.cache_file)
            lines = load_text(self.cache_file).split('\n')
            for line in lines:
                if line:
                    item = json.loads(line)
                    self.cache.update(item)

    def _save_cache(self):
        # dump to lines
        lines = []
        for key in self.updated_keys:
            value = self.cache[key]
            line = json.dumps({key: value})
            lines.append(line)
        # save to file
        print('Save cache:', self.cache_file)
        with open(self.cache_file, 'a') as fout:
            fout.writelines((f'{line}\n' for line in lines))
        # clear records
        self.updated_keys.clear()


    def __del__(self):
        with self.lock:
            self._save_cache()
            if len(self.counter) > 0:
                print(f'{self.cache_file} new responses: {self.counter}')

    def cachekey(self, kwargs, identifier=None):
        key = json.dumps(kwargs) + (identifier if identifier else '')
        key = hashlib.md5(key.encode()).hexdigest()
        return key

    def update_cache(self, key, response, category):
        with self.lock:
            assert len(key) > 0 and len(response) > 0
            self.cache[key] = response
            self.updated_keys.add(key)
            self.counter[category] += 1
            if len(self.updated_keys) >= self.n_update:
                self._save_cache()

    def count_exception(self):
        with self.lock:
            self.counter['exception'] += 1

    def get_cache(self, key):
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            return None
