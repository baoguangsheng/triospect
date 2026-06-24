# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
import tqdm
import argparse
import json
import time

class ModelAPI:
    def __init__(self, config):
        # model clients
        self.client = self.prepare_client(config)

    def prepare_client(self, config):
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=config.api_base,
            api_key=config.api_key,
            api_version=config.api_version)
        return client

    def evaluate(self, kwargs):
        # model_name = kwargs['model']
        # print(f'{model_name}, {kwargs}')
        response = self.client.completions.create(**kwargs)
        result = response.choices[0]
        return result
