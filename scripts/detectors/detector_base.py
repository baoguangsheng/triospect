# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .utils import load_json
from types import SimpleNamespace

class DetectorBase:
    def __init__(self, config_name):
        self.config_name = config_name
        self.config = load_json(f'./scripts/detectors/configs/{config_name}.json')
        self.config = SimpleNamespace(**self.config)

    def compute_crit(self, text):
        raise NotImplementedError

    def __str__(self):
        return self.config_name
