# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import json
from os import path
import time
import joblib
import argparse
import random
from tqdm import tqdm
from types import SimpleNamespace
from .utils import load_json, save_json, get_pool_executor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from .detector_base import DetectorBase
from .response_cache import ResponseCache
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture


SYSTEM_PROMPT = 'You are a professional writer.'

TASK_PROMPT_EXTRACT_CONTENT_SIMPLIFY = \
'''Simplify the text in {lang} language to make it clear and concise while preserving its meaning:
{text}
'''

TASK_PROMPT_EXTRACT_LANGUAGE_REPLACE = \
'''Replace the main points of the text in {lang} language with a generic topic while preserving the language expression:
{text}
'''
MD_FEATURES = ['text', 'content', 'expression']

class FeaturePreserver:
    def __init__(self, config):
        self.config = config
        self.client = self.create_client_openai()
        self.sys_prompt = SYSTEM_PROMPT
        self.task_prompts = {
            'content': TASK_PROMPT_EXTRACT_CONTENT_SIMPLIFY,
            'expression': TASK_PROMPT_EXTRACT_LANGUAGE_REPLACE,
        }
        self.lang = 'en'  # default language
        # cache
        self.cache = ResponseCache(path.join(config.cache_dir, f'{type(self).__name__}.md_detector.jsonl'))

    def create_client_openai(self):
        from openai import OpenAI
        return OpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key)

    def _get_task_prompt(self, prompt, text):
        kwargs = {'text': text, 'lang': self.lang}
        template = self.task_prompts[prompt]
        return template.format(**kwargs)

    def _extract(self, prompt, text):
        model_name = self.config.transform_model_name
        sys_prompt = self.sys_prompt
        task_prompt = self._get_task_prompt(prompt, text)
        kwargs = {"model": model_name, "max_tokens": self.config.transform_max_tokens,
                  "temperature": 1.0, "top_p": self.config.transform_top_p, 'extra_body': {"random_seed": 42},
                  "messages": [{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": task_prompt}],
                  }
        # search cache
        key = self.cache.cachekey(kwargs, '')
        response = self.cache.get_cache(key)
        if response is not None:
            return response

        # retry 1 time
        ntry = 1
        for idx in range(ntry):
            try:
                response = self.client.chat.completions.create(**kwargs)
                response = response.choices[0].message.content
                self.cache.update_cache(key, response, model_name)
                return response
            except Exception as e:
                if idx < ntry - 1:
                    print(f'{self.config.api_base}, {model_name}, {kwargs}: {e}. Retrying ...')
                    time.sleep(5)
                    continue
                self.cache.count_exception()
                raise e
            
    def extract(self, text):
        executor = get_pool_executor(self.config.transform_workers)
        futures = []
        for prompt in self.task_prompts:
            future = executor.submit(self._extract, prompt, text)
            futures.append(future)
        # collect result
        result = {'text': text}
        for prompt, future in zip(self.task_prompts, futures):
            result[prompt] = future.result()
        return result


class DelegateModel:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'GaussianMixture':
            # use single component by default
            self.model = [GaussianMixture(n_components=1, covariance_type='full', random_state=0),
                          GaussianMixture(n_components=1, covariance_type='full', random_state=0)]
        else:
            self.model = eval(f'{model_name}()')

    def fit(self, xs, ys):
        if self.model_name == 'GaussianMixture':
            xs0 = [x for x, y in zip(xs, ys) if y == 0]
            xs1 = [x for x, y in zip(xs, ys) if y == 1]
            self.model[0].fit(xs0)
            self.model[1].fit(xs1)
        else:
            self.model = self.model.fit(xs, ys)
        return self

    def predict(self, xs):
        if self.model_name == 'GaussianMixture':
            proba0 = np.exp(self.model[0].score_samples(xs))
            proba1 = np.exp(self.model[1].score_samples(xs))
            return proba1 / (proba0 + proba1)
        return self.model.predict(xs)

# the detector
class MdDetector(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.gpt = FeaturePreserver(self.config)
        from . import get_detector
        self.detector = get_detector(self.config.detector_name)
        self.classifier = self._load_classifier()
        self.features = MD_FEATURES

    def _load_classifier(self):
        model = joblib.load(self.config.classifier)
        return model

    def _classify(self, crits):
        features = [crits]
        features = np.nan_to_num(features, nan=0).tolist()
        crit = self.classifier.predict(features)[0]
        return crit

    def compute_crit(self, text):
        texts = self.gpt.extract(text)
        texts = [texts[fea] for fea in self.features]
        crits = [self.detector.compute_crit(t) for t in texts]
        prob = self._classify(crits)
        return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./classifiers")    
    parser.add_argument('--data_path', type=str, default="./data/our")
    parser.add_argument('--datasets', type=str, default='essay.dev,arxiv.dev,writing.dev,news.dev')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
   
    random.seed(args.seed)
    np.random.seed(args.seed)

    def load_datasets(data_path, datasets):
        datasets = datasets.split(',') if type(datasets) == str else datasets
        items = []
        for dataset in datasets:
            dataset_file = path.join(data_path, f'{dataset}.json')
            items.extend(load_json(dataset_file))
        return items

    def gaussian_classifier(X_train, Y_train):
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

        clf = DelegateModel('GaussianMixture')
        clf.fit(X_train, Y_train)
        # test the model
        y_pred = clf.predict(X_test)
        y_pred = y_pred > 0.5
        
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        print("Accuracy:", accuracy_score(Y_test, y_pred), "F1 score", f1_score(Y_test, y_pred))
        print(classification_report(Y_test, y_pred))
        return clf

    def test_detector(data):
        X_train = [item for item in data]
        Y_train = [item['task_detect'] for item in data]
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        # detector
        detector = MdDetector('md_detector')
        y_pred = []
        for item in tqdm(X_test):
            prob = detector.compute_crit(item['generation'])
            y_pred.append(prob > 0.5)
        
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        print("Accuracy:", accuracy_score(Y_test, y_pred), "F1 score", f1_score(Y_test, y_pred))
        print(classification_report(Y_test, y_pred))
        exit(0)

    args.datasets = args.datasets.split(',')
    data = load_datasets(args.data_path, args.datasets)
    random.shuffle(data)
    data = data[:500]  
    
    # load config
    config_name = 'md_detector'
    config = load_json(f'./scripts/detectors/configs/{config_name}.json')
    config = SimpleNamespace(**config)
    # initialize feature preserver and detector
    from . import get_detector
    gpt = FeaturePreserver(config)
    detector = get_detector(config.detector_name)
    # transform texts
    print('Transorming texts ...')
    X_train = [gpt.extract(item['generation']) for item in tqdm(data)]
    Y_train = [item['task_detect'] for item in data]
    # compute crits
    print('Computing crits ...')
    features = MD_FEATURES
    X_train = [[detector.compute_crit(item[fea]) for fea in features] for item in tqdm(X_train)]
    classifier = gaussian_classifier(X_train, Y_train)
    
    model_file = path.join(args.result_path, f'md_detector.classifier.humanize-16k.pkl')
    joblib.dump(classifier, model_file)
    print(f'Model saved to {model_file}')

    test_detector(data)
