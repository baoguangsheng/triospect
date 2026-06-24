# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
import json
import time
from os import path
from types import SimpleNamespace
from .response_cache import ResponseCache
from .detector_base import DetectorBase
from .utils import load_json, save_json, get_pool_executor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    # Extract n-grams from the list of tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    # Find common words
    common_words = common_elements(tokens1, tokens2)

    # Find common n-grams (let's say up to 3-grams for this example)
    common_ngrams = set()
    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 3-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]

def get_data_stat(data_json):
    from fuzzywuzzy import fuzz
    for idx, each in enumerate(data_json):
        original = each['input']
        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for i in range(4)]
        cnt = 0
        whole_combined=''
        for pp in each.keys():
            if pp != 'common_features':
                whole_combined += (' ' + each[pp])
                res = calculate_sentence_common(original, each[pp])
                statistic_res[pp] = res
                all_statistic_res = sum_for_list(all_statistic_res, res)
                ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                cnt += 1
        
        each['fzwz_features'] = ratio_fzwz
        each['common_features'] = statistic_res
        each['avg_common_features'] = [a/cnt for a in all_statistic_res]
        each['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)
    return data_json

def get_feature_vec(input_json):
    all_list = []
    for idx, each in enumerate(input_json):
        raw = tokenize_and_normalize(each['input'])
        r_len = len(raw)*1.0
        assert r_len != 0

        each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
        for ek in each['common_features'].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])
        
        each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

        for ek in each['fzwz_features'].keys():
            each_data_fea.extend(each['fzwz_features'][ek])

        all_list.append(np.array(each_data_fea))

    all_list = np.vstack(all_list)
    return all_list


class Rewriter:
    def __init__(self, config):
        self.config = config
        self.cache = ResponseCache(path.join(config.cache_dir, f'{type(self).__name__}.raidar.jsonl'))
        self.client = self.prepare_client()
        # predefined prompts
        self.sys_prompt = 'You are a professional writer.'
        self.prompts = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

    def prepare_client(self):
        api_base = self.config.api_base
        api_key = self.config.api_key
        from openai import OpenAI
        client = OpenAI(
            base_url=api_base,
            api_key=api_key)
        return client

    def prompt_generate(self, task_prompt):
        model_name = self.config.rewriting_model_name
        kwargs = {"model": model_name, "temperature": 1.0, "top_p": 1.0, 'max_tokens': 1024,
                  "messages": [{"role": "system", "content": self.sys_prompt},
                               {"role": "user", "content": task_prompt}],
                  }

        key = self.cache.cachekey(kwargs)
        response_text = self.cache.get_cache(key)
        if response_text is None:
            # retry 1 time
            ntry = 2
            for idx in range(ntry):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    response = response.choices[0].message.content
                    break
                except Exception as e:
                    if idx < ntry - 1:
                        print(f'{model_name}, {kwargs}: {e}. Retrying ...')
                        time.sleep(5)
                        continue
                    self.cache.count_exception()
                    raise e
            self.cache.update_cache(key, response, model_name)
        else:
            response = response_text
        return response

    def rewrite(self, text):
        executor = get_pool_executor(self.config.rewriting_workers)
        futures = []
        for prompt in self.prompts:
            future = executor.submit(self.prompt_generate, prompt + ': ' + text)
            futures.append(future)
        # collect result
        result = {'input': text}
        for prompt, future in zip(self.prompts, futures):
            result[prompt] = future.result()
        return result


# the detector
class Raidar(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.rewriter = Rewriter(self.config)
        # self.preprocess, self.classifier = self._load_classifier()
        self.classifier = self._load_classifier()

    def _load_classifier(self):
        model = joblib.load(self.config.classifier)
        return model
    
    def compute_crit(self, text):
        try:
            result = self.rewriter.rewrite(text)
            features = get_feature_vec(get_data_stat([result]))
            # features = self.preprocess.transform(features)
            crit = self.classifier.predict_proba(features)[0, 1]
            return float(crit)
        except Exception as e:
            print(e)
            return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_test/our")    
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

    def xgboost_classifier(X_train, Y_train):
        X_train = get_feature_vec(X_train)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

        print(f'Training classifier with features {X_train.shape}...')
        # xgboost classifier
        import xgboost as xgb
        clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=10, random_state=42)
        # # Logistic Regression
        # clf = LogisticRegression()
        # # Neural Network
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        # clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
        # fit and test the model
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        print("Accuracy:", accuracy_score(Y_test, y_pred), "F1 score", f1_score(Y_test, y_pred))
        print(classification_report(Y_test, y_pred))
        return clf # (scaler, clf)

    def test_detector(data):
        X_train = [item for item in data]
        Y_train = [item['task_detect'] for item in data]
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        # detector
        detector = Raidar('raidar')
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

    test_detector(data)

    # load config
    config_name = 'raidar'
    config = load_json(f'./scripts/detectors/configs/{config_name}.json')
    config = SimpleNamespace(**config)
    # initialize feature preserver and detector
    rewriter = Rewriter(config)

    X_train = [rewriter.rewrite(item['generation']) for item in tqdm(data)]
    Y_train = [item['task_detect'] for item in data]

    X_train = get_data_stat(X_train)
    classifier = xgboost_classifier(X_train, Y_train)
    
    model_file = path.join(args.result_path, f'raidar.classifier.pkl')
    joblib.dump(classifier, model_file)
    print(f'Model saved to {model_file}')
