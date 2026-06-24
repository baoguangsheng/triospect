# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import os.path as path
import random
import numpy as np
import tqdm
import argparse
import json
import time
from copy import deepcopy
from detectors import get_detector
from utils import load_json, save_json, is_machine, is_human, get_lang, get_risk_level, parse_record, count_words
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
from sklearn.mixture import GaussianMixture
import joblib

def load_datasets(data_path, datasets):
    datasets = datasets.split(',') if type(datasets) == str else datasets
    items = []
    for dataset in datasets:
        dataset_file = path.join(data_path, f'{dataset}.json')
        items.extend(load_json(dataset_file))
    return items

def load_results(result_path, datasets, detector):
    datasets = datasets.split(',') if type(datasets) == str else datasets
    results = []
    for dataset in datasets:
        result_file = path.join(result_path, f'{dataset}.{detector}.json')
        items = load_json(result_file)
        lang = get_lang(dataset)
        if lang != 'en':
            for item in items:
                item['domain'] += f'-{lang}'
        results.extend(items)
    return results

def save_results(results, result_path, dataset, detector):
    result_file = path.join(result_path, f'{dataset}.{detector}.json')
    save_json(result_file, results)
    return result_file

def save_detector(config, model, result_path, category, detector):
    temp_path = path.join(result_path, 'temp')
    if not path.exists(temp_path):
        os.makedirs(temp_path)
    config_file = path.join(temp_path, f'{category}.{detector}.config.json')
    save_json(config_file, config)
    if model is not None:
        model_file = path.join(temp_path, f'{category}.{detector}.model.pkl')
        joblib.dump(model, model_file)

def load_detector(result_path, category, detector):
    temp_path = path.join(result_path, 'temp')
    config_file = path.join(temp_path, f'{category}.{detector}.config.json')
    config = load_json(config_file)
    model_file = path.join(temp_path, f'{category}.{detector}.model.pkl')
    model = joblib.load(model_file) if path.exists(model_file) else None
    return config, model


def save_report(report, result_path, category, detector):
    temp_path = path.join(result_path, 'temp')
    if not path.exists(temp_path):
        os.makedirs(temp_path)
    report_file = path.join(temp_path, f'{category}.{detector}.report.json')
    save_json(report_file, report)
    return report_file


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

class DelegateDetector:
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.name_md, self.detector_name = self._split_name(name)
        self.detector = None
        self.feature_fields = self._get_feature_fields(self.name_md)
        self.eval_fields = self.feature_fields

    def _get_feature_fields(self, name_md):
        abbr_fields = {
            'f0': 'generation',  # original text (T)
            'f1': 'f_summarize', 
            'f2': 'f_outline',  
            'f3': 'f_simplify', 
            'f4': 'f_list', 
            'f5': 'f_substitute', 
            'f6': 'f_replace',
        }
        if name_md is None:
            # empty setting
            return [abbr_fields['f0']]
        if name_md == 'triospect':
            # default setting
            return [abbr_fields[f] for f in "f0-f3-f6".split('-')]
        # customized setting
        return [abbr_fields[f] for f in name_md.split('-')]

    def _split_name(self, name):
        start = name.find('(')
        end = name.find(')')
        if start >= 0 and end >= 0:
            return name[:start], name[start + 1:end]
        return None, name

    def _prepare(self, dataset):
        # check and skip if result file exists
        result_file = path.join(self.args.result_path, f'{dataset}.{self.detector_name}.json')
        if path.exists(result_file):
            # print(f'Skip preparing, using existing file: {result_file}')
            return
        # initialize or use the cached detector
        if self.detector is None:
            self.detector = get_detector(self.detector_name)
        # compute detection criterion
        print(f'Preparing {dataset}.{self.detector_name}.json ...')
        items = load_datasets(self.args.data_path, dataset)
        # compute criteria
        results = []
        for item in tqdm.tqdm(items, desc=f"Computing {self.detector_name} criteria"):
            result = deepcopy(item)
            # default behavior
            for field in self.eval_fields:
                if field in item:
                    target = item[field]
                    crit = self.detector.compute_crit(target)
                    result[f'{field}_crit'] = crit
            results.append(result)
        save_results(results, self.args.result_path, dataset, self.detector_name)

    def prepare(self, datasets):
        for dataset in datasets:
            self._prepare(dataset)


    def _fit_threshold(self, datasets, category, label_fn):
        results = load_results(self.args.result_path, datasets, self.detector_name)
        random.shuffle(results)
        results = results[: int(len(results) * self.args.dev_ratio)]
        print(f'Fit threshold on {len(results)} samples for {self.name}.')
        # prepare data
        assert len(self.feature_fields) == 1
        field = self.feature_fields[0]
        pairs = [(item[f'{field}_crit'], label_fn(item)) for item in results]
        pairs = [(0 if str(c) == 'nan' else c, l) for c, l in pairs if l is not None]
        crits = [c for c, _ in pairs]
        labels = [l for _, l in pairs]
        # identify direction
        crits_pos = [c for c, l in pairs if l]
        crits_neg = [c for c, l in pairs if not l]
        pos_bigger = np.mean(crits_pos) > np.mean(crits_neg)
        if not pos_bigger:
            print(f'WARNING: Negative samples ({np.mean(crits_neg)}) are larger than positive samples ({np.mean(crits_pos)}).')
            crits = [-c for c in crits]

        # find threshold
        fpr, tpr, thresholds = roc_curve(labels, crits)
        j = tpr - fpr  # Youden’s J statistic
        index = np.argmax(j)
        threshold = thresholds[index]
        # verify classifier
        preds =[crit >= threshold for crit in crits]
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        # save detector
        config = {'detector': self.detector_name,
                  'fit_data': {'samples': len(results), 'positives': int(np.sum(labels)), 'field': field,
                               'pos_mean': float(np.mean(crits_pos)), 'pos_std': float(np.std(crits_pos)),
                               'neg_mean': float(np.mean(crits_neg)), 'neg_std': float(np.std(crits_neg)),
                               'pos_bigger': bool(pos_bigger)},
                  'threshold': threshold}
        save_detector(config, None, self.args.result_path, category, self.name)

    def _fit_model(self, datasets, category, label_fn):
        results = load_results(self.args.result_path, datasets, self.detector_name)
        random.shuffle(results)
        results = results[: int(len(results) * self.args.dev_ratio)]
        print(f'Fit model on {len(results)} samples for {self.name}.')
        # prepare data
        pairs = [([item[f'{field}_crit'] for field in self.feature_fields], label_fn(item))
                 for item in results]
        pairs = [([0 if str(c) == 'nan' else c for c in cs], l) for cs, l in pairs if l is not None]
        features = [cs for cs, _ in pairs]
        labels = [l for _, l in pairs]
        # fit a model
        model = DelegateModel(self.args.model).fit(features, labels)
        crits = model.predict(features)
        crits_pos = [c for c, l in zip(crits, labels) if l]
        crits_neg = [c for c, l in zip(crits, labels) if not l]
        # find threshold
        fpr, tpr, thresholds = roc_curve(labels, crits)
        j = tpr - fpr  # Youden’s J statistic
        index = np.argmax(j)
        threshold = thresholds[index]
        # verify classifier
        preds =[crit >= threshold for crit in crits]
        f1_pred = f1_score(labels, preds)
        # save detector
        fields = ','.join(self.feature_fields)
        config = {'detector': self.detector_name,
                  'fit_data': {'samples': len(results), 'positives': int(np.sum(labels)), 'fields': fields,
                               'pos_mean': float(np.mean(crits_pos)), 'pos_std': float(np.std(crits_pos)),
                               'neg_mean': float(np.mean(crits_neg)), 'neg_std': float(np.std(crits_neg)), 
                               'pos_bigger': True},
                  'threshold': threshold}
        save_detector(config, model, self.args.result_path, category, self.name)

    def fit(self, datasets, category, label_fn):
        if len(self.feature_fields) == 1:
            self._fit_threshold(datasets, category, label_fn)
        else:
            self._fit_model(datasets, category, label_fn)


    def _predict_threshold(self, config, model, item):
        # assert model is None
        assert len(self.feature_fields) == 1
        field = self.feature_fields[0]
        crit = item[f'{field}_crit']
        crit = crit if crit == crit else 0
        pos_bigger = config['fit_data']['pos_bigger']
        threshold = config['threshold']
        crit = crit if pos_bigger else -crit
        pred = crit >= threshold
        return crit, pred

    def _predict_model(self, config, model, item):
        features = [[item[f'{field}_crit'] for field in self.feature_fields]]
        features = np.nan_to_num(features, nan=0).tolist()
        crit = model.predict(features)[0]
        pos_bigger = config['fit_data']['pos_bigger']
        threshold = config['threshold']
        crit = crit if pos_bigger else -crit
        pred = crit >= threshold
        return crit, pred

    def _eval(self, results, config, model, label_fn, group):
        # predict
        triples = []
        for item in results:
            label = label_fn(item)
            if label is None:
                continue
            if len(self.feature_fields) == 1:
                crit, pred = self._predict_threshold(config, model, item)
            else:
                crit, pred = self._predict_model(config, model, item)
            triples.append((crit, bool(pred), bool(label), item))
        # save wrong predictions
        if group is None:
            attack = 'synonym'
            detector = config['detector']
            results = [t for t in triples if t[3]['note'].find(f'attack({attack})') < 0]
            crit0 = [t[3]['generation_crit'] for t in results]
            crit1 = [t[3]['f_simplify_crit'] for t in results]
            crit2 = [t[3]['f_replace_crit'] for t in results]
            print(f'Pearson for {attack}:', np.corrcoef(crit0, crit1)[0, 1].round(4), np.corrcoef(crit0, crit2)[0, 1].round(4))
            save_json(f'./exp_test/{detector}.tuples.json', results)
            print(f'Save results to ./exp_test/{detector}.tuples.json')
        # make sure positive and negative samples are balanced
        pos_triples = [t for t in triples if t[2] == 1]
        neg_triples = [t for t in triples if t[2] == 0]
        if len(pos_triples) != len(neg_triples):
            print(f'WARNING: Eval with positive {len(pos_triples)} but negative {len(neg_triples)} in group {group}')
            min_len = min(len(pos_triples), len(neg_triples))
            pos_triples = random.sample(pos_triples, min_len)
            neg_triples = random.sample(neg_triples, min_len)
            assert len(pos_triples) == len(neg_triples)
            triples = pos_triples + neg_triples
        assert len(pos_triples) > 0
        assert len(neg_triples) > 0
        # auroc, f1, tpr@fpr5%
        crits = [t[0] for t in triples]
        preds = [t[1] for t in triples]
        labels = [t[2] for t in triples]
        fpr, tpr, thresholds = roc_curve(labels, crits)
        auroc = auc(fpr, tpr)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        tpr05 = [t for f, t in zip(fpr, tpr) if f <= 0.05]
        tpr05 = tpr05[-1] if len(tpr05) > 0 else 0.0
        tpr01 = [t for f, t in zip(fpr, tpr) if f <= 0.01]
        tpr01 = tpr01[-1] if len(tpr01) > 0 else 0.0
        tpr001 = [t for f, t in zip(fpr, tpr) if f <= 0.001]
        tpr001 = tpr001[-1] if len(tpr001) > 0 else 0.0
        pos_crits = [t[0] for t in pos_triples]
        neg_crits = [t[0] for t in neg_triples]
        result = {'group': group,
                'npos': int(np.sum(labels)), 'nsamples': len(labels),
                'pos_mean': np.mean(pos_crits), 'pos_std': np.std(pos_crits),
                'neg_mean': np.mean(neg_crits), 'neg_std': np.std(neg_crits),
                'auroc': auroc, 'f1': f1, 'acc':acc, 'tpr05': tpr05, 'tpr01': tpr01, 'tpr001': tpr001}
        return result

    def eval(self, datasets, category, label_fn, group_fn):
        results = load_results(self.args.result_path, datasets, self.detector_name)
        config, model = load_detector(self.args.result_path, category, self.name)
        fit_data = config['fit_data']
        groups = set([None] + [group_fn(item) for item in results])
        print(f'Eval groups: {groups}')
        report = {}
        for group in groups:
            group_results = [item for item in results if group is None or group_fn(item) == group]
            result = self._eval(group_results, config, model, label_fn, group)
            result['category'] = category
            report[group] = result
        save_report(report, self.args.result_path, category, self.name)
        return report

def get_detect_label_fns(categorize):
    label_fns = {
        'detect': (lambda item: item[f'task_{categorize}']),
    }
    return label_fns

def print_report(groups, detectors, categories, reports, type='eval'):
    type_templates = {
        'eval': '{auroc:.3f}/{f1:.2f}/{acc:.2f}/{tpr01:.2f}',
        'distrib': '{pos_mean:.2f}({pos_std:.2f}) vs {neg_mean:.2f}({neg_std:.2f})'
    }

    results = []
    for group in groups:
        for idx, detector_name in enumerate(detectors):
            group_str = 'ALL' if group is None else group
            cols0 = [f'{group_str}:']
            cols1 = [f'{detector_name}:']
            for category in categories:
                report = reports[f'{detector_name}-{category}'][group]
                cols0.append('{group}:{category}({npos}/{nsamples})'.format(**report))
                cols1.append(type_templates[type].format(**report))
            if idx == 0:
                results.append('\t'.join(cols0))
            results.append('\t'.join(cols1))

    print('\n'.join(results))


def main(args):
    print(args)
    if args.categorize in ['detect']:
        label_fns = get_detect_label_fns(args.categorize)
    else:
        raise NotImplemented

    def _group_fn(item):
        if args.group == 'domain':
            return item['domain']
        else:
            raise NotImplemented

    group_fn = _group_fn
    print(f'Processing {args.datasets} ...')
    categories = list(label_fns.keys())
    groups = set()
    reports = {}
    for detector_name in args.detectors:
        random.seed(args.seed)
        np.random.seed(args.seed)
        detector = DelegateDetector(args, detector_name)
        detector.prepare(args.datasets)
        for category in categories:
            devs = [dataset for dataset in args.datasets if dataset.endswith('.dev')]
            tests = [dataset for dataset in args.datasets if dataset.endswith('.test')]
            if len(devs) == 0 and len(tests) == 0:
                devs = tests = args.datasets
            detector.fit(devs, category, label_fns[category])
            report = detector.eval(tests, category, label_fns[category], group_fn)
            reports[f'{detector_name}-{category}'] = report
            groups.update(report.keys())
    # print
    print_report(groups, args.detectors, categories, reports, 'distrib')
    print_report(groups, args.detectors, categories, reports, 'eval')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_test/raid")
    parser.add_argument('--data_path', type=str, default="./data/raid")
    parser.add_argument('--model', type=str, default="GaussianMixture")
    parser.add_argument('--categorize', type=str, default="detect")
    parser.add_argument('--group', type=str, default="domain")
    parser.add_argument('--detectors', type=str, default="fast_detect")
    parser.add_argument('--datasets', type=str, default="raid.dev,raid.test")
    parser.add_argument('--dev_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    args.detectors = args.detectors.split(',')

    random.seed(args.seed)
    np.random.seed(args.seed)
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
