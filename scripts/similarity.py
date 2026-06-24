
from shapely import box
import torch
import spacy
import numpy as np
from os import path
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .utils import group_by_id, save_json, split_by_level, load_json


class SimCseSimilarity:

    def __init__(self, model_name="../../huggingface/models/princeton-nlp_sup-simcse-roberta-large"):
        # model_name = princeton-nlp/sup-simcse-roberta-large
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.text2embeds = {}

    def _get_embedding(self, text):
        if text in self.text2embeds:
            return self.text2embeds[text]
        embed = self.model.encode([text], convert_to_tensor=True)[0]
        self.text2embeds[text] = embed
        return embed

    def similarity(self, text1, text2):
        embed1 = self._get_embedding(text1)
        embed2 = self._get_embedding(text2)
        return cosine_similarity(embed1, embed2, dim=0).item()
   

# pip install spacy==3.8.0
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz
class PosBleuSimilarity:
    def __init__(self, spacy_model="en_core_web_sm", n_gram=4):
        self.nlp = spacy.load(spacy_model)
        self.n_gram = n_gram
        self.smooth = SmoothingFunction().method1
        self.text2poseqs = {}

    def _get_pos_sequence(self, text):
        if text in self.text2poseqs:
            return self.text2poseqs[text]
        doc = self.nlp(text)
        seq = [token.pos_ for token in doc]
        self.text2poseqs[text] = seq
        return seq

    def similarity(self, text1, text2):
        seq1 = self._get_pos_sequence(text1)
        seq2 = self._get_pos_sequence(text2)

        # BLEU expects list of reference sequences
        bleu = sentence_bleu([seq1], seq2, 
                             weights=tuple([1/self.n_gram]*self.n_gram), 
                             smoothing_function=self.smooth)
        return bleu


''' Similarity before and after Transformation:
    Cols: Transformation, SimCSE change, POS-BLEU change
    Rows: f1, f2, f3, f4, f5, f6
'''

def load_datasets(data_path, datasets):
    datasets = datasets.split(',') if type(datasets) == str else datasets
    items = []
    for dataset in datasets:
        dataset_file = path.join(data_path, f'{dataset}.json')
        items.extend(load_json(dataset_file))
    return items


def similarity_before_and_after_transformation(data):
    np.random.seed(42)
    np.random.shuffle(data)
    data = data[:1000]  # limit to 1000 samples for efficiency

    # export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
    simCse = SimCseSimilarity()
    posBleu = PosBleuSimilarity()

    # SimCSE similarity
    sim_tcs = []
    sim_tes = []
    all_texts = []
    for item in tqdm(data, desc="SimCSE Similarity"):
        T = item['generation']
        Tc = item['f_simplify']
        Te = item['f_replace']
        all_texts.extend([T])
        sim_tc = simCse.similarity(T, Tc)
        sim_te = simCse.similarity(T, Te)
        sim_tcs.append(sim_tc)
        sim_tes.append(sim_te)

    sim_rds = []
    for _ in tqdm(range(100), desc="SimCSE Similarity (Random)"):
        t0, t1, t2 = np.random.choice(all_texts, 3, replace=False)
        sim_rd1 = simCse.similarity(t0, t1)
        sim_rd2 = simCse.similarity(t0, t2)
        sim_rds.extend([sim_rd1, sim_rd2])

    print("SimCSE Similarity:")
    print(f"SimCSE(random): {round(np.mean(sim_rds), 2)} ({round(np.std(sim_rds), 2)})")
    print(f"SimCSE(T, Te): {round(np.mean(sim_tes), 2)} ({round(np.std(sim_tes), 2)})")
    print(f"SimCSE(T, Tc): {round(np.mean(sim_tcs), 2)} ({round(np.std(sim_tcs), 2)})")

    # POS-BLEU similarity
    sim_tcs = []
    sim_tes = []
    all_texts = []
    for item in tqdm(data, desc="POS-BLEU Similarity"):
        T = item['generation']
        Tc = item['f_simplify']
        Te = item['f_replace']
        all_texts.extend([T])
        sim_tc = posBleu.similarity(T, Tc)
        sim_te = posBleu.similarity(T, Te)
        sim_tcs.append(sim_tc)
        sim_tes.append(sim_te)

    sim_rds = []
    for idx in tqdm(range(100), desc="POS-BLEU Similarity (Random)"):
        t0, t1, t2 = [str(t) for t in np.random.choice(all_texts, 3, replace=False)]
        sim_rd1 = posBleu.similarity(t0, t1)
        sim_rd2 = posBleu.similarity(t0, t2)
        sim_rds.extend([sim_rd1, sim_rd2])

    print("POS-BLEU Similarity:")
    print(f"POS-BLEU(random): {round(np.mean(sim_rds), 2)} ({round(np.std(sim_rds), 2)})")
    print(f"POS-BLEU(T, Tc): {round(np.mean(sim_tcs), 2)} ({round(np.std(sim_tcs), 2)})")
    print(f"POS-BLEU(T, Te): {round(np.mean(sim_tes), 2)} ({round(np.std(sim_tes), 2)})")


''' Similarity before and after attack:
    Cols: Attack, SimCSE change, POS-BLEU change
    Rows: diversify, mimic, humbot.ai, bypassgpt.ai, undetectable.ai, human edit 
'''

def load_humanize16k_per_attack():
    data_path = './data/our'
    datasets = 'essay.dev,arxiv.dev,writing.dev,news.dev,essay.test,arxiv.test,writing.test,news.test'
    items = load_datasets(data_path, datasets)
    grouped_items = group_by_id(items)
    attack_items = {}
    for hum_id, items in grouped_items.items():
        if items[2] is None or items[3] is None:
            continue
        # attack: (T, T')
        attack = items[2]['process_records'][-1].split(',')[0]
        if attack not in attack_items:
            attack_items[attack] = []
        attack_items[attack].append((items[3]['generation'], items[2]['generation']))
        # other type of pairs
        # (before attack, after attack)
        key = "(T, T')"
        if key not in attack_items:
            attack_items[key] = []
        attack_items[key].append((items[3]['generation'], items[2]['generation'])) 
        # AI tool (before attack, after attack)
        key = "https"
        if attack.find(key) >= 0:
            if key not in attack_items:
                attack_items[key] = []
            attack_items[key].append((items[3]['generation'], items[2]['generation'])) 
        # (before gen, after gen)
        key = '(Th, Tm)'
        if key not in attack_items:
            attack_items[key] = []
        attack_items[key].append((items[0]['generation'], items[3]['generation']))  
        # (generation, f_simplify)
        key = '(T, Tc)'
        if key not in attack_items:
            attack_items[key] = []
        attack_items[key].append((items[0]['generation'], items[0]['f_simplify']))
        attack_items[key].append((items[2]['generation'], items[2]['f_simplify']))  
        attack_items[key].append((items[3]['generation'], items[3]['f_simplify']))  
        # (generation, f_replace)
        key = '(T, Te)'
        if key not in attack_items:
            attack_items[key] = []
        attack_items[key].append((items[0]['generation'], items[0]['f_replace']))
        attack_items[key].append((items[2]['generation'], items[2]['f_replace']))  
        attack_items[key].append((items[3]['generation'], items[3]['f_replace']))  
    return attack_items


def similarity_before_and_after_attack():
    attack_items = load_humanize16k_per_attack()

    # export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
    simCse = SimCseSimilarity()
    posBleu = PosBleuSimilarity()
    max_select = 200
    np.random.seed(42)
    # compute similarity for each attack type
    attack_sims = {}
    for attack, items in attack_items.items():
        # if attack.find('human') < 0:
        #     continue
        np.random.shuffle(items)
        items = items[:max_select]  # limit to 1000 samples for efficiency
        sim_cses = []
        pos_bleus = []
        for item in tqdm(items, desc=f"Processing attack {attack}"):
            text_before_attack = item[0]
            text_after_attack = item[1]
            sim_cse = simCse.similarity(text_before_attack, text_after_attack)
            pos_bleu = posBleu.similarity(text_before_attack, text_after_attack)
            sim_cses.append(sim_cse)
            pos_bleus.append(pos_bleu)
        sim_cses.sort()
        pos_bleus.sort()
        attack_sims[attack] = (sim_cses, pos_bleus)
    # save results to file
    save_json('./exp_analysis/similarity_before_and_after_attack.json', attack_sims)
    # print results
    for attack, (sim_cses, pos_bleus) in attack_sims.items():
        print(f"Attack: {attack}, SimCSE: {np.mean(sim_cses):.4f}({np.std(sim_cses):.4f}), POS-BLEU: {np.mean(pos_bleus):.4f}({np.std(pos_bleus):.4f})")


def draw_attack_similarity_boxplot():
    import matplotlib.pyplot as plt
    attacks = ['Humanize:action(diversify)', 
               'Humanize:action(mimic)', 
               'Humanize:action(https://humbot.ai/)',
               'Humanize:action(https://bypassgpt.ai/)', 
               'Humanize:action(https://undetectable.ai/)',
               'Humanize:action(human-editing)']
    xticks = ['diversify', 'mimic', 'humbot.ai', 'bypassgpt.ai', 'undetectable.ai', 'human edit']
    attack_sims = load_json('./exp_analysis/similarity_before_and_after_attack.json')
    attack_sims = [[attack_sims[attack][0], attack_sims[attack][1]] for attack in attacks]
    attack_sims = sum(attack_sims, [])  # flatten list of tuples
    xs = [i + (0.2 if i % 2 == 0 else -0.2) for i in range(len(attack_sims))]

    # plot
    colors = ['tab:blue', 'tab:orange']
    labels = ['SimCSE similarity', 'POS-BLEU similarity']

    # plot
    nrows = 1
    ncols = 1
    plt.clf()
    fig = plt.figure(figsize=(5, 2.5))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    axs.set_axisbelow(True)
    axs.grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
    axs.grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
    box = axs.boxplot(attack_sims, positions=xs, patch_artist=True, showfliers=False)
    for i, median in enumerate(box['medians']):
        median.set_color(colors[i % 2])
        median.set_linewidth(2)
    for i, patch in enumerate(box['boxes']):
        # patch.set_edgecolor(colors[i % 2])
        patch.set_facecolor(colors[i % 2])
        patch.set_alpha(0.4)
        patch.set_linewidth(1)
        patch.set_label(labels[i % 2])
    axs.set_xlabel('Humanizing Attack')
    axs.set_xticks([0.5, 2.5, 4.5, 6.5, 8.5, 10.5], xticks)
    axs.tick_params(axis='x', labelrotation=20, labelsize=8, pad=2)

    axs.set_ylabel('Similarity Score')
    plt.ylim(0.2, 1.0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    fig.legend(handles=box['boxes'][:2], loc='upper center', fontsize=7, ncol=2, handlelength=1)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.subplots_adjust(left=0.12, bottom=0.30, right=0.98, top=0.89)
    plt.savefig('./exp_analysis/attack_similarity_boxplot.pdf')


def draw_transform_similarity_boxplot():
    import matplotlib.pyplot as plt
    transforms = ['(T, Tc)', '(T, Te)']
    xticks = ['$(T,\\hat{T}_c)$', '$(T,\\hat{T}_e)$']
    transform_sims = load_json('./exp_analysis/similarity_before_and_after_attack.json')
    sim_cses = [transform_sims[transform][0] for transform in transforms]
    pos_bleus = [transform_sims[transform][1] for transform in transforms]
    sim_cse0 = np.mean(transform_sims['(T, T\')'][0])
    pos_bleu0 = np.mean(transform_sims['(T, Te)'][1])

    # plot
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    labels = ['SimCSE similarity', 'POS-BLEU similarity', 'Attack Mean']

    # plot
    nrows = 1
    ncols = 2
    plt.clf()
    fig = plt.figure(figsize=(5, 2.5))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=False, sharey=False)

    sims = [sim_cses, pos_bleus]
    handles = []
    for j in range(ncols):
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        line = axs[j].axhline(sim_cse0 if j == 0 else pos_bleu0, linestyle='--', linewidth=1, color=colors[2], label=labels[2])
        box = axs[j].boxplot(sims[j], patch_artist=True, showfliers=False)
        for median in box['medians']:
            median.set_color(colors[j])
            median.set_linewidth(2)
        for patch in box['boxes']:
            # patch.set_edgecolor(colors[i % 2])
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.4)
            patch.set_linewidth(1)
            patch.set_label(labels[j])
        handles.append(box['boxes'][0])
        if j == 1:
            handles.append(line)
        axs[j].set_xlabel('Transformation')
        axs[j].set_xticks([1, 2], xticks)
        axs[j].set_ylabel('Similarity Score')

    # plt.ylim(0.2, 1.0)
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    fig.legend(handles=handles, loc='upper center', fontsize=7, ncol=3, handlelength=1)
    plt.subplots_adjust(wspace=0.40, hspace=0.05)
    fig.subplots_adjust(left=0.12, bottom=0.30, right=0.98, top=0.89)
    plt.savefig('./exp_analysis/transform_similarity_boxplot.pdf')


def test_case():
    simCse = SimCseSimilarity()
    posBleu = PosBleuSimilarity()

    text1 = "The cat is on the mat."
    text2 = "A cat sits on a mat."
    text3 = "The dog is in the park."

    print("Content Similarity:", simCse.similarity(text1, text2), simCse.similarity(text1, text3))
    print("Expression Similarity:", posBleu.similarity(text1, text2), posBleu.similarity(text1, text3))

    

if __name__ == "__main__":
    # test_case()
    similarity_before_and_after_attack()
    draw_attack_similarity_boxplot()
    draw_transform_similarity_boxplot()

    # data = load_datasets('./data/our', 'essay.dev,arxiv.dev,writing.dev,news.dev')
    # similarity_before_and_after_transformation(data)
    # levels = split_by_level(data)
    # for idx, level in enumerate(levels):
    #     if len(level) == 0:
    #         continue
    #     print(f"Level {idx}:")
    #     similarity_before_and_after_transformation(level)
