from .baselines import Baselines
from .fast_detect_gpt import FastDetectGPT
from .binoculars import Binoculars
from .radar import Radar
from .roberta import RoBERTa
from .detect_llm import LRR
from .imbd import IMBD
from .raidar import Raidar
from .md_detector import MdDetector
from .md_detector2 import MdDetector2


def get_detector(name):
    name_detectors = {
        'roberta': ('roberta', RoBERTa),
        'radar': ('radar', Radar),
        'log_perplexity': ('log_perplexity', Baselines),
        'log_rank': ('log_rank', Baselines),
        'lrr': ('lrr', LRR),
        'fast_detect': ('fast_detect', FastDetectGPT),
        'binoculars': ('binoculars', Binoculars),
        'imbd': ('imbd', IMBD),
        'raidar': ('raidar', Raidar),
        'md_detector': ('md_detector', MdDetector),
        'md_detector2': ('md_detector2', MdDetector2),
    }
    if name in name_detectors:
        config_name, detector_class = name_detectors[name]
        return detector_class(config_name)
    else:
        raise NotImplementedError
