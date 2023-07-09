import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from data import IEDataset
from model import OneIE, Ident
from config import Config
from util import save_result
from data import IEDatasetEval
from convert import json_to_cs
from scorer import score_graphs, score_ident
from graph import Graph
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task

cur_dir = os.path.dirname(os.path.realpath(__file__))

def load_model(model_path, device=0, gpu=False):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']

    # recover the model
    model = Ident(config, vocabs)
    model.load_state_dict(state['model'])

    if gpu:
        model.cuda(device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)

    return model, tokenizer, config, vocabs


def evalate(model_path, test_file, device=0, gpu=False,beam_size=20):
    # set gpu device
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config, vocabs = load_model(model_path, device=device, gpu=gpu)


    test_set = IEDataset(test_file, gpu=gpu,
                         relation_mask_self=config.relation_mask_self,
                         relation_directional=config.relation_directional,
                         symmetric_relations=config.symmetric_relations)


    test_set.numberize(tokenizer, vocabs)


    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        graphs = model.predict(batch)

        test_gold_graphs.extend(batch.graphs)
        test_pred_graphs.extend(graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)

    test_scores = score_ident(test_gold_graphs, test_pred_graphs)

    return test_scores

parser = ArgumentParser()
parser.add_argument('-m', '--model_path', help='path to the trained model')
parser.add_argument('-i', '--test_file', help='path to the input file')
parser.add_argument('--gpu', action='store_true', help='use gpu')
parser.add_argument('-d', '--device', default=0, type=int, help='gpu device index')


args = parser.parse_args()
test_scores = evalate(model_path=args.model_path,
                      test_file=args.test_file,
                      device=args.device,
                      gpu=args.gpu)