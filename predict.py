import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig,AlbertTokenizer
import random
import numpy as np
from model import OneIE
from config import Config
from util import save_result
from data import IEDatasetEval, IEDataset
from convert import json_to_cs
from scorer import score_graphs
from util import generate_vocabs
cur_dir = os.path.dirname(os.path.realpath(__file__))
format_ext_mapping = {'txt': 'txt', 'ltf': 'ltf.xml', 'json': 'json',
                      'json_single': 'json'}

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(1024)

def load_model(model_path, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    if 'albert' in config.bert_model_name:
        # breakpoint()
        config.bert_cache_dir = os.path.join(cur_dir, 'albert')
    else:
        config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'], False)
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)

    if 'albert' in config.bert_model_name:
        # tokenizer = AlbertTokenizer.from_pretrained(config.bert_model_name,
        #                                       cache_dir=config.bert_cache_dir,
        #                                       do_lower_case=False)
        tokenizer = AlbertTokenizer.from_pretrained(config.bert_model_name)
        # breakpoint()
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)

    return model, tokenizer, config


def load_previous_model(model_path, device=0, gpu=False):
    print('Loading the previous model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    
    config.bert_cache_dir = os.path.join(cur_dir, 'albert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'], False)
    model.beam_size = 5
    if gpu:
        model.cuda(device)

    tokenizer = AlbertTokenizer.from_pretrained(config.bert_model_name,
                                            cache_dir=config.bert_cache_dir,
                                            do_lower_case=False)

    return model, tokenizer, config, vocabs


def predict_document(path, model, tokenizer, config, batch_size=10, 
                     max_length=128, gpu=False, input_format='txt',
                     language='english'):
    """
    :param path (str): path to the input file.
    :param model (OneIE): pre-trained model object.
    :param tokenizer (BertTokenizer): BERT tokenizer.
    :param config (Config): configuration object.
    :param batch_size (int): Batch size (default=20).
    :param max_length (int): Max word piece number (default=128).
    :param gpu (bool): Use GPU or not (default=False).
    :param input_format (str): Input file format (txt or ltf, default='txt).
    :param langauge (str): Input document language (default='english').
    """
    test_set = IEDatasetEval(path, max_length=max_length, gpu=gpu,
                             input_format=input_format, language=language)
    test_set.numberize(tokenizer)
    # test_set = IEDataset(path, gpu=gpu)
    # train_set = IEDataset('data/dygie/train.oneie.json', gpu=gpu)
    # dev_set = IEDataset('data/dygie/dev.oneie.json', gpu=gpu)
    # vocabs = generate_vocabs([train_set, dev_set, test_set])
    # test_set.numberize(tokenizer,vocabs)
    # document info
    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set)
    }
    # info = {
    #     'doc_id': 0,
    #     'ori_sent_num': 10,
    #     'sent_num': 10
    # }
    # prediction result
    result = []
    # test_gold_graphs = []
    # test_pred_graphs = []
    
    start_time = time.perf_counter()
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            collate_fn=test_set.collate_fn):
        # breakpoint()
        
        graphs = model.predict(batch)
        # test_gold_graphs.extend(batch.graphs)
        # test_pred_graphs.extend(graphs)
        # end_time = time.time()
        for graph, tokens, sent_id, token_ids in zip(graphs, batch.tokens,
                                                     batch.sent_ids,
                                                     batch.token_ids):
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
            result.append((sent_id, token_ids, tokens, graph))
    end_time = time.perf_counter()
    speed = 832/(end_time-start_time)
    print(speed)
    # breakpoint()
    # breakpoint()
    # test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
    #                            relation_directional=config.relation_directional)
    # print(test_scores)
    # infer_speed = 10/(sum(infer_time[2:])/(len(infer_time)-2))    
    
    return result, info


def predict(model_path, input_path, output_path, log_path=None, cs_path=None,
         batch_size=5, max_length=200, device=0, gpu=False,
         file_extension='json', beam_size=5, input_format='txt',
         language='english'):
    """Perform information extraction.
    :param model_path (str): Path to the pre-trained model file.
    :param input_path (str): Path to the input directory.
    :param output_path (str): Path to the output directory.
    :param log_path (str): Path to the log file.
    :param cs_path (str): (optional) Path to the cold-start format output directory.
    :param batch_size (int): Batch size (default=50).
    :param max_length (int): Max word piece number for each sentence (default=128).
    :param device (int): GPU device index (default=0).
    :param gpu (bool): Use GPU (default=False).
    :param file_extension (str): Input file extension. Only files ending with the
    given extension will be processed (default='txt').
    :param beam_size (int): Beam size of the decoder (default=5).
    :param input_format (str): Input file format (txt or ltf, default='txt').
    :param language (str): Document language (default='english').
    """
    # set gpu device
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config = load_model(model_path, device=device, gpu=gpu,
                                          beam_size=beam_size)
    # model, tokenizer, config, _ = load_previous_model(model_path, device=device, gpu=gpu)
    # breakpoint()
    # get the list of documents
    # breakpoint()
    file_list = glob.glob(os.path.join(input_path, 'test.oneie.{}'.format(file_extension)))
    # log writer
    if log_path:
        log_writer = open(log_path, 'w', encoding='utf-8')
    # run the model; collect result and info
    doc_info_list = []
    progress = tqdm.tqdm(total=len(file_list), ncols=75)
    for f in file_list:
        progress.update(1)
        try:
            doc_result, doc_info = predict_document(
                f, model, tokenizer, config, batch_size=batch_size,
                max_length=max_length, gpu=gpu, input_format=input_format,
                language=language)
            # save json format result
            doc_id = doc_info['doc_id']
            with open(os.path.join(output_path, '{}.json'.format(doc_id)), 'w') as w:
                for sent_id, token_ids, tokens, graph in doc_result:
                    output = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'token_ids': token_ids,
                        'tokens': tokens,
                        'graph': graph.to_dict()
                    }
                    w.write(json.dumps(output) + '\n')
            # write doc info
            if log_path:
                log_writer.write(json.dumps(doc_info) + '\n')
                log_writer.flush()
        except Exception as e:
            traceback.print_exc()
            if log_path:
                log_writer.write(json.dumps(
                    {'file': file, 'message': str(e)}) + '\n')
                log_writer.flush()
    progress.close()

    # convert to the cold-start format
    if cs_path:
        print('Converting to cs format')
        json_to_cs(output_path, cs_path)


parser = ArgumentParser()
parser.add_argument('-m', '--model_path', help='path to the trained model')
parser.add_argument('-i', '--input_dir', help='path to the input folder (ltf files)')
parser.add_argument('-o', '--output_dir', help='path to the output folder (json files)')
parser.add_argument('-l', '--log_path', default=None, help='path to the log file')
parser.add_argument('-c', '--cs_dir', default=None, help='path to the output folder (cs files)')
parser.add_argument('--gpu', action='store_true', help='use gpu')
parser.add_argument('-d', '--device', default=0, type=int, help='gpu device index')
parser.add_argument('-b', '--batch_size', default=30, type=int, help='batch size')
parser.add_argument('--max_len', default=128, type=int, help='max sentence length')
parser.add_argument('--beam_size', default=5, type=int, help='beam set size')
parser.add_argument('--lang', default='english', help='Model language')
parser.add_argument('--format', default='txt', help='Input format (txt, ltf, json)')

args = parser.parse_args()
extension = format_ext_mapping.get(args.format, 'ltf.xml')

predict(
    model_path=args.model_path,
    input_path=args.input_dir,
    output_path=args.output_dir,
    cs_path=args.cs_dir,
    log_path=args.log_path,
    batch_size=args.batch_size,
    max_length=args.max_len,
    device=args.device,
    gpu=args.gpu,
    beam_size=args.beam_size,
    file_extension=extension,
    input_format=args.format,
    language=args.lang,
)