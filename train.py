import os
import json
import pdb
from textwrap import wrap
import time
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, AlbertTokenizer, AutoTokenizer, RobertaTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup)
from model import OneIE
from graph import Graph
from config import Config
from data import IEDataset,IEDatasetEval
from scorer import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task
import random
import numpy as np
# import nni
# from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
# from predict import load_model

cur_dir = os.path.dirname(os.path.realpath(__file__))

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


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True



def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reset_config(config, params):
    for param in params:
        config.param = params[param]
    return config

#===================================================================================
def encode_vocab_from_guideline(guideline_dict, tokenizer, vocabs):
    
    # entity_type_stoi = vocabs['entity_type']
    relation_type_stoi = vocabs['relation_type']
    relation_guidelines = {'piece_idx':[None]*len(relation_type_stoi),'attn_mask':[None]*len(relation_type_stoi)}
    for relation in relation_type_stoi:
        piece_idxs = tokenizer.encode(guideline_dict[relation],
                                        add_special_tokens=True,
                                        max_length=128,
                                        truncation=True)
        if len(piece_idxs)>128:
            piece_idxs = piece_idxs[:128]
            attn_mask = [1] * 128
        else:
            pad_num = 128 - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num
        relation_guidelines['piece_idx'][relation_type_stoi[relation]] = piece_idxs
        relation_guidelines['attn_mask'][relation_type_stoi[relation]] = attn_mask
        # relation_guidelines[relation_type_stoi[relation]] = ({'piece_idx':piece_idxs,'attn_mask':attn_mask})
    return relation_guidelines
#===================================================================================

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('--seed', default=1024)
args = parser.parse_args()
config = Config.from_json_file(args.config)


# breakpoint()
# params = nni.get_next_parameter()
# config = reset_config(config, params)

# print(config.to_dict())
# seed_torch(args.seed)
# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.log_path, os.path.basename(args.config).split('.')[0]+'_'+timestamp)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
writer_dir = os.mkdir(os.path.join(output_dir, 'runs'))
writer = SummaryWriter(writer_dir)
best_role_model = os.path.join(output_dir, 'best.role.mdl')
final_role_model = os.path.join(output_dir,'final.role.mdl')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')
final_dev_result_file = os.path.join(output_dir, 'final.result.dev.json')
final_test_result_file = os.path.join(output_dir, 'final.result.test.json')

# datasets
model_name = config.bert_model_name
if 'albert' in model_name:
    tokenizer = AlbertTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
elif 'scibert' in config.bert_model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
elif 'roberta' in model_name:
    tokenizer = RobertaTokenizer.from_pretrained(model_name,cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
# breakpoint()

train_set = IEDataset(config.train_file, gpu=use_gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title)
dev_set = IEDataset(config.dev_file, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
test_set = IEDataset(config.test_file, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations)    

vocabs = generate_vocabs([train_set, dev_set, test_set])


#===================================================================================
if config.use_guideliens:
    guideline_path = config.guideline_path
    try:
        guideline_dict = json.load(open(guideline_path,'r'))
    except:
        print("Can not find guideline_path!!!!!!!!!!")
        exit()
    vocab_from_guideline = encode_vocab_from_guideline(guideline_dict, tokenizer, vocabs) 
else:
    vocab_from_guideline = None
#==================================================================================

test_train = False
if not test_train:
    train_set.numberize(tokenizer, vocabs)
    dev_set.numberize(tokenizer, vocabs)
    test_set.numberize(tokenizer, vocabs)

# valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)
#---------------------------------------------------------------------------
if os.path.exists(config.valid_pattern_path):
    valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)
else:
    valid_patterns = None
#---------------------------------------------------------------------------

batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)


if test_train:
    model, tokenizer, config, vocabs = load_previous_model('log/ace05-R/sib+cop-ident2-noshare_20220417_035016/final.role.mdl', device=config.gpu_device, gpu = config.use_gpu)
    train_set.numberize(tokenizer, vocabs)
    dev_set.numberize(tokenizer, vocabs)
    test_set.numberize(tokenizer, vocabs)
    # optimizer
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and 'crf' not in n and 'global_feature' not in n],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate, 'weight_decay': 0
        }
    ]
    # breakpoint()
    optimizer = AdamW(params=param_groups)
    schedule = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=batch_num * config.warmup_epoch * config.max_epoch,
                                            num_training_steps=batch_num * config.max_epoch)
    # test_set = IEDatasetEval(config.test_file, max_length=200, gpu=use_gpu,input_format='json',)
    # test_set.numberize(tokenizer)
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        
        loss = model(batch)
        loss = loss * (1 / config.accumulate_step)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    continue
                else:
                    print(name, param.grad.sum())
        breakpoint()
        torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        # progress.update(1)
        graphs = model.predict(batch)
        if config.ignore_first_header:
            for inst_idx, sent_id in enumerate(batch.sent_ids):
                if int(sent_id.split('-')[-1]) < 4:
                    graphs[inst_idx] = Graph.empty_graph(vocabs)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        
        
        test_gold_graphs.extend(batch.graphs)
        test_pred_graphs.extend(graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)
    # save_results('test_predict.json',test_pred_graphs,test_sent_ids, test_tokens)
    # breakpoint()
    # progress.close()
    test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                               relation_directional=config.relation_directional)
    breakpoint()
else:
    # initialize the model
    model = OneIE(config, vocabs, valid_patterns, guidelines = vocab_from_guideline)
    model.load_bert(model_name, cache_dir=config.bert_cache_dir)
# model.load_ident_model(config.ident_model_path, device=config.gpu_device, gpu=config.use_gpu)
if use_gpu:
    model.cuda(device=config.gpu_device)

if config.use_guideliens:
    _ = model.guideline_encode()

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                   and 'crf' not in n and 'global_feature' not in n],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                   and ('crf' in n or 'global_feature' in n)],
        'lr': config.learning_rate, 'weight_decay': 0
    }
]
# breakpoint()
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)
# schedule = get_cosine_schedule_with_warmup(optimizer,
#                                            num_warmup_steps=batch_num * config.warmup_epoch,
#                                            num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)

global_step = 0
global_feature_max_step = int(config.global_warmup * batch_num) + 1
print('global feature max step:', global_feature_max_step)

tasks = ['entity', 'trigger', 'relation', 'role','relation+']
best_dev = {k: 0 for k in tasks}

idx = 1
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))
    total_loss = 0
    # training set
    # progress = tqdm.tqdm(total=batch_num, ncols=75,
    #                      desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    batch_time = []
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        # breakpoint()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,) as prof:
        #     with record_function("model"):
        start_time = time.time()
        loss = model(batch)
        # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=50))
        # breakpoint()
        loss = loss * (1 / config.accumulate_step)
        # print(loss)
        loss.backward()
        end_time = time.time()
        batch_time.append(end_time-start_time)
        # breakpoint()
        total_loss += loss.item()

        if (batch_idx + 1) % config.accumulate_step == 0:
            # progress.update(1)
            global_step += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         try:
            #             # breakpoint()
            #             writer.add_scalar(name, param.grad.sum(), epoch)
            #             # print(name, param.grad.sum())  
            #         except:
            #             print(name,'~~~~~~~~~~~')
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             continue
            #         else:
            #             writer.add_scalar(name, param.grad.sum(), idx)
            #             idx += 1
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    # train_speed = 10/(sum(batch_time[2:])/(len(batch_time)-2))
    # breakpoint()
    # writer.add_scalar('train_loss', total_loss, epoch)
    # try:
    #     writer.add_histogram('le_potential', model.role_entity_potential, epoch)
    # except:
    #     pass
    # try:
    #     writer.add_histogram('tle_potential', model.event_role_entity_potential, epoch)
    # except:
    #     pass
    # try:
    #     # breakpoint()
    # breakpoint()
    # writer.add_histogram('relation_unary_score', model.debug['relation_unary'], epoch)
    # writer.add_histogram('unary_rel_type_reprs', model.unary_relation_type_reps, epoch)
    # writer.add_histogram('Gradient/relation_type_reps', model.unary_relation_type_reps.grad.data, epoch)
    # writer.add_histogram('Gradient/start_ent_reps', model.start_entity_ffn.weight.grad.data, epoch)
    # writer.add_histogram('Gradient/end_ent_reps', model.end_entity_ffn.weight.grad.data, epoch)
    # if model.debug:
    #     writer.add_histogram('binary_rel_type_reprs', model.debug['relation_reps'], epoch)
    #     writer.add_histogram('sib_message_1', model.debug['sib_message_1'], epoch)
    #     writer.add_histogram('sib_message_2', model.debug['sib_message_2'], epoch)
    #     writer.add_histogram('sib_message_3', model.debug['sib_message_3'], epoch)
    #     writer.add_histogram('cop_message_1', model.debug['cop_message_1'], epoch)
    #     writer.add_histogram('cop_message_2', model.debug['cop_message_2'], epoch)
    #     writer.add_histogram('cop_message_3', model.debug['cop_message_3'], epoch)
    # except:
    #     pass
    # progress.close()
    print("total_loss:{}".format(total_loss))
    # dev set
    # progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
    #                      desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        # progress.update(1)
        graphs = model.predict(batch)
        if config.ignore_first_header:
            for inst_idx, sent_id in enumerate(batch.sent_ids):
                if int(sent_id.split('-')[-1]) < 4:
                    graphs[inst_idx] = Graph.empty_graph(vocabs)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        
        dev_gold_graphs.extend(batch.graphs)
        dev_pred_graphs.extend(graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    # progress.close()
    # breakpoint()
    dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                              relation_directional=config.relation_directional)

    # if dev_scores['relation']['f']*100 < 10:
    #     torch.save(state, os.path.join(output_dir, '{}.role.mdl'.format(epoch)))
    
    # if dev_scores['relation']['f']*100 < 1:
    #     exit()  
    
    for task in tasks:
        if dev_scores[task]['f'] > best_dev[task]:
            best_dev[task] = dev_scores[task]['f']
            if 'ace05-R' in config.log_path or 'scierc' in config.log_path:
                if task == 'relation':
                    print('Saving best role model')
                    torch.save(state, best_role_model)
                    best_dev_role_model = True
                    # breakpoint()
                    save_result(dev_result_file,
                                dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                                dev_tokens)
                    # dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                    #           relation_directional=config.relation_directional)
            else:
                if task == 'role':
                    print('Saving best role model')
                    torch.save(state, best_role_model)
                    best_dev_role_model = True
                    save_result(dev_result_file,
                                dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                                dev_tokens)

    # test set
    # progress = tqdm.tqdm(total=test_batch_num, ncols=75,
    #                      desc='Test {}'.format(epoch))
    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        # progress.update(1)
        graphs = model.predict(batch)
        if config.ignore_first_header:
            for inst_idx, sent_id in enumerate(batch.sent_ids):
                if int(sent_id.split('-')[-1]) < 4:
                    graphs[inst_idx] = Graph.empty_graph(vocabs)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        test_gold_graphs.extend(batch.graphs)
        test_pred_graphs.extend(graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)
    # progress.close()
    test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                               relation_directional=config.relation_directional)

    # nni.report_intermediate_result(dev_scores['role']['f'])

    if best_dev_role_model:
        save_result(test_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)

    result = json.dumps(
        {'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
        if best_dev_role_model:
            w.write(result + '\n')
    print('Log file', log_file)


# nni.report_final_result(test_scores['role']['f'])


torch.save(state, final_role_model)
save_result(final_dev_result_file,dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)
save_result(final_test_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)
if 'ace05-R' or 'scierc' in config.log_path:
    best_score_by_task(log_file, 'relation')
else:
    best_score_by_task(log_file, 'role')
