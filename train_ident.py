import os
import json
import pdb
import time
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, RobertaTokenizer, AlbertTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from model import OneIE, Ident
from graph import Graph
from config import Config
from data import IEDataset
from scorer import score_graphs, score_ident
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('--seed', default=1234)
args = parser.parse_args()
config = Config.from_json_file(args.config)
# print(config.to_dict())
# set_seed(args.seed)
# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.log_path, os.path.basename(args.config).split('.')[0]+'_'+timestamp)
# breakpoint()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_ident_model = os.path.join(output_dir, 'best.ident.mdl')
final_ident_model = os.path.join(output_dir,'final.ident.mdl')
# dev_result_file = os.path.join(output_dir, 'result.dev.json')
# test_result_file = os.path.join(output_dir, 'result.test.json')
# final_dev_result_file = os.path.join(output_dir, 'final.result.dev.json')
# final_test_result_file = os.path.join(output_dir, 'final.result.test.json')

# datasets
model_name = config.bert_model_name
if 'albert' in model_name:
    tokenizer = AlbertTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
elif 'roberta' in model_name:
    tokenizer = RobertaTokenizer.from_pretrained(model_name,cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
elif 'scibert' in config.bert_model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False)
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

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)


batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = Ident(config, vocabs)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
if use_gpu:
    model.cuda(device=config.gpu_device)

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
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs)

global_step = 0
global_feature_max_step = int(config.global_warmup * batch_num) + 1
print('global feature max step:', global_feature_max_step)

tasks = ['entity', 'trigger']
best_dev = {k: 0 for k in tasks}
best_avg_dev = 0
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # training set
    # progress = tqdm.tqdm(total=batch_num, ncols=75,
    #                      desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):
        # pdb.set_trace()
        loss = model(batch)
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            # progress.update(1)
            global_step += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    # progress.close()

    # dev set
    # progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
    #                      desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    # if epoch==50:
    #     breakpoint()
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        # progress.update(1)
        graphs = model.predict(batch)
        # if config.ignore_first_header:
        #     for inst_idx, sent_id in enumerate(batch.sent_ids):
        #         if int(sent_id.split('-')[-1]) < 4:
        #             graphs[inst_idx] = Graph.empty_graph(vocabs)
        # for graph in graphs:
        #     graph.clean(relation_directional=config.relation_directional,
        #                 symmetric_relations=config.symmetric_relations)
        # breakpoint()
        dev_gold_graphs.extend(batch.graphs)
        dev_pred_graphs.extend(graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    # progress.close()
    dev_scores = score_ident(dev_gold_graphs, dev_pred_graphs)

    # breakpoint()
    # current_avg_dev = (dev_scores['entity_id'] + dev_scores['trigger_id'])/2
    if 'ace05-R' in config.log_path or 'scierc' in config.log_path:
        current_avg_dev = dev_scores['entity_id']
    else:
        current_avg_dev = dev_scores['trigger_id']
    if current_avg_dev > best_avg_dev:
        best_avg_dev = current_avg_dev
        print('Saving best role model')
        torch.save(state, best_ident_model)


    # test set
    # progress = tqdm.tqdm(total=test_batch_num, ncols=75,
    #                      desc='Test {}'.format(epoch))
    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        # progress.update(1)
        graphs = model.predict(batch)

        test_gold_graphs.extend(batch.graphs)
        test_pred_graphs.extend(graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)
    # progress.close()
    test_scores = score_ident(test_gold_graphs, test_pred_graphs)

    # if best_dev_role_model:
    #     save_result(test_result_file, test_gold_graphs, test_pred_graphs,
    #                 test_sent_ids, test_tokens)

    result = json.dumps(
        {'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
        if best_dev_role_model:
            w.write(result + '\n')
    print('Log file', log_file)

torch.save(state, final_ident_model)

# best_score_by_task(log_file, 'role')
