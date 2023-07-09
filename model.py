import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from graph import Graph
from transformers import BertModel, AutoModel,AlbertModel,RobertaModel
from global_feature import generate_global_feature_vector, generate_global_feature_maps
from util import normalize_score
from opt_einsum import contract
import os
from config import Config
from collections import Counter, namedtuple, defaultdict
from data import Batch, Instance
from torch.nn.parameter import Parameter
from sparsemax import Sparsemax

cur_dir = os.path.dirname(os.path.realpath(__file__))

def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets


def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def graphs_to_node_idxs(graphs):
    """
    :param graphs (list): A list of Graph objects.
    :return: entity/trigger index matrix, mask tensor, max number, and max length
    """
    entity_idxs, entity_masks = [], []
    trigger_idxs, trigger_masks = [], []
    # 
    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_entity_len = max(max([e[1] - e[0] for e in graph.entities] + [1])
                         for graph in graphs)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)
    for graph in graphs:
        seq_entity_idxs, seq_entity_masks = [], []
        seq_trigger_idxs, seq_trigger_masks = [], []
        for entity in graph.entities:
            entity_len = entity[1] - entity[0]
            seq_entity_idxs.extend([i for i in range(entity[0], entity[1])])
            seq_entity_idxs.extend([0] * (max_entity_len - entity_len))
            seq_entity_masks.extend([1.0 / entity_len] * entity_len)
            seq_entity_masks.extend([0.0] * (max_entity_len - entity_len))
        seq_entity_idxs.extend([0] * max_entity_len * (max_entity_num - graph.entity_num))
        seq_entity_masks.extend([0.0] * max_entity_len * (max_entity_num - graph.entity_num))
        entity_idxs.append(seq_entity_idxs)
        entity_masks.append(seq_entity_masks)

        for trigger in graph.triggers:
            trigger_len = trigger[1] - trigger[0]
            seq_trigger_idxs.extend([i for i in range(trigger[0], trigger[1])])
            seq_trigger_idxs.extend([0] * (max_trigger_len - trigger_len))
            seq_trigger_masks.extend([1.0 / trigger_len] * trigger_len)
            seq_trigger_masks.extend([0.0] * (max_trigger_len - trigger_len))
        seq_trigger_idxs.extend([0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        seq_trigger_masks.extend([0.0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        trigger_idxs.append(seq_trigger_idxs)
        trigger_masks.append(seq_trigger_masks)

    return (
        entity_idxs, entity_masks, max_entity_num, max_entity_len,
        trigger_idxs, trigger_masks, max_trigger_num, max_trigger_len,
    )


def graphs_to_label_idxs(graphs, max_entity_num=-1, max_trigger_num=-1,
                         relation_directional=False,
                         symmetric_relation_idxs=None):
    """Convert a list of graphs to label index and mask matrices
    :param graphs (list): A list of Graph objects.
    :param max_entity_num (int) Max entity number (default = -1).
    :param max_trigger_num (int) Max trigger number (default = -1).
    """
    if max_entity_num == -1:
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
    if max_trigger_num == -1:
        max_trigger_num = max(max([g.trigger_num for g in graphs]), 1)
    (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    ) = [[] for _ in range(8)]
    for graph in graphs:
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask,
        ) = graph.to_label_idxs(max_entity_num, max_trigger_num,
                                relation_directional=relation_directional,
                                symmetric_relation_idxs=symmetric_relation_idxs)
        batch_entity_idxs.append(entity_idxs)
        batch_entity_mask.append(entity_mask)
        batch_trigger_idxs.append(trigger_idxs)
        batch_trigger_mask.append(trigger_mask)
        batch_relation_idxs.append(relation_idxs)
        batch_relation_mask.append(relation_mask)
        batch_role_idxs.append(role_idxs)
        batch_role_mask.append(role_mask)
    return (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    )


def generate_pairwise_idxs(num1, num2):
    """Generate all pairwise combinations among entity mentions (relation) or
    event triggers and entity mentions (argument role).

    For example, if there are 2 triggers and 3 mentions in a sentence, num1 = 2,
    and num2 = 3. We generate the following vector:

    idxs = [0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4]

    Suppose `trigger_reprs` and `entity_reprs` are trigger/entity representation
    tensors. We concatenate them using:

    te_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)

    After that we select vectors from `te_reprs` using (incomplete code) to obtain
    pairwise combinations of all trigger and entity vectors.

    te_reprs = torch.gather(te_reprs, 1, idxs)
    te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)

    :param num1: trigger number (argument role) or entity number (relation)
    :param num2: entity number (relation)
    :return (list): a list of indices
    """
    idxs = []
    for i in range(num1):
        for j in range(num2):
            idxs.append(i)
            idxs.append(j + num1)
    return idxs


def tag_paths_to_spans(paths, token_nums, vocab):
    """Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)

            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        batch_mentions.append(mentions)
    return batch_mentions


def remove_overlap_entities(gold_entities, wrong_entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []

    for entity in gold_entities:
        start, end = entity[0], entity[1]
        # for i in range(start, end):
        #     if tokens[i]:
        #         continue
        # entities_.append(entity)
        for i in range(start, end):
            # 
            tokens[i] = 1
    for wrong_entity in wrong_entities:
        start, end = wrong_entity[0], wrong_entity[1]
        overlap = False
        for i in range(start, end):
            if tokens[i]:
                overlap = True
                break
        if not overlap:
            entities_.append(wrong_entity)
    return entities_


class High_Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        # self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                # inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs

#==============================================================================
class Guide_Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            inputs = self.activation(inputs)
            inputs = self.dropout(inputs)
        return inputs
#==============================================================================


class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        if activation == "":
            self.activation = None 
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:  
            self.activation = getattr(torch, activation)
        
        # self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        # breakpoint()
        for i, layer in enumerate(self.layers):
            if i > 0:
                if self.activation:
                    inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        # self.same_type = self.map_same_types()
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)

        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        # 
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        # 
        return norm

    def loglik(self, logits, labels, lens):
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores


class OneIE(nn.Module):
   
   
    def __init__(self,
                 config,
                 vocabs,
                 valid_patterns=None, guidelines = None,):
        super().__init__()

        self.test_potential = []

        # jointly training identification and classification or splitly training
        self.split_train = config.split_train

        # vocabularies
        self.config = config
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.trigger_label_stoi = vocabs['trigger_label']
        self.mention_type_stoi = vocabs['mention_type']
        self.entity_type_stoi = vocabs['entity_type']
        self.event_type_stoi = vocabs['event_type']
        self.relation_type_stoi = vocabs['relation_type']
        self.role_type_stoi = vocabs['role_type']
        self.entity_label_itos = {i:s for s, i in self.entity_label_stoi.items()}
        self.trigger_label_itos = {i:s for s, i in self.trigger_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        self.event_type_itos = {i: s for s, i in self.event_type_stoi.items()}
        self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.trigger_label_num = len(self.trigger_label_stoi)
        self.mention_type_num = len(self.mention_type_stoi)
        self.entity_type_num = len(self.entity_type_stoi)
        self.event_type_num = len(self.event_type_stoi)
        self.relation_type_num = len(self.relation_type_stoi)
        self.role_type_num = len(self.role_type_stoi)
        self.valid_relation_entity = set()
        self.valid_event_role = set()
        self.valid_role_entity = set()
        if valid_patterns:
            self.valid_event_role = valid_patterns['event_role']
            self.valid_relation_entity = valid_patterns['relation_entity']
            self.valid_role_entity = valid_patterns['role_entity']
            try:
                self.valid_relation_start_entity = valid_patterns['relation_start_entity']
                self.valid_relation_end_entity = valid_patterns['relation_end_entity']
            except:
                pass
            # ------------------------------------------------------------------------------
            # if config.use_high_order_tre:
            #     self.tre_valid_pattern_mask = self.event_role_entity_factor_mask()
                # 
            # ------------------------------------------------------------------------------
        self.relation_directional = config.relation_directional
        self.symmetric_relations = config.symmetric_relations
        self.symmetric_relation_idxs = {self.relation_type_stoi[r]
                                        for r in self.symmetric_relations}

        # BERT encoder
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        if 'albert' in config.bert_model_name:
            self.bert = AlbertModel(bert_config)
        elif 'roberta' in config.bert_model_name:
            self.bert = RobertaModel(bert_config)
        elif 'scibert' in config.bert_model_name:
            self.bert = BertModel(bert_config)
        else:
            self.bert = BertModel(bert_config)
        # self.bert = BertModel(bert_config)
        # breakpoint()
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        # local classifiers
        self.use_entity_type = config.use_entity_type
        self.binary_dim = self.bert_dim * 2
        linear_bias = config.linear_bias
        linear_dropout = config.linear_dropout
        entity_hidden_num = config.entity_hidden_num
        mention_hidden_num = config.mention_hidden_num
        event_hidden_num = config.event_hidden_num
        relation_hidden_num = config.relation_hidden_num
        role_hidden_num = config.role_hidden_num
        role_input_dim = self.binary_dim + (self.entity_type_num if self.use_entity_type else 0)


        # self.entity_layer_norm = nn.LayerNorm(self.bert_dim)
        self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num,
                                        bias=linear_bias)
        self.trigger_label_ffn = nn.Linear(self.bert_dim, self.trigger_label_num,
                                         bias=linear_bias)

        try:
            self.entity_hidden_size = config.entity_hidden_size
        except:
            self.entity_hidden_size = self.bert_dim

        if self.config.new_score:
            self.unary_entity_type_reps = Parameter(torch.empty(self.entity_type_num, entity_hidden_num))
            torch.nn.init.kaiming_uniform_(self.unary_entity_type_reps, a=math.sqrt(5))
            self.entity_type_ffn = nn.Linear(self.bert_dim, entity_hidden_num)
            self.linear_entity_dropout = nn.Dropout(p=config.linear_dropout)
        else:
            self.entity_type_ffn = Linears([self.bert_dim, entity_hidden_num,
                                        self.entity_type_num],
                                       dropout_prob=linear_dropout,
                                       bias=linear_bias,
                                       activation=config.linear_activation)
        self.mention_type_ffn = Linears([self.bert_dim, mention_hidden_num,
                                         self.mention_type_num],
                                        dropout_prob=linear_dropout,
                                        bias=linear_bias,
                                        activation=config.linear_activation)
        #===============================================================================================
        if self.config.new_score:
            self.unary_trigger_type_reps = Parameter(torch.empty(self.event_type_num, event_hidden_num))
            torch.nn.init.kaiming_uniform_(self.unary_trigger_type_reps, a=math.sqrt(5))
            self.event_type_ffn = nn.Linear(self.bert_dim, event_hidden_num)
            self.linear_trigger_dropout = nn.Dropout(p=config.linear_dropout)
        else:
            self.event_type_ffn = Linears([self.bert_dim, event_hidden_num,
                                       self.event_type_num],
                                      dropout_prob=linear_dropout,
                                      bias=linear_bias,
                                      activation=config.linear_activation)
        #===============================================================================================
        self.start_entity_ffn = nn.Linear(self.bert_dim, self.entity_hidden_size)
        self.end_entity_ffn = nn.Linear(self.bert_dim, self.entity_hidden_size)
        
        if config.split_rel_ident:
            self.start_entity_ident_ffn = nn.Linear(self.bert_dim, self.bert_dim)
            self.end_entity_ident_ffn = nn.Linear(self.bert_dim, self.bert_dim)
            self.relation_ident_ffn = Linears([self.binary_dim, relation_hidden_num,2],
                                         dropout_prob=linear_dropout,
                                         bias=linear_bias,
                                         activation=config.linear_activation)
        


        if self.config.new_score:
            self.unary_relation_type_reps = Parameter(torch.empty(self.relation_type_num, self.entity_hidden_size))
            torch.nn.init.kaiming_uniform_(self.unary_relation_type_reps, a=math.sqrt(5))
            self.linear_start_dropout = nn.Dropout(p=config.linear_dropout)
            self.linear_end_dropout = nn.Dropout(p=config.linear_dropout)
        else:
            self.relation_type_ffn = Linears([self.binary_dim, relation_hidden_num,
                                          self.relation_type_num],
                                         dropout_prob=linear_dropout,
                                         bias=linear_bias,
                                         activation=config.linear_activation)
        #===============================================================================
        self.use_guideliens = config.use_guideliens
        self.relation_type_reprs = None
        if config.use_guideliens:
            self.guideline_piexs = torch.tensor(guidelines['piece_idx'])
            self.guideline_attn_masks = torch.tensor(guidelines['attn_mask'])
            self.entity_ffn = Guide_Linears([self.binary_dim, relation_hidden_num],
                                                dropout_prob=linear_dropout,
                                                bias=linear_bias,
                                                activation=config.linear_activation)
            self.relation_ffn = Guide_Linears([self.bert_dim, relation_hidden_num],
                                                dropout_prob=linear_dropout,
                                                bias=linear_bias,
                                                activation=config.linear_activation)
        #===============================================================================
        
        if self.config.new_score:
            self.unary_role_type_reps = Parameter(torch.empty(self.role_type_num, role_hidden_num))
            torch.nn.init.kaiming_uniform_(self.unary_role_type_reps, a=math.sqrt(5))
            self.unary_trigger_ffn = nn.Linear(self.bert_dim, role_hidden_num)
            argument_dim = self.bert_dim + (self.entity_type_num if self.use_entity_type else 0)
            self.unary_argument_ffn = nn.Linear(argument_dim, role_hidden_num)
            self.unary_trigger_dropout = nn.Dropout(p=config.linear_dropout)
            self.unary_argument_dropout = nn.Dropout(p=config.linear_dropout)

        else:
            self.role_type_ffn = Linears([role_input_dim, role_hidden_num,
                                      self.role_type_num],
                                     dropout_prob=linear_dropout,
                                     bias=linear_bias,
                                     activation=config.linear_activation)
        # global features
        self.use_global_features = config.use_global_features
        #------------------------------------------------------------------------------
        if self.use_global_features:
            self.global_features = config.global_features
            self.global_feature_maps = generate_global_feature_maps(vocabs, valid_patterns)
            self.global_feature_num = sum(len(m) for k, m in self.global_feature_maps.items()
                                          if k in self.global_features or
                                          not self.global_features)
            # 
            self.global_feature_weights = nn.Parameter(
                torch.zeros(self.global_feature_num).fill_(-0.0001))
        #------------------------------------------------------------------------------

        # decoder
        self.beam_size = config.beam_size
        self.beta_v = config.beta_v
        self.beta_e = config.beta_e
        # loss functions
        self.entity_criteria = torch.nn.CrossEntropyLoss()
        self.event_criteria = torch.nn.CrossEntropyLoss()
        self.mention_criteria = torch.nn.CrossEntropyLoss()
        self.relation_criteria = torch.nn.CrossEntropyLoss()
        self.role_criteria = torch.nn.CrossEntropyLoss()
        if config.split_rel_ident:
            self.relation_ident_criteria = torch.nn.CrossEntropyLoss()
        # others
        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)
        self.trigger_crf = CRF(self.trigger_label_stoi, bioes=False)
        self.pad_vector = nn.Parameter(torch.randn(1, 1, self.bert_dim))

        # -------------------------------------------------------------------------------
        # high-order
        # self.use_high_order = config.use_high_order
        self.use_high_order_tl = config.use_high_order_tl
        self.use_high_order_le = config.use_high_order_le
        self.use_high_order_tre = config.use_high_order_tre
        self.use_high_order_sibling = config.use_high_order_sibling
        self.use_high_order_coparent = config.use_high_order_coparent
        self.use_high_order_ere = config.use_high_order_ere
        self.use_high_order_er = config.use_high_order_er
        self.use_high_order_re_sibling = config.use_high_order_re_sibling
        self.use_high_order_re_coparent = config.use_high_order_re_coparent
        self.use_high_order_re_grandparent = config.use_high_order_re_grandparent
        self.use_high_order_rr_coparent = config.use_high_order_rr_coparent
        self.use_high_order_rr_grandparent = config.use_high_order_rr_grandparent
        self.decomp_size = config.decomp_size
        self.tre_decomp_size = config.tre_decomp_size
        if self.use_high_order_tl:
            self.trigger_tl_W = High_Linears([self.bert_dim, event_hidden_num,
                                       self.decomp_size],dropout_prob=linear_dropout,
                                       bias=linear_bias)
            self.entity_tl_W = High_Linears([self.bert_dim, event_hidden_num,
                                       self.decomp_size],dropout_prob=linear_dropout,
                                       bias=linear_bias)
            # self.trigger_tl_W = nn.Linear(self.bert_dim, self.decomp_size,bias=False)
            # self.entity_tl_W = nn.Linear(self.bert_dim, self.decomp_size, bias=False)
            self.event_type_tl_W = Parameter(torch.empty(self.event_type_num, self.decomp_size))
            self.role_type_tl_W = Parameter(torch.empty(self.role_type_num, self.decomp_size))
            torch.nn.init.kaiming_uniform_(self.event_type_tl_W, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.role_type_tl_W, a=math.sqrt(5))
        if self.use_high_order_le:
            if self.config.new_potential:
                self.role_entity_potential = Parameter(torch.empty(self.role_type_num, self.entity_type_num))
                # torch.nn.init.uniform_(self.role_entity_potential, a=0,b=1)
                torch.nn.init.kaiming_uniform_(self.role_entity_potential, a=math.sqrt(5))
            else:
                self.trigger_le_W = High_Linears([self.bert_dim, event_hidden_num,
                                        self.decomp_size],dropout_prob=linear_dropout,
                                        bias=linear_bias)
                self.entity_le_W = High_Linears([self.bert_dim, event_hidden_num,
                                        self.decomp_size],dropout_prob=linear_dropout,
                                        bias=linear_bias)
                # self.trigger_le_W = nn.Linear(self.bert_dim, self.decomp_size,bias=False)
                # self.entity_le_W = nn.Linear(self.bert_dim, self.decomp_size, bias=False)
                self.role_type_le_W = Parameter(torch.empty(self.role_type_num, self.decomp_size))
                self.entity_type_le_W = Parameter(torch.empty(self.entity_type_num, self.decomp_size))
                torch.nn.init.kaiming_uniform_(self.role_type_le_W, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.entity_type_le_W, a=math.sqrt(5))
        if self.use_high_order_tre:
            if self.config.new_potential:
                self.event_role_entity_potential = Parameter(torch.empty(self.event_type_num, self.role_type_num, self.entity_type_num))
                # torch.nn.init.uniform_(self.event_role_entity_potential, a=0,b=1)
                torch.nn.init.kaiming_uniform_(self.event_role_entity_potential, a=math.sqrt(5))
            else:
                self.trigger_tre_W = nn.Linear(self.bert_dim, self.tre_decomp_size,bias=True)
                self.trigger_tre_dropout = nn.Dropout(p=config.linear_dropout)
                self.entity_tre_W = nn.Linear(self.bert_dim, self.tre_decomp_size,bias=True)
                self.entity_tre_dropout = nn.Dropout(p=config.linear_dropout)
                self.event_type_W = Parameter(torch.empty(self.event_type_num-1, self.tre_decomp_size))
                self.role_type_W = Parameter(torch.empty(self.role_type_num, self.tre_decomp_size))
                self.entity_type_W = Parameter(torch.empty(self.entity_type_num-1, self.tre_decomp_size))
                torch.nn.init.kaiming_uniform_(self.event_type_W, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.role_type_W, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.entity_type_W, a=math.sqrt(5))
        if self.use_high_order_sibling:
            self.trigger_sib_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
            self.trigger_sib_dropout = nn.Dropout(p=config.linear_dropout)
            self.entity_sib_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
            self.entity_sib_dropout = nn.Dropout(p=config.linear_dropout)
            self.role_type_sib_W = Parameter(torch.empty(self.role_type_num, self.decomp_size))
            torch.nn.init.kaiming_uniform_(self.role_type_sib_W, a=math.sqrt(5))
        if self.use_high_order_coparent:
            self.trigger_cop_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
            self.trigger_cop_dropout = nn.Dropout(p=config.linear_dropout)
            self.entity_cop_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
            self.entity_cop_dropout = nn.Dropout(p=config.linear_dropout)
            self.role_type_cop_W = Parameter(torch.empty(self.role_type_num, self.decomp_size))
            torch.nn.init.kaiming_uniform_(self.role_type_cop_W, a=math.sqrt(5))
        
        if self.use_high_order_ere:
            if self.config.new_potential:
                if self.config.share_relation_type_reps:
                    self.relation_type_ere_W = nn.Linear(self.entity_hidden_size, self.decomp_size, bias=True)
                    self.entity_type_ere_W = nn.Linear(entity_hidden_num, self.decomp_size, bias=True)
                else:
                    self.entity_type_ere_W = Parameter(torch.empty(self.entity_type_num-1, self.decomp_size))
                    self.relation_type_ere_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
                    torch.nn.init.kaiming_uniform_(self.entity_type_ere_W, a=math.sqrt(5))
                    torch.nn.init.kaiming_uniform_(self.relation_type_ere_W, a=math.sqrt(5))
            else:
                self.entity_start_ere_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
                self.entity_end_ere_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
                self.ere_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.ere_end_dropout = nn.Dropout(p=config.linear_dropout)
                if self.config.share_relation_type_reps:
                    self.relation_type_ere_W = nn.Linear(self.entity_hidden_size, self.decomp_size, bias=True)
                    self.entity_type_ere_W = nn.Linear(entity_hidden_num, self.decomp_size, bias=True)
                else:
                    self.entity_type_ere_W = Parameter(torch.empty(self.entity_type_num-1, self.decomp_size))
                    self.relation_type_ere_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
                    torch.nn.init.kaiming_uniform_(self.entity_type_ere_W, a=math.sqrt(5))
                    torch.nn.init.kaiming_uniform_(self.relation_type_ere_W, a=math.sqrt(5))
        
        if self.use_high_order_er:
            if self.config.new_potential:
                self.entity_relation_potential = Parameter(torch.empty(self.entity_type_num-1, self.relation_type_num))
                self.relation_entity_potential = Parameter(torch.empty(self.relation_type_num, self.entity_type_num-1))
                torch.nn.init.kaiming_uniform_(self.entity_relation_potential, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.relation_entity_potential, a=math.sqrt(5))
            else:
                self.start_entity_er_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
                self.end_entity_er_W = nn.Linear(self.bert_dim, self.decomp_size,bias=True)
                self.er_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.er_end_dropout = nn.Dropout(p=config.linear_dropout)
                self.entity_type_er_W = Parameter(torch.empty(self.entity_type_num-1, self.decomp_size))
                if self.config.share_relation_type_reps:
                    self.relation_type_er_W = nn.Linear(self.entity_hidden_size, self.decomp_size, bias=True)
                else:
                    self.relation_type_er_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
                    # self.er_rel_type_dropout = nn.Dropout(p=config.linear_dropout)
                    torch.nn.init.kaiming_uniform_(self.relation_type_er_W, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.entity_type_er_W, a=math.sqrt(5))           
        
        if self.use_high_order_re_sibling:

            if self.config.decomp:
                self.entity_start_re_sib_W = nn.Linear(self.bert_dim, self.config.entity_hidden_size,bias=True)
                self.entity_end_re_sib_W = nn.Linear(self.bert_dim, self.entity_hidden_size,bias=True)
                self.sib_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.sib_end_dropout = nn.Dropout(p=config.linear_dropout)
                
                self.entity_start_re_sib_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)
                self.entity_end_re_sib_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)
            else:
                self.entity_start_re_sib_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.entity_end_re_sib_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.sib_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.sib_end_dropout = nn.Dropout(p=config.linear_dropout)

            if self.config.share_relation_type_reps:
                self.relation_type_re_sib_W = nn.Linear(self.entity_hidden_size, self.decomp_size, bias=False)
            else:
                self.relation_type_re_sib_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
                torch.nn.init.kaiming_uniform_(self.relation_type_re_sib_W, a=math.sqrt(5))
                #---------------------------------------------------------------------------
                # self.relation_type_sib_dropout = nn.Dropout(p=config.linear_dropout)
                #---------------------------------------------------------------------------
        
        if self.use_high_order_re_coparent:

            if self.config.decomp:
                self.entity_start_re_cop_W = nn.Linear(self.bert_dim, self.entity_hidden_size,bias=True)
                self.entity_end_re_cop_W = nn.Linear(self.bert_dim, self.entity_hidden_size,bias=True)
                self.cop_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.cop_end_dropout = nn.Dropout(p=config.linear_dropout)

                self.entity_start_re_cop_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)
                self.entity_end_re_cop_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)
            else:
                self.entity_start_re_cop_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.entity_end_re_cop_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.cop_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.cop_end_dropout = nn.Dropout(p=config.linear_dropout)

            if self.config.share_relation_type_reps:
                self.relation_type_re_cop_W = nn.Linear(self.entity_hidden_size, self.decomp_size, bias=False)
            else:
                self.relation_type_re_cop_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
                torch.nn.init.kaiming_uniform_(self.relation_type_re_cop_W, a=math.sqrt(5))
        
        if self.use_high_order_re_grandparent:

            if self.config.decomp:
                self.entity_start_re_gp_W = nn.Linear(self.bert_dim, self.entity_hidden_size,bias=True)
                self.entity_mid_re_gp_W = nn.Linear(self.bert_dim, self.entity_hidden_size,bias=True)
                self.entity_end_re_gp_W = nn.Linear(self.bert_dim, self.entity_hidden_size,bias=True)
                self.gp_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.gp_mid_dropout = nn.Dropout(p=config.linear_dropout)
                self.gp_end_dropout = nn.Dropout(p=config.linear_dropout)

                self.entity_start_re_gp_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)
                self.entity_mid_re_gp_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)            
                self.entity_end_re_gp_decomp_ffn = nn.Linear(self.config.entity_hidden_size,self.config.decomp_size,bias=False)
            else:
                self.entity_start_re_gp_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.entity_mid_re_gp_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.entity_end_re_gp_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
                self.gp_start_dropout = nn.Dropout(p=config.linear_dropout)
                self.gp_mid_dropout = nn.Dropout(p=config.linear_dropout)
                self.gp_end_dropout = nn.Dropout(p=config.linear_dropout)
           

            if self.config.share_relation_type_reps:
                self.relation_type_re_gp_W = nn.Linear(self.entity_hidden_size, self.decomp_size, bias=False)
            else:
                self.relation_type_re_gp_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
                torch.nn.init.kaiming_uniform_(self.relation_type_re_gp_W, a=math.sqrt(5))
            
        if self.use_high_order_rr_coparent:
            self.trigger_start_re_cop_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
            self.entity_start_re_cop_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
            self.entity_end_re_cop_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
            self.cop_trigger_start_dropout = nn.Dropout(p=config.linear_dropout)
            self.cop_start_dropout = nn.Dropout(p=config.linear_dropout)
            self.cop_end_dropout = nn.Dropout(p=config.linear_dropout)
            self.role_type_re_cop_W = Parameter(torch.empty(self.role_type_num, self.decomp_size))
            self.relation_type_re_cop_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
            torch.nn.init.kaiming_uniform_(self.role_type_re_cop_W, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.relation_type_re_cop_W, a=math.sqrt(5))
        
        if self.use_high_order_rr_grandparent:
            self.trigger_start_re_gp_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
            self.entity_start_re_gp_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
            self.entity_end_re_gp_W = nn.Linear(self.bert_dim, self.config.decomp_size,bias=True)
            self.gp_trigger_start_dropout = nn.Dropout(p=config.linear_dropout)
            self.gp_start_dropout = nn.Dropout(p=config.linear_dropout)
            self.gp_end_dropout = nn.Dropout(p=config.linear_dropout)
            self.role_type_re_gp_W = Parameter(torch.empty(self.role_type_num, self.decomp_size))
            self.relation_type_re_gp_W = Parameter(torch.empty(self.relation_type_num, self.decomp_size))
            torch.nn.init.kaiming_uniform_(self.role_type_re_gp_W, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.relation_type_re_gp_W, a=math.sqrt(5))

        if self.config.train_alpha:
            # breakpoint()
            self.alpha_role_tre = nn.Parameter(torch.tensor([1.]))
            self.alpha_role_sib = nn.Parameter(torch.tensor([1.]))
            self.alpha_entity_tre = nn.Parameter(torch.tensor([1.]))
            self.alpha_event_tre = nn.Parameter(torch.tensor([1.]))
        else:
            self.alpha_role_tre = self.config.alpha_role_tre
            self.alpha_role_sib = self.config.alpha_role_sib
            self.alpha_entity_tre = self.config.alpha_entity_tre
            self.alpha_event_tre = self.config.alpha_event_tre

        
        # -------------------------------------------------------------------------------
        if self.split_train:
            # breakpoint()
            self.load_ident_model(config.ident_model_path, device=config.gpu_device, gpu=config.use_gpu)
        
        self.debug = {}


    def load_ident_model(self, model_path, device=0, gpu=False):
        print('Loading the model from {}'.format(model_path))
        map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
        state = torch.load(model_path, map_location=map_location)

        config = state['config']
        if type(config) is dict:
            config = Config.from_dict(config)
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # config.bert_cache_dir = os.path.join(cur_dir, 'bert')
        vocabs = state['vocabs']

        # recover the model
        self.ident_model = Ident(config, vocabs)
        self.ident_model.load_state_dict(state['model'], False)
        if gpu:
            self.ident_model.cuda(device)


    def load_bert(self, name, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('Loading pre-trained BERT model {}'.format(name))
        if 'scibert' in name:
            self.bert = AutoModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                                )
        elif 'albert' in name:
            # breakpoint()
            self.bert = AlbertModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                                )
        elif 'roberta' in name:
            self.bert = RobertaModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                                )
        else:
            self.bert = BertModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                              output_hidden_states=True)


    #================================================================
    def guideline_encode(self):
        # breakpoint()
        with torch.no_grad():
            all_bert_outputs = self.bert(self.guideline_piexs.to(self.config.gpu_device), attention_mask=self.guideline_attn_masks.to(self.config.gpu_device))
            relation_type_output = all_bert_outputs[1]
            self.relation_reprs = relation_type_output
        return relation_type_output
    #================================================================


    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks,output_hidden_states=True)
        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs


    def scores(self, bert_outputs, graphs, entity_types_onehot=None,
               predict=False):
        (
            entity_idxs, sq_entity_masks, entity_num, entity_len,
            trigger_idxs, sq_trigger_masks, trigger_num, trigger_len,
        ) = graphs_to_node_idxs(graphs)
        # 
        batch_size, _, bert_dim = bert_outputs.size()

        # entity_idxs: max_entity_len=5, max_entity_num=6, -> [1 0 0 0 0; 2 3 0 0 0; 16 17 18 19 0; ...] 6 nums 5 segments;
        entity_idxs = bert_outputs.new_tensor(entity_idxs, dtype=torch.long)
        trigger_idxs = bert_outputs.new_tensor(trigger_idxs, dtype=torch.long)
        sq_entity_masks = bert_outputs.new_tensor(sq_entity_masks)
        sq_trigger_masks = bert_outputs.new_tensor(sq_trigger_masks)
        

       
        #--------------------------------------------------------------------------------------
        self.entity_masks = torch.reshape(torch.reshape(sq_entity_masks,(-1,entity_len)).sum(-1),(batch_size,-1))
        self.trigger_masks = torch.reshape(torch.reshape(sq_trigger_masks,(-1,trigger_len)).sum(-1),(batch_size,-1))
        self.rel_masks = torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1) # B x n x n
        self.rel_masks = self.rel_masks * torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                        (self.rel_masks.device), 0)

        #--------------------------------------------------------------------------------------

        # breakpoint()
        # entity type scores
        entity_idxs = entity_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        sq_entity_masks = sq_entity_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        entity_words = torch.gather(bert_outputs, 1, entity_idxs)
        entity_words = entity_words * sq_entity_masks
        entity_words = entity_words.view(batch_size, entity_num, entity_len, bert_dim)
        entity_reprs = entity_words.sum(2)
        # entity_reprs = self.entity_layer_norm(entity_reprs)
        if self.config.new_score:
            entity_self_reprs = self.entity_type_ffn(entity_reprs)
            entity_self_reprs = self.linear_entity_dropout(entity_self_reprs)
            entity_type_scores = contract('bek,tk->bet', entity_self_reprs, self.unary_entity_type_reps)
        else:
            entity_type_scores = self.entity_type_ffn(entity_reprs)

        # mention type scores
        mention_type_scores = self.mention_type_ffn(entity_reprs)

        # trigger type scores
        trigger_idxs = trigger_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        sq_trigger_masks = sq_trigger_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_words = torch.gather(bert_outputs, 1, trigger_idxs)
        trigger_words = trigger_words * sq_trigger_masks
        trigger_words = trigger_words.view(batch_size, trigger_num, trigger_len, bert_dim)
        trigger_reprs = trigger_words.sum(2)
        #===============================================================================================
        if self.config.new_score:
            trigger_self_reprs = self.event_type_ffn(trigger_reprs)
            trigger_self_reprs = self.linear_trigger_dropout(trigger_self_reprs)
            event_type_scores = contract('btk,lk->btl', trigger_self_reprs, self.unary_trigger_type_reps)
        else:
            event_type_scores = self.event_type_ffn(trigger_reprs)
        #===============================================================================================

        # relation type score
        ee_idxs = generate_pairwise_idxs(entity_num, entity_num) #[0, 3, 0, 4, 0, 5, 1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5]
        # breakpoint()
        ee_idxs = entity_idxs.new(ee_idxs)
        ee_idxs = ee_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        start_entity_reps = self.start_entity_ffn(entity_reprs)
        end_entity_reps = self.end_entity_ffn(entity_reprs)
        if self.config.new_score:
            start_entity_reps = self.linear_start_dropout(start_entity_reps)
            end_entity_reps = self.linear_end_dropout(end_entity_reps)
            relation_type_scores = contract('bsk,bek,rk->bser', start_entity_reps, end_entity_reps, self.unary_relation_type_reps)
            relation_type_scores = torch.reshape(relation_type_scores,(batch_size,-1,self.relation_type_num)).contiguous()
        else:
            ee_reprs = torch.cat([start_entity_reps, end_entity_reps], dim=1)
            ee_reprs = torch.gather(ee_reprs, 1, ee_idxs)
            ee_reprs = ee_reprs.view(batch_size, -1, 2 * self.entity_hidden_size)
            #=======================================================================
            if self.use_guideliens:
                ee_reprs = self.entity_ffn(ee_reprs)
                # breakpoint()
                # relation_reprs = self.guideline_encode(self.guideline_piexs.to(ee_reprs.device), self.guideline_attn_masks.to(ee_reprs.device))
                relation_reprs = self.relation_ffn(self.relation_reprs)
                # breakpoint()
                relation_type_scores = contract('bnk,ek->bne', ee_reprs, relation_reprs)
            #=======================================================================
            else:
                # breakpoint()
                relation_type_scores = self.relation_type_ffn(ee_reprs)
                if self.config.split_rel_ident:
                    start_entity_ident_reps = self.start_entity_ident_ffn(entity_reprs)
                    end_entity_ident_reps = self.end_entity_ident_ffn(entity_reprs)
                    ee_ident_reprs = torch.cat([start_entity_ident_reps, end_entity_ident_reps], dim=1)
                    ee_ident_reprs = torch.gather(ee_ident_reprs, 1, ee_idxs)
                    ee_ident_reprs = ee_ident_reprs.view(batch_size, -1, 2 * bert_dim)
                    relation_ident_scores = self.relation_ident_ffn(ee_ident_reprs)
        # breakpoint()

        # role type score
        #==========================================================================================
        if self.config.new_score:
            # breakpoint()
            unary_trigger_reps = self.unary_trigger_ffn(trigger_reprs)
            unary_trigger_reps = self.unary_trigger_dropout(unary_trigger_reps)
            if self.use_entity_type:
                if predict:
                    entity_type_scores_softmax = entity_type_scores.softmax(dim=2)
                    # entity_type_scores_softmax = entity_type_scores_softmax.repeat(1, trigger_num, 1)
                    unary_entity_reprs = torch.cat([entity_reprs, entity_type_scores_softmax], dim=2)
                else:
                    # entity_types_onehot = entity_types_onehot.repeat(1, trigger_num, 1)
                    unary_entity_reprs = torch.cat([entity_reprs, entity_types_onehot], dim=2)
            unary_argument_reps = self.unary_argument_ffn(unary_entity_reprs)
            unary_argument_reps = self.unary_argument_dropout(unary_argument_reps)
            role_type_scores = contract('bsk,bek,rk->bser', unary_trigger_reps, unary_argument_reps, self.unary_role_type_reps)
            role_type_scores = torch.reshape(role_type_scores,(batch_size,-1,self.role_type_num)).contiguous()
        else:
            te_idxs = generate_pairwise_idxs(trigger_num, entity_num)
            te_idxs = entity_idxs.new(te_idxs)
            te_idxs = te_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
            te_reprs = torch.cat([trigger_reprs, entity_reprs], dim=1)
            te_reprs = torch.gather(te_reprs, 1, te_idxs)
            te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)
            if self.use_entity_type:
                if predict:
                    entity_type_scores_softmax = entity_type_scores.softmax(dim=2)
                    entity_type_scores_softmax = entity_type_scores_softmax.repeat(1, trigger_num, 1)
                    te_reprs = torch.cat([te_reprs, entity_type_scores_softmax], dim=2)
                else:
                    entity_types_onehot = entity_types_onehot.repeat(1, trigger_num, 1)
                    te_reprs = torch.cat([te_reprs, entity_types_onehot], dim=2)
            role_type_scores = self.role_type_ffn(te_reprs)
        #==========================================================================================


        #---------------------------------------------------------------------------------------------------------------
        # high order
        # if self.use_high_order_tl or self.use_high_order_le or self.use_high_order_tre \
        #         or self.use_high_order_ere or self.use_high_order_sibling \
        #         or self.use_high_order_coparent or self.use_high_order_er or self.use_high_order_re_sibling \
        #         or self.use_high_order_re_coparent or self.use_high_order_re_grandparent or self.use_high_order_rr_coparent or self.use_high_order_rr_grandparent:
        #     self.use_high_order = True
        # else:
        #     self.use_high_order = False
        
        # if self.use_high_order:
        event_role_factor = None
        role_entity_factor = None
        event_role_entity_factor = None
        event_role_sibling_factor = None
        event_role_cop_factor = None
        entity_relation_entity_factor = None
        entity_relation_factor = None
        relation_entity_factor = None
        entity_relation_sib_factor = None
        entity_relation_cop_factor = None
        entity_relation_gp_factor = None
        start_entity_relation_factor = None
        end_entity_relation_factor = None
        event_relation_cop_factor = None
        event_relation_gp_factor = None

        
        if self.use_high_order_tl:
            event_role_factor = self.event_role_factor_score(trigger_reprs,entity_reprs)
        
        if self.use_high_order_le:
            if self.config.new_potential:
                if self.valid_role_entity and self.config.penalized:
                    le_valid_pattern_mask = self.role_entity_factor_mask()
                    le_valid_pattern_mask = le_valid_pattern_mask.to(self.role_entity_potential.device)
                    role_entity_factor = self.role_entity_potential * le_valid_pattern_mask
                role_entity_factor = self.role_entity_potential.repeat(batch_size,trigger_num,entity_num,1,1)
                # tre_pairs_mask: B x T x E
                self.tre_pairs_mask = torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
                role_entity_factor = role_entity_factor * torch.unsqueeze(torch.unsqueeze(self.tre_pairs_mask, -1), -1)
            else:
                role_entity_factor = self.role_entity_factor_score(trigger_reprs,entity_reprs)
        
        if self.use_high_order_tre:
            if self.config.new_potential:
                event_role_entity_factor = self.event_role_entity_potential.repeat(batch_size,trigger_num,entity_num,1,1,1)
                # tre_pairs_mask: B x T x E
                self.tre_pairs_mask = torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
                event_role_entity_factor = event_role_entity_factor * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.tre_pairs_mask, -1), -1),-1)
            else:
                event_role_entity_factor = self.event_role_entity_factor_score(trigger_reprs, entity_reprs)

        if self.use_high_order_ere:
            entity_relation_entity_factor = self.entity_relation_entity_factor_score(entity_reprs)

        if self.use_high_order_sibling:
            event_role_sibling_factor = self.event_entity_sibling_factor_score(trigger_reprs, entity_reprs)

        if self.use_high_order_er:
            if self.config.new_potential:
                # B x E x E x EL x RL
                entity_relation_factor = self.entity_relation_potential.repeat(batch_size,entity_num,entity_num,1,1)
                # B x E x E x RL x EL
                relation_entity_factor = self.relation_entity_potential.repeat(batch_size,entity_num,entity_num,1,1)
                
                # er_pairs_mask: B x E x E
                ere_pairs_mask = torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
                entity_relation_factor = entity_relation_factor * torch.unsqueeze(torch.unsqueeze(ere_pairs_mask, -1), -1)
                relation_entity_factor = relation_entity_factor * torch.unsqueeze(torch.unsqueeze(ere_pairs_mask, -1), -1)

                # mask diag
                entity_relation_factor = entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x 8 x 7 x E x E
                entity_relation_factor = entity_relation_factor * \
                                torch.unsqueeze(torch.unsqueeze(
                                    torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                    (entity_relation_factor.device), 0), 0), 0)
                entity_relation_factor = entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x E x E x 8 x 7
                relation_entity_factor = relation_entity_factor.permute(0, 3, 4, 1, 2)  # B x 8 x 7 x E x E
                relation_entity_factor = relation_entity_factor * \
                                torch.unsqueeze(torch.unsqueeze(
                                    torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                    (relation_entity_factor.device), 0), 0), 0)
                relation_entity_factor = relation_entity_factor.permute(0, 3, 4, 1, 2)  # B x E x E x 8 x 7
            else:
                if self.config.test_er:
                    # breakpoint()
                    start_entity_relation_factor, end_entity_relation_factor = self.entity_relation_factor_score(entity_reprs)                        
                else:
                    entity_relation_factor = self.entity_relation_factor_score(entity_reprs)
        
        if self.use_high_order_coparent:
            event_role_cop_factor = self.event_entity_coparent_factor_score(trigger_reprs, entity_reprs)

        if self.use_high_order_re_sibling:
            entity_relation_sib_factor = self.entity_relation_sib_factor_score(entity_reprs)
        
        if self.use_high_order_re_coparent:
            entity_relation_cop_factor = self.entity_relation_cop_factor_score(entity_reprs)

        if self.use_high_order_re_grandparent:
            entity_relation_gp_factor = self.entity_relation_gp_factor_score(entity_reprs)

        if self.use_high_order_rr_coparent:
            event_relation_cop_factor = self.event_relation_cop_factor_score(trigger_reprs, entity_reprs)
        
        if self.use_high_order_rr_grandparent:
            event_relation_gp_factor = self.event_relation_gp_factor_score(trigger_reprs, entity_reprs)

        event_type_scores_q, role_type_scores, entity_type_scores_q, relation_type_scores \
            = self.mfvi(event_type_scores[:,:,1:],role_type_scores,entity_type_scores[:,:,1:],relation_type_scores,
                                                            event_role_factor = event_role_factor,
                                                            role_entity_factor = role_entity_factor,
                                                            event_role_entity_factor=event_role_entity_factor,
                                                            event_role_sibling_factor=event_role_sibling_factor,
                                                            event_role_cop_factor=event_role_cop_factor,
                                                            entity_relation_entity_factor=entity_relation_entity_factor,
                                                            relation_entity_factor = relation_entity_factor,
                                                            entity_relation_factor = entity_relation_factor,
                                                            entity_relation_sib_factor=entity_relation_sib_factor,
                                                            entity_relation_cop_factor=entity_relation_cop_factor,
                                                            entity_relation_gp_factor=entity_relation_gp_factor,
                                                            start_entity_relation_factor = start_entity_relation_factor,
                                                            end_entity_relation_factor = end_entity_relation_factor,
                                                            event_relation_cop_factor = event_relation_cop_factor,
                                                            event_relation_gp_factor = event_relation_gp_factor,
                                                            )
        
        entity_type_scores = torch.cat((entity_type_scores[:,:,0:1],entity_type_scores_q),-1)
        event_type_scores = torch.cat((event_type_scores[:,:,0:1],event_type_scores_q),-1)
        
        if self.config.split_rel_ident:
            return (entity_type_scores, mention_type_scores, event_type_scores, relation_ident_scores,
                relation_type_scores, role_type_scores)  
        
        return (entity_type_scores, mention_type_scores, event_type_scores,
                relation_type_scores, role_type_scores)


    def rebatch(self, batch, entities, triggers):
        # batch_graphs = batch.graphs
        # instances = batch.instances
        data=[]

        for pred_entities, pred_triggers, instance in zip(entities, triggers, batch.instances):
            vocabs = instance.graph.vocabs
            entity_list = instance.graph.entities
            trigger_list = instance.graph.triggers
            relation_list = instance.graph.relations
            role_list = instance.graph.roles
            mention_list = instance.graph.mentions
            # entity_type_idxs = instance.entity_type_idxs
            # event_type_idxs = instance.event_type_idxs
            # relation_type_idxs = instance.relation_type_idxs
            # mention_type_idxs = instance.mention_type_idxs
            # role_type_idxs = instance.role_type_idxs
            # entity_num = instance.entity_num
            # trigger_num = instance.trigger_num

            # new entity list and mention list
            pred_entities_set = set([(entity[0],entity[1]) for entity in pred_entities])
            gold_entities_set = set([(entity[0],entity[1]) for entity in entity_list])
            wrong_preds = pred_entities_set.difference(gold_entities_set)
            wrong_preds = remove_overlap_entities(gold_entities_set, wrong_preds)
            # 
            for wrong_pred in wrong_preds:
                entity_list.append((wrong_pred[0],wrong_pred[1],0))
                mention_list.append((wrong_pred[0],wrong_pred[1],0))
            entity_list.sort(key=lambda x: x[0])
            mention_list.sort(key=lambda x: x[0])

            entity_type_idxs = [entity[-1] for entity in entity_list]
            mention_type_idxs = [mention[-1] for mention in mention_list]


            # new trigger list
            pred_triggers_set = set([(trigger[0],trigger[1]) for trigger in pred_triggers])
            gold_triggers_set = set([(trigger[0],trigger[1]) for trigger in trigger_list])
            wrong_preds = list(pred_triggers_set.difference(gold_triggers_set))
            wrong_preds = remove_overlap_entities(gold_triggers_set, wrong_preds)
            for wrong_pred in wrong_preds:
                trigger_list.append((wrong_pred[0],wrong_pred[1],0))
            trigger_list.sort(key=lambda x: x[0])
            event_type_idxs = [event[-1] for event in trigger_list]

            # nums
            entity_num = len(entity_list)
            trigger_num = len(trigger_list)

            # if entity_num != instance.entity_num:
            #     
            # if trigger_num != instance.trigger_num:
            #     

            # relation types
            relation_type_idxs = [[0 for _ in range(entity_num)]
                                  for _ in range(entity_num)]
            for relation in instance.graph.relations:
                entity1 = instance.graph.entities[relation[0]]
                entity2 = instance.graph.entities[relation[1]]
                relation_type_idxs[entity_list.index(entity1)][entity_list.index(entity2)] = relation[2]
                if not self.config.relation_directional:
                    relation_type_idxs[entity_list.index(entity2)][entity_list.index(entity1)] = relation[2]
                if self.config.symmetric_relations and self.relation_type_itos[relation[2]] in self.config.symmetric_relations:
                    relation_type_idxs[entity_list.index(entity2)][entity_list.index(entity1)] = relation[2]
            if self.config.relation_mask_self:
                for i in range(len(relation_type_idxs)):
                    relation_type_idxs[i][i] = -100

            # 
            # role types
            role_type_idxs = [[0 for _ in range(entity_num)]
                                  for _ in range(trigger_num)]
            for role in role_list:
                trigger = instance.graph.triggers[role[0]]
                entity = instance.graph.entities[role[1]]
                role_type_idxs[trigger_list.index(trigger)][entity_list.index(entity)] = role[2]

            new_graph = Graph(
                entities=entity_list,
                triggers=trigger_list,
                relations=relation_list,
                roles=role_list,
                mentions=mention_list,
                vocabs=vocabs,
            )

            new_instance = Instance(
                sent_id=instance.sent_id,
                tokens=instance.tokens,
                pieces=instance.pieces,
                piece_idxs=instance.piece_idxs,
                token_lens=instance.token_lens,
                attention_mask=instance.attention_mask,
                entity_label_idxs=instance.entity_label_idxs,
                trigger_label_idxs=instance.trigger_label_idxs,
                entity_type_idxs=entity_type_idxs,
                event_type_idxs=event_type_idxs,
                relation_type_idxs=relation_type_idxs,
                mention_type_idxs=mention_type_idxs,
                role_type_idxs=role_type_idxs,
                graph=new_graph,
                entity_num=entity_num,
                trigger_num=trigger_num,
            )

            data.append(new_instance)

        new_batch = self.collate_fn((data))

        return new_batch


    def collate_fn(self, batch):
        # 
        batch_piece_idxs = []
        batch_tokens = []
        batch_entity_labels, batch_trigger_labels = [], []
        batch_entity_types, batch_event_types = [], []
        batch_relation_types, batch_role_types = [], []
        batch_mention_types = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        max_entity_num = max([inst.entity_num for inst in batch] + [1])
        max_trigger_num = max([inst.trigger_num for inst in batch] + [1])

        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)
            batch_tokens.append(inst.tokens)
            # for identification
            batch_entity_labels.append(inst.entity_label_idxs +
                                       [0] * (max_token_num - token_num))
            batch_trigger_labels.append(inst.trigger_label_idxs +
                                        [0] * (max_token_num - token_num))
            # for classification
            batch_entity_types.extend(inst.entity_type_idxs +
                                      [-100] * (max_entity_num - inst.entity_num))
            batch_event_types.extend(inst.event_type_idxs +
                                     [-100] * (max_trigger_num - inst.trigger_num))
            batch_mention_types.extend(inst.mention_type_idxs +
                                       [-100] * (max_entity_num - inst.entity_num))
            for l in inst.relation_type_idxs:
                batch_relation_types.extend(
                    l + [-100] * (max_entity_num - inst.entity_num))
            batch_relation_types.extend(
                [-100] * max_entity_num * (max_entity_num - inst.entity_num))
            for l in inst.role_type_idxs:
                batch_role_types.extend(
                    l + [-100] * (max_entity_num - inst.entity_num))
            batch_role_types.extend(
                [-100] * max_entity_num * (max_trigger_num - inst.trigger_num))

        if self.config.use_gpu:
            batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.cuda.FloatTensor(
                batch_attention_masks)

            batch_entity_labels = torch.cuda.LongTensor(batch_entity_labels)
            batch_trigger_labels = torch.cuda.LongTensor(batch_trigger_labels)
            batch_entity_types = torch.cuda.LongTensor(batch_entity_types)
            batch_mention_types = torch.cuda.LongTensor(batch_mention_types)
            batch_event_types = torch.cuda.LongTensor(batch_event_types)
            batch_relation_types = torch.cuda.LongTensor(batch_relation_types)
            batch_role_types = torch.cuda.LongTensor(batch_role_types)

            token_nums = torch.cuda.LongTensor(token_nums)
        else:
            batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)

            batch_entity_labels = torch.LongTensor(batch_entity_labels)
            batch_trigger_labels = torch.LongTensor(batch_trigger_labels)
            batch_entity_types = torch.LongTensor(batch_entity_types)
            batch_mention_types = torch.LongTensor(batch_mention_types)
            batch_event_types = torch.LongTensor(batch_event_types)
            batch_relation_types = torch.LongTensor(batch_relation_types)
            batch_role_types = torch.LongTensor(batch_role_types)

            token_nums = torch.LongTensor(token_nums)

        return Batch(
            instances= batch,
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            entity_label_idxs=batch_entity_labels,
            trigger_label_idxs=batch_trigger_labels,
            entity_type_idxs=batch_entity_types,
            mention_type_idxs=batch_mention_types,
            event_type_idxs=batch_event_types,
            relation_type_idxs=batch_relation_types,
            role_type_idxs=batch_role_types,
            graphs=batch_graphs,
            token_nums=token_nums,
        )


    def forward(self, batch):
        # encoding
        # from torch.profiler import profile, record_function, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,) as prof:
        #     with record_function("model"):
        bert_outputs = self.encode(batch.piece_idxs,
                                batch.attention_masks,
                                batch.token_lens)
        batch_size, _, _ = bert_outputs.size()


        # lists of [[[start,end,label],...],[[],...],....]

        if self.config.rebatch:
            predicted_entities, predicted_triggers = self.ident_model.provide(batch)

            batch = self.rebatch(batch, predicted_entities, predicted_triggers)


        #
        # entity type indices -> one hot
        entity_types = batch.entity_type_idxs.view(batch_size, -1)
        entity_types = torch.clamp(entity_types, min=0)
        entity_types_onehot = bert_outputs.new_zeros(*entity_types.size(),
                                                    self.entity_type_num)
        entity_types_onehot.scatter_(2, entity_types.unsqueeze(-1), 1)


        # identification
        if not self.split_train:
            entity_label_scores = self.entity_label_ffn(bert_outputs)
            trigger_label_scores = self.trigger_label_ffn(bert_outputs)

            entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
            entity_label_loglik = self.entity_crf.loglik(entity_label_scores,
                                                        batch.entity_label_idxs,
                                                        batch.token_nums)
            trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
            trigger_label_loglik = self.trigger_crf.loglik(trigger_label_scores,
                                                        batch.trigger_label_idxs,
                                                        batch.token_nums)

        # classification
        scores = self.scores(bert_outputs, batch.graphs, entity_types_onehot)
        if self.config.split_rel_ident:
            ( entity_type_scores, mention_type_scores, event_type_scores, relation_ident_scores, 
                relation_type_scores, role_type_scores
            ) = scores
            relation_ident_scores = relation_ident_scores.view(-1,2)
        else:
            ( entity_type_scores, mention_type_scores, event_type_scores,
                relation_type_scores, role_type_scores
            ) = scores
        try:
            relation_type_scores = relation_type_scores.view(-1, self.relation_type_num)
            entity_type_scores = entity_type_scores.view(-1, self.entity_type_num)
            event_type_scores = event_type_scores.view(-1, self.event_type_num)
            role_type_scores = role_type_scores.view(-1, self.role_type_num)
            mention_type_scores = mention_type_scores.view(-1, self.mention_type_num)
        except:
            breakpoint()


        # breakpoint()
        if self.config.split_rel_ident:
            # gold_rel_ident = torch.zeros_like(batch.relation_type_idxs)
            gold_rel_ident = torch.where(batch.relation_type_idxs>0,1,batch.relation_type_idxs)
            # gold_rel_cls = torch.zeros_like(batch.relation_type_idxs)
            gold_rel_cls = torch.where(batch.relation_type_idxs>0, batch.relation_type_idxs, -100)
            # breakpoint()
            classification_loss = self.entity_criteria(entity_type_scores,
                                                batch.entity_type_idxs) + \
                            self.event_criteria(event_type_scores,
                                                batch.event_type_idxs) + \
                            self.role_criteria(role_type_scores,
                                                batch.role_type_idxs) + \
                            self.mention_criteria(mention_type_scores,
                                                    batch.mention_type_idxs) + \
                            9*self.relation_ident_criteria(relation_ident_scores,
                                                    gold_rel_ident) + \
                            self.relation_criteria(relation_type_scores,
                                                        gold_rel_cls)
            
        else:
            # classification_loss = self.entity_criteria(entity_type_scores, batch.entity_type_idxs)+self.relation_criteria(relation_type_scores, batch.relation_type_idxs)
            classification_loss = self.entity_criteria(entity_type_scores,
                                                batch.entity_type_idxs) + \
                            self.event_criteria(event_type_scores,
                                                batch.event_type_idxs) + \
                            self.role_criteria(role_type_scores,
                                                batch.role_type_idxs) + \
                            self.mention_criteria(mention_type_scores,
                                                    batch.mention_type_idxs) + \
                            self.relation_criteria(relation_type_scores,
                                                        batch.relation_type_idxs)
            # breakpoint()
        if not self.split_train:
            loss = classification_loss - entity_label_loglik.mean() - trigger_label_loglik.mean()
        else:
            loss = classification_loss

        # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=50))
        # breakpoint()
        # global features
        if self.use_global_features:
            gold_scores = self.compute_graph_scores(batch.graphs, scores)
            top_graphs = self.generate_locally_top_graphs(batch.graphs, scores)
            top_scores = self.compute_graph_scores(top_graphs, scores)
            global_loss = (top_scores - gold_scores).clamp(min=0)
            loss = loss + global_loss.mean()
        return loss

        
    def predict_(self, batch):
        self.eval()

        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()
        # 

        # identification
        if self.split_train:
            entities, triggers = self.ident_model.provide(batch)
        else:
            entity_label_scores = self.entity_label_ffn(bert_outputs)
            entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
            trigger_label_scores = self.trigger_label_ffn(bert_outputs)
            trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
            _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                                batch.token_nums)
            _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                    batch.token_nums)
            entities = tag_paths_to_spans(entity_label_preds,
                                        batch.token_nums,
                                        self.entity_label_stoi)
            triggers = tag_paths_to_spans(trigger_label_preds,
                                        batch.token_nums,
                                        self.trigger_label_stoi)

        

        node_graphs = [Graph(e, t, [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]
        scores = self.scores(bert_outputs, node_graphs, predict=True)
        max_entity_num = max(max(len(seq_entities) for seq_entities in entities), 1)

        batch_graphs = []
        # Decode each sentence in the batch
        for i in range(batch_size):
            seq_entities, seq_triggers = entities[i], triggers[i]
            spans = sorted([(*i, True) for i in seq_entities] +
                           [(*i, False) for i in seq_triggers],
                           key=lambda x: (x[0], x[1], not x[-1]))
            entity_num, trigger_num = len(seq_entities), len(seq_triggers)
            if entity_num == 0 and trigger_num == 0:
                # skip decoding
                batch_graphs.append(Graph.empty_graph(self.vocabs))
                continue
           
            graph = self.decode(spans,
                                entity_type_scores=scores[0][i],
                                mention_type_scores=scores[1][i],
                                event_type_scores=scores[2][i],
                                relation_type_scores=scores[3][i],
                                role_type_scores=scores[4][i],
                                entity_num=max_entity_num)
            batch_graphs.append(graph)

        self.train()
        return batch_graphs


    def compute_graph_scores(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores
        ) = scores
        label_idxs = graphs_to_label_idxs(graphs)
        label_idxs = [entity_type_scores.new_tensor(idx,
                                               dtype=torch.long if i % 2 == 0
                                               else torch.float)
                      for i, idx in enumerate(label_idxs)]
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask
        ) = label_idxs
        # Entity score
        entity_idxs = entity_idxs.unsqueeze(-1)
        entity_scores = torch.gather(entity_type_scores, 2, entity_idxs)
        entity_scores = entity_scores.squeeze(-1) * entity_mask
        entity_score = entity_scores.sum(1)
        # Trigger score
        trigger_idxs = trigger_idxs.unsqueeze(-1)
        trigger_scores = torch.gather(trigger_type_scores, 2, trigger_idxs)
        trigger_scores = trigger_scores.squeeze(-1) * trigger_mask
        trigger_score = trigger_scores.sum(1)
        # Relation score
        relation_idxs = relation_idxs.unsqueeze(-1)
        relation_scores = torch.gather(relation_type_scores, 2, relation_idxs)
        relation_scores = relation_scores.squeeze(-1) * relation_mask
        relation_score = relation_scores.sum(1)
        # Role score
        role_idxs = role_idxs.unsqueeze(-1)
        role_scores = torch.gather(role_type_scores, 2, role_idxs)
        role_scores = role_scores.squeeze(-1) * role_mask
        role_score = role_scores.sum(1)

        score = entity_score + trigger_score + role_score + relation_score

        global_vectors = [generate_global_feature_vector(g, self.global_feature_maps, features=self.global_features)
                          for g in graphs]
        global_vectors = entity_scores.new_tensor(global_vectors)
        global_weights = self.global_feature_weights.unsqueeze(0).expand_as(global_vectors)
        global_score = (global_vectors * global_weights).sum(1)
        score = score + global_score

        return score


    def generate_locally_top_graphs(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores
        ) = scores
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
        top_graphs = []
        for graph_idx, graph in enumerate(graphs):
            entity_num = graph.entity_num
            trigger_num = graph.trigger_num
            _, top_entities = entity_type_scores[graph_idx].max(1)
            top_entities = top_entities.tolist()[:entity_num]
            top_entities = [(i, j, k) for (i, j, _), k in
                            zip(graph.entities, top_entities)]
            _, top_triggers = trigger_type_scores[graph_idx].max(1)
            top_triggers = top_triggers.tolist()[:trigger_num]
            top_triggers = [(i, j, k) for (i, j, _), k in
                            zip(graph.triggers, top_triggers)]
            
            top_relation_scores, top_relation_labels = relation_type_scores[graph_idx].max(1)
            top_relation_scores = top_relation_scores.tolist()
            top_relation_labels = top_relation_labels.tolist()
            top_relations = [(i, j) for i, j in zip(top_relation_scores, top_relation_labels)]
            top_relation_list = []
            for i in range(entity_num):
                for j in range(entity_num):
                    if i < j:
                        score_1, label_1 = top_relations[i * max_entity_num + j]
                        score_2, label_2 = top_relations[j * max_entity_num + i]
                        if score_1 > score_2 and label_1 != 0:
                            top_relation_list.append((i, j, label_1))
                        if score_2 > score_1 and label_2 != 0: 
                            top_relation_list.append((j, i, label_2))

            _, top_roles = role_type_scores[graph_idx].max(1)
            top_roles = top_roles.tolist()
            top_roles = [(i, j, top_roles[i * max_entity_num + j])
                         for i in range(trigger_num) for j in range(entity_num)
                         if top_roles[i * max_entity_num + j] != 0]
            top_graphs.append(Graph(
                entities=top_entities,
                triggers=top_triggers,
                # relations=top_relations,
                relations=top_relation_list,
                roles=top_roles,
                vocabs=graph.vocabs
            ))
        return top_graphs


    def trim_beam_set(self, beam_set, beam_size):
        if len(beam_set) > beam_size:
            beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
            beam_set = beam_set[:beam_size]
        return beam_set


    def compute_graph_score(self, graph):
        score = graph.graph_local_score
        if self.use_global_features:
            global_vector = generate_global_feature_vector(graph,
                                                           self.global_feature_maps,
                                                           features=self.global_features)
            global_vector = self.global_feature_weights.new_tensor(global_vector)
            global_score = global_vector.dot(self.global_feature_weights).item()
            score = score + global_score
        return score


    def decode(self,
               spans,
               entity_type_scores,
               mention_type_scores,
               event_type_scores,
               relation_type_scores,
               role_type_scores,
               entity_num):

        beam_set = [Graph.empty_graph(self.vocabs)]
        entity_idx, trigger_idx = 0, 0

        for start, end, _, is_entity_node in spans:
            # 1. node step
            if is_entity_node:
                node_scores = entity_type_scores[entity_idx].tolist()
            else:
                node_scores = event_type_scores[trigger_idx].tolist()
            node_scores_norm = normalize_score(node_scores)
            node_scores = [(s, i, n) for i, (s, n) in enumerate(zip(node_scores,
                                                                node_scores_norm))]
            node_scores.sort(key=lambda x: x[0], reverse=True)
            top_node_scores = node_scores[:self.beta_v]


            beam_set_ = []
            for graph in beam_set:
                for score, label, score_norm in top_node_scores:
             
                    graph_ = graph.copy()
                    if is_entity_node:
                        graph_.add_entity(start, end, label, score, score_norm)
                    else:
                        graph_.add_trigger(start, end, label, score, score_norm)
                    beam_set_.append(graph_)
            beam_set = beam_set_

            # 2. edge step
            if is_entity_node:
                # add a new entity: new relations, new argument roles
                for i in range(entity_idx):
                    # add relation edges
                    edge_scores_1 = relation_type_scores[i * entity_num + entity_idx].tolist()
                    edge_scores_2 = relation_type_scores[entity_idx * entity_num + i].tolist()
                    edge_scores_norm_1 = normalize_score(edge_scores_1)
                    edge_scores_norm_2 = normalize_score(edge_scores_2)

                    if self.relation_directional:
                        edge_scores = [(max(s1, s2), n2 if s1 < s2 else n1, i, s1 < s2)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]
                    else:
                        edge_scores = [(max(s1, s2), n2 if s1 < n2 else n1, i, False)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, score_norm, label, inverse in top_edge_scores:
                            rel_cur_ent = label * 100 + graph.entities[-1][-1]
                            rel_pre_ent = label * 100 + graph.entities[i][-1]
                            if label == 0 or (rel_pre_ent in self.valid_relation_entity and
                                              rel_cur_ent in self.valid_relation_entity):
                                graph_ = graph.copy()
                                if self.relation_directional and inverse:
                                    graph_.add_relation(entity_idx, i, label, score, score_norm)
                                else:
                                    graph_.add_relation(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_relation(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 200:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                for i in range(trigger_idx):
                    # add argument role edges
                    edge_scores = role_type_scores[i * entity_num + entity_idx].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores, edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            role_entity = label * 100 + graph.entities[-1][-1]
                            event_role = graph.triggers[i][-1] * 100 + label
                            if label == 0 or (event_role in self.valid_event_role and
                                              role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_role(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)
                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            else:
                # add a new trigger: new argument roles
                for i in range(entity_idx):
                    edge_scores = role_type_scores[trigger_idx * entity_num + i].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores,
                                                                            edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            event_role = graph.triggers[-1][-1] * 100 + label
                            role_entity = label * 100 + graph.entities[i][-1]
                            if label == 0 or (event_role in self.valid_event_role
                                              and role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(trigger_idx, i, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_role(trigger_idx, i, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            if is_entity_node:
                entity_idx += 1
            else:
                trigger_idx += 1
        beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
        graph = beam_set[0]

        # predict mention types
        _, mention_types = mention_type_scores.max(dim=1)
        mention_types = mention_types[:entity_idx]
        mention_list = [(i, j, l.item()) for (i, j, k), l
                        in zip(graph.entities, mention_types)]
        graph.mentions = mention_list

     

        return graph
    
    
    # ------------------------------------------------------------------------------------------------  
    def predict(self, batch):
        self.eval()

        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()



        # identification

        if self.split_train:
            if self.config.gold_ent:
                entities_ = [graph.entities for graph in batch.graphs]
                entities = []
                for sent in entities_:
                    sent_entity = []
                    if (not sent) or (not sent[0]):
                        entities.append([])
                    else:
                        for entity in sent:
                            sent_entity.append([entity[0],entity[1], self.entity_type_itos[entity[2]]])
                        entities.append(sent_entity)

                triggers_ = [graph.triggers for graph in batch.graphs]
                triggers = []
                for sent in triggers_:
                    sent_trigger = []
                    if (not sent) or (not sent[0]):
                        triggers.append([])
                    else:
                        for trigger in sent:
                            sent_trigger.append([trigger[0],trigger[1], self.event_type_itos[trigger[2]]])
                        triggers.append(sent_trigger)
                # breakpoint()
            else:
                entities, triggers = self.ident_model.provide(batch)
        else:
            entity_label_scores = self.entity_label_ffn(bert_outputs)
            entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
            trigger_label_scores = self.trigger_label_ffn(bert_outputs)
            trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
            _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                                   batch.token_nums)
            _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                     batch.token_nums)
            entities = tag_paths_to_spans(entity_label_preds,
                                          batch.token_nums,
                                          self.entity_label_stoi)
            triggers = tag_paths_to_spans(trigger_label_preds,
                                          batch.token_nums,
                                          self.trigger_label_stoi)

       

        node_graphs = [Graph(e, t, [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]
        
        scores = self.scores(bert_outputs, node_graphs, predict=True)
        max_entity_num = max(max(len(seq_entities) for seq_entities in entities), 1)
        max_event_num = max(max(len(seq_events) for seq_events in triggers), 1)

        batch_graphs = []

        entity_scores, entity_types = scores[0].max(dim=-1)
        mention_scores, mention_types = scores[1].max(dim=-1)
        event_scores, event_types = scores[2].max(dim=-1)
        
        if self.config.split_rel_ident:
            relation_scores, relation_types = scores[4].max(dim=-1)
            relation_ident_scores, relation_idents = scores[3].max(dim=-1)
            role_scores, role_types = scores[5].max(dim=-1)
        else:
            relation_scores, relation_types = scores[3].max(dim=-1)
            role_scores, role_types = scores[4].max(dim=-1)
        for i in range(batch_size):
            
            i_entities = entities[i]
            entity_num = len(i_entities)
            i_entity_types = entity_types[i][:entity_num]
            i_entity_scores = entity_scores[i][:entity_num]
            # 
            entity_list = []
            for j in range(len(i_entity_types)):
                entity_list.append((i_entities[j][0],i_entities[j][1],int(i_entity_types[j])))
            
           
            i_mention_types = mention_types[i][:entity_num]
            mention_list = []
            for j in range(len(i_mention_types)):
                mention_list.append((i_entities[j][0], i_entities[j][1], int(i_mention_types[j])))

            i_events = triggers[i]
            trigger_num = len(i_events)
            i_event_types = event_types[i][:trigger_num]
            i_event_scores = event_scores[i][:trigger_num]
            event_list = []
            for j in range(len(i_event_types)):
                event_list.append((i_events[j][0], i_events[j][1], int(i_event_types[j])))

            # breakpoint()
            i_relations = relation_types.view(-1,max_entity_num,max_entity_num)[i][:entity_num,:entity_num]
            if self.config.split_rel_ident:
                i_relation_idents = relation_idents.view(-1,max_entity_num,max_entity_num)[i][:entity_num,:entity_num]
                i_relations = i_relations * i_relation_idents
            mask_diag = (1-torch.eye(entity_num,entity_num)).type_as(i_relations).to(i_relations.device)
            i_relations = i_relations*mask_diag
            i_relation_scores = relation_scores.view(-1,max_entity_num,max_entity_num)[i][:entity_num,:entity_num]
            select = i_relations>0
            i_relation_scores = i_relation_scores[select]
            pred_relations = torch.cat((torch.nonzero(select,as_tuple=False), i_relations[select].unsqueeze(-1)), -1)
            relation_list = [tuple(i.tolist()) for i in pred_relations]

            i_roles = role_types.view(-1, max_event_num, max_entity_num)[i][:trigger_num, :entity_num]
            i_role_scores = role_scores.view(-1, max_event_num, max_entity_num)[i][:trigger_num, :entity_num]
            select = i_roles > 0
            i_role_scores = i_role_scores[select]
            pred_roles = torch.cat((torch.nonzero(select,as_tuple=False), i_roles[select].unsqueeze(-1)), -1)
            # roles_list = [tuple(i.tolist()) for i in pred_roles]
            roles_list = []
            for i in pred_roles:
                trigger_idx, entity_idx, role_type = i
                event_type = event_list[trigger_idx][-1]
                entity_type = entity_list[entity_idx][-1]
                if int(event_type*100+role_type) in self.valid_event_role and int(role_type*100+entity_type) in self.valid_role_entity:
                    roles_list.append(tuple(i.tolist()))

            graph = Graph(entity_list, event_list, relation_list, roles_list, self.vocabs, mention_list)
            graph.entity_scores = i_entity_scores.tolist()
            # 
            graph.trigger_scores = i_event_scores.tolist()
            graph.relation_scores = i_relation_scores.tolist()
            graph.role_scores = i_role_scores.tolist()
            batch_graphs.append(graph)
        
        self.train()
        return batch_graphs


    # ------------------------------------------------------------------------------------------------
    def event_role_factor_mask(self):
        mask = torch.zeros(self.event_type_num, self.role_type_num)
        for event_role in self.valid_event_role:
            event_idx = int(event_role / 100)
            role_idx = event_role % 100
            mask[event_idx, role_idx] = 1
        # mask[:, 0] = 1
        return mask

        
    # ------------------------------------------------------------------------------------------------
    def role_entity_factor_mask(self):
        mask = torch.zeros(self.role_type_num, self.entity_type_num).fill_(0)
        for role_entity in self.valid_role_entity:
            role_idx = int(role_entity / 100)
            entity_idx = role_entity % 100
            mask[role_idx, entity_idx] = 1
        mask[0, :] = 0
        mask[:, 0] = 0
        return mask


    # ------------------------------------------------------------------------------------------------
    def event_role_entity_factor_mask(self):
        mask = torch.zeros(self.event_type_num, self.role_type_num, self.entity_type_num).fill_(-1111)
        role_entity_dict = {}
        for role_entity in self.valid_role_entity:
            role_idx = int(role_entity / 100)
            if role_idx in role_entity_dict:
                role_entity_dict[role_idx].append(role_entity % 100)
            else:
                role_entity_dict[role_idx] = [role_entity % 100]
        for event_role in self.valid_event_role:
            event_idx = int(event_role / 100)
            role_idx = event_role % 100
            for entity_idx in role_entity_dict[role_idx]:
                mask[event_idx, role_idx, entity_idx] = 1111
        # event_idx, none, entity_idx is legal
        mask[1:, 0, 1:] = 0
        return mask


    def event_role_entity_factor_mask_nonrole(self):
        mask = torch.zeros(self.event_type_num-1, self.role_type_num, self.entity_type_num-1).fill_(1.)
        mask[:, 0, :] = 0.
        return mask


    # ------------------------------------------------------------------------------------------------
    def entity_relation_entity_factor_mask(self):
        mask = torch.zeros(self.entity_type_num-1, self.relation_type_num, self.entity_type_num-1).fill_(-1111)
        relation_entity_dict = {}
        for relation_entity in self.valid_relation_entity:
            relation_idx = int(relation_entity / 100)
            entity_idx = relation_entity % 100
            if relation_idx in relation_entity_dict:
                relation_entity_dict[relation_idx].append(entity_idx-1)
            else:
                relation_entity_dict[relation_idx] = [entity_idx-1]
        for relation_idx in relation_entity_dict:
            for entity_idx_s in relation_entity_dict[relation_idx]:
                for entity_idx_e in relation_entity_dict[relation_idx]:
                    mask[entity_idx_s, relation_idx, entity_idx_e] = 1
        mask[:,0,:] = 0
        return mask


    # ------------------------------------------------------------------------------------------------
    def entity_relation_factor_mask(self):
        mask = torch.zeros(self.entity_type_num, self.relation_type_num).fill_(-1111)
        for relation_entity in self.valid_relation_entity:
            relation_idx = int(relation_entity / 100)
            entity_idx = relation_entity % 100
            mask[entity_idx, relation_idx] = 1
        mask[1:, 0] = 0
        return mask


    # ------------------------------------------------------------------------------------------------
    def event_role_factor_score(self, trigger_reps, entity_reps):
        trigger_reps = self.trigger_tl_W(trigger_reps)
        entity_reps =  self.entity_tl_W(entity_reps)
        event_type_reps = self.event_type_tl_W  # 34 x k
        role_type_reps = self.role_type_tl_W  # 23 x k
       

        # (B x T x E x 34 x 23) * (34 x 23) tl_valid_pattern_mask
        event_role_factor = contract('bnk,bmk,tk,lk-> bnmtl', trigger_reps, entity_reps, event_type_reps,
                                            role_type_reps)
        # 
        if self.valid_event_role:
            tl_valid_pattern_mask = self.event_role_factor_mask()
            tl_valid_pattern_mask = tl_valid_pattern_mask.to(event_role_factor.device)
            event_role_factor = event_role_factor * torch.unsqueeze(
                torch.unsqueeze(torch.unsqueeze(tl_valid_pattern_mask, 0), 0), 0)
            # event_role_factor = torch.where(tl_valid_pattern_mask>0, event_role_factor, torch.tensor(-1111.).to(event_role_factor.device))
            # event_role_factor[:,:,:,:,0] = torch.tensor(0.).to(event_role_factor.device)


        # tre_pairs_mask: B x T x E
        tre_pairs_mask = torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
        event_role_factor = event_role_factor * torch.unsqueeze(torch.unsqueeze(tre_pairs_mask, -1), -1)

        return event_role_factor


    # ------------------------------------------------------------------------------------------------
    def role_entity_factor_score(self, trigger_reps, entity_reps):
        trigger_reps = self.trigger_le_W(trigger_reps)
        entity_reps =  self.entity_le_W(entity_reps)
        role_type_reps = self.role_type_le_W  # 23 x k
        entity_type_reps = self.entity_type_le_W # 8 x k

        # breakpoint()
        # (B x T x E x 23 x 8) * (23 x 8) le_valid_pattern_mask
        role_entity_factor = contract('bnk,bmk,lk,ek-> bnmle', trigger_reps, entity_reps, role_type_reps,
                                            entity_type_reps)
        if self.valid_role_entity:
            le_valid_pattern_mask = self.role_entity_factor_mask()
            le_valid_pattern_mask = le_valid_pattern_mask.to(role_entity_factor.device)
            role_entity_factor = role_entity_factor * torch.unsqueeze(
                torch.unsqueeze(torch.unsqueeze(le_valid_pattern_mask, 0), 0), 0)
       
        # tre_pairs_mask: B x T x E
        tre_pairs_mask = torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
        role_entity_factor = role_entity_factor * torch.unsqueeze(torch.unsqueeze(tre_pairs_mask, -1), -1)
        # breakpoint()

        return role_entity_factor


    # ------------------------------------------------------------------------------------------------
    def event_role_entity_factor_score(self, trigger_reps, entity_reps):
        # 
        tre_trigger_reps = self.trigger_tre_W(trigger_reps)
        tre_trigger_reps = self.trigger_tre_dropout(tre_trigger_reps)  # B x T x k
        tre_entity_reps =  self.entity_tre_W(entity_reps)
        tre_entity_reps = self.entity_tre_dropout(tre_entity_reps)  # B x E x k
        event_type_reps = self.event_type_W  # 34-1 x k
        role_type_reps = self.role_type_W  # 23 x k
        entity_type_reps = self.entity_type_W  # 8-1 x k

        # (B x T x E x 34 x 23 x 8) * (34 x 23 x 8) tre_valid_pattern_mask
        event_role_entity_factor = contract('bnk,bmk,tk,rk,ek-> bnmtre', tre_trigger_reps, tre_entity_reps, event_type_reps,
                                            role_type_reps, entity_type_reps)
        # 
        # if self.valid_role_entity and self.valid_event_role:
        #     tre_valid_pattern_mask = self.event_role_entity_factor_mask()
        #     tre_valid_pattern_mask = tre_valid_pattern_mask.to(event_role_entity_factor.device)
        #     event_role_entity_factor = event_role_entity_factor * torch.unsqueeze(
        #         torch.unsqueeze(torch.unsqueeze(tre_valid_pattern_mask, 0), 0), 0)
       
        # tre_pairs_mask: B x T x E
        tre_pairs_mask = torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
        event_role_entity_factor = event_role_entity_factor * torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(tre_pairs_mask, -1), -1), -1)

        return event_role_entity_factor


    # ------------------------------------------------------------------------------------------------
    def event_entity_sibling_factor_score(self, trigger_reps, entity_reps):
        batch_size, trigger_num, _ = trigger_reps.shape
        batch_size, entity_num, _ = entity_reps.shape
        sib_trigger_reps = self.trigger_sib_W(trigger_reps)   # B x T x k
        sib_trigger_reps = self.trigger_sib_dropout(sib_trigger_reps)
        sib_entity_reps = self.entity_sib_W(entity_reps)  # B x E x k
        sib_entity_reps = self.entity_sib_dropout(sib_entity_reps)

        role_reps = self.role_type_sib_W
        role_num = role_reps.shape[0]
        # (B x T x R x R x E x E)
        # 
        event_role_entity_factor = contract('bnk,rk,ek,bik,bjk-> bnreij', sib_trigger_reps, role_reps, role_reps,
                                            sib_entity_reps, sib_entity_reps)
        # 
        # event_role_entity_factor = torch.reshape(event_role_entity_factor, (-1,entity_num,entity_num))
        # B x T x R x R x E x E
        event_role_entity_factor = event_role_entity_factor * \
                                   torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                       torch.ones(entity_num, entity_num).to(
                                           event_role_entity_factor.device).fill_diagonal_(0), 0), 0), 0), 0)

        # B x T x R x E x R x E
        event_role_entity_factor = torch.transpose(event_role_entity_factor, 3, 4)
        # B*T x R*E x R*E
        event_role_entity_factor = torch.reshape(event_role_entity_factor,
                                                 (-1, entity_num * role_num, entity_num * role_num))
        # btr_1e_1r_2e_2 = btr_2e_2r_1e_1 : bt3122 = bt2231
        event_role_entity_factor = torch.triu(event_role_entity_factor) + torch.transpose(
            torch.triu(event_role_entity_factor), 1, 2)
        # B x T x R x E x R x E
        event_role_entity_factor = torch.reshape(event_role_entity_factor,
                                                 (batch_size, trigger_num, role_num, entity_num, role_num, entity_num))

        # B x T x R x E x R x E -> B x T x E x E x R x R
        event_role_entity_factor = event_role_entity_factor.permute(0, 1, 3, 5, 2, 4)
        # 
        # tre_pairs_mask: B x T x E x E
        tee_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.entity_masks, 1) * torch.unsqueeze(self.entity_masks, -1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.trigger_masks, -1), -1)
        event_role_entity_factor = event_role_entity_factor * torch.unsqueeze(torch.unsqueeze(tee_triples_mask, -1), -1)

        return event_role_entity_factor


    # ------------------------------------------------------------------------------------------------
    def event_entity_coparent_factor_score(self, trigger_reps, entity_reps):
        batch_size, trigger_num, _ = trigger_reps.shape
        batch_size, entity_num, _ = entity_reps.shape
        cop_trigger_reps = self.trigger_cop_W(trigger_reps)  # B x T x k
        cop_trigger_reps = self.trigger_cop_dropout(cop_trigger_reps)
        cop_entity_reps = self.entity_cop_W(entity_reps)  # B x E x k
        cop_entity_reps = self.entity_cop_dropout(cop_entity_reps)
        role_reps = self.role_type_cop_W
        role_num = role_reps.shape[0]
        # (B x E x R x R x T x T)
        # 
        event_role_entity_factor = contract('bek,mk,nk,bik,bjk-> bemnij', cop_entity_reps, role_reps, role_reps,
                                            cop_trigger_reps, cop_trigger_reps)
        
        # B x E x R x R x T x T
        event_role_entity_factor = event_role_entity_factor * torch.unsqueeze(torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(
                torch.ones(trigger_num, trigger_num).to(event_role_entity_factor.device).fill_diagonal_(0), 0), 0), 0),
            0)
        # B x E x R x T x R x T
        event_role_entity_factor = torch.transpose(event_role_entity_factor, 3, 4)
        # B*E x R*T x R*T
        event_role_entity_factor = torch.reshape(event_role_entity_factor,
                                                 (-1, trigger_num * role_num, trigger_num * role_num))
        # btr_1e_1r_2e_2 = btr_2e_2r_1e_1 : bt3122 = bt2231
        event_role_entity_factor = torch.triu(event_role_entity_factor) + torch.transpose(
            torch.triu(event_role_entity_factor), 1, 2)

        # B x E x R x T x R x T
        event_role_entity_factor = torch.reshape(event_role_entity_factor,
                                                 (batch_size, entity_num, role_num, trigger_num, role_num, trigger_num))

        # B x E x R x T x R x T -> B x T x T x E x R x R
        event_role_entity_factor = event_role_entity_factor.permute(0, 3, 5, 1, 2, 4)
        # 
        # tre_pairs_mask: B x T x T x E
        tte_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.trigger_masks, -1), -1)
        event_role_entity_factor = event_role_entity_factor * torch.unsqueeze(torch.unsqueeze(tte_triples_mask, -1), -1)

        return event_role_entity_factor


    # ------------------------------------------------------------------------------------------------
    def entity_relation_entity_factor_score(self, entity_reps):
        batch_size, entity_num, _ = entity_reps.shape
        
        if self.config.new_potential:
            # breakpoint()
            entity_relation_entity_factor = contract('tk,rk,vk->trv', self.entity_type_ere_W, self.relation_type_ere_W, self.entity_type_ere_W)
            entity_relation_entity_factor = entity_relation_entity_factor.repeat(batch_size, entity_num, entity_num, 1, 1, 1)
        else:
            if self.config.relation_directional:
                ere_entity_start_reps = self.entity_start_ere_W(entity_reps)  # B x E x k
                ere_entity_start_reps = self.ere_start_dropout(ere_entity_start_reps)
                ere_entity_end_reps = self.entity_end_ere_W(entity_reps)  # B x E x k
                ere_entity_end_reps = self.ere_end_dropout(ere_entity_end_reps)
            else:
                ere_entity_start_reps = self.entity_start_ere_W(entity_reps)
            if self.config.share_relation_type_reps:
                ere_entity_type_reps = self.entity_type_ere_W(self.unary_entity_type_reps[1:])  # 8-1 x k
                ere_relation_type_reps = self.relation_type_ere_W(self.unary_relation_type_reps)  # 7 x k
            else:
                ere_entity_type_reps = self.entity_type_ere_W  # 8-1 x k
                ere_relation_type_reps = self.relation_type_ere_W  # 7 x k
            entity_type_num = ere_entity_type_reps.shape[0]
            relation_type_num = ere_relation_type_reps.shape[0]

            # (B x E x E x 8 x 7 x 8) * (8 x 7 x 8) ere_valid_pattern_mask
            if self.config.relation_directional:
                entity_relation_entity_factor = contract('bsk,bek,tk,rk,vk->bsetrv', ere_entity_start_reps, ere_entity_end_reps,
                                                        ere_entity_type_reps, ere_relation_type_reps, ere_entity_type_reps)
            else:
                entity_relation_entity_factor = contract('bsk,bek,tk,rk,vk->bsetrv', ere_entity_start_reps, ere_entity_start_reps,
                                                     ere_entity_type_reps, ere_relation_type_reps, ere_entity_type_reps)
        # breakpoint()
        # if self.valid_relation_entity:
        #     ere_valid_pattern_mask = self.entity_relation_entity_factor_mask()
        #     ere_valid_pattern_mask = ere_valid_pattern_mask.to(entity_relation_entity_factor.device)
        #     entity_relation_entity_factor = entity_relation_entity_factor * torch.unsqueeze(
        #         torch.unsqueeze(torch.unsqueeze(ere_valid_pattern_mask, 0), 0), 0)

        # ere_pairs_mask: B x E x E
        ere_pairs_mask = torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
        entity_relation_entity_factor = entity_relation_entity_factor * torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(ere_pairs_mask, -1), -1), -1)

        # self-loop mask
        entity_relation_entity_factor = entity_relation_entity_factor.permute(0, 3, 4, 5, 1, 2)  # B x 8 x 7 x 8 x E x E
        entity_relation_entity_factor = entity_relation_entity_factor * \
                                        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                            torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                            (entity_relation_entity_factor.device), 0), 0), 0), 0)

        if not self.relation_directional:
            # symmetry
            entity_relation_entity_factor = torch.reshape(entity_relation_entity_factor, (-1, entity_num, entity_num))
            entity_relation_entity_factor = torch.triu(entity_relation_entity_factor) + torch.transpose(
                torch.triu(entity_relation_entity_factor), 1, 2)
            entity_relation_entity_factor = torch.reshape(entity_relation_entity_factor,
                                                          (batch_size, entity_type_num, relation_type_num,
                                                           entity_type_num, entity_num, entity_num))

        entity_relation_entity_factor = entity_relation_entity_factor.permute(0, 4, 5, 1, 2, 3)

        return entity_relation_entity_factor


    # ------------------------------------------------------------------------------------------------
    def entity_relation_factor_score(self, entity_reps):
        batch_size, entity_num, _ = entity_reps.shape
        er_start_entity_reps = self.start_entity_er_W(entity_reps)  # B x E x k
        er_start_entity_reps = self.er_start_dropout(er_start_entity_reps)
        er_end_entity_reps = self.end_entity_er_W(entity_reps)  # B x E x k
        er_end_entity_reps = self.er_end_dropout(er_end_entity_reps)
        entity_type_reps = self.entity_type_er_W  # 8-1 x k
        if self.config.share_relation_type_reps:
            relation_type_reps = self.relation_type_er_W(self.relation_type_reps)
        else:
            relation_type_reps = self.relation_type_er_W  # 7 x k


        entity_type_num = entity_type_reps.shape[0]
        relation_type_num = relation_type_reps.shape[0]

        # (B x E x E x 8 x 7) * (8 x 7) re_valid_pattern_mask
        if self.config.test_er:
            start_entity_relation_factor = contract('bsk,nk,rk->bsnr', er_start_entity_reps, entity_type_reps, relation_type_reps)
            end_entity_relation_factor = contract('bsk,nk,rk->bsnr', er_end_entity_reps, entity_type_reps, relation_type_reps)
            start_entity_relation_factor = torch.unsqueeze(start_entity_relation_factor,2).repeat(1,1,entity_num,1,1)
            end_entity_relation_factor = torch.unsqueeze(end_entity_relation_factor,1).repeat(1,entity_num,1,1,1)
            # breakpoint()
            # er_pairs_mask: B x E x E
            ere_pairs_mask = torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
            start_entity_relation_factor = start_entity_relation_factor * torch.unsqueeze(torch.unsqueeze(ere_pairs_mask, -1), -1)
            start_entity_relation_factor = start_entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x 8 x 7 x E x E
            start_entity_relation_factor = start_entity_relation_factor * \
                                    torch.unsqueeze(torch.unsqueeze(
                                        torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                        (start_entity_relation_factor.device), 0), 0), 0)
            start_entity_relation_factor = start_entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x E x E x 8 x 7

            end_entity_relation_factor = end_entity_relation_factor * torch.unsqueeze(torch.unsqueeze(ere_pairs_mask, -1), -1)
            end_entity_relation_factor = end_entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x 8 x 7 x E x E
            end_entity_relation_factor = end_entity_relation_factor * \
                                    torch.unsqueeze(torch.unsqueeze(
                                        torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                        (end_entity_relation_factor.device), 0), 0), 0)
            end_entity_relation_factor = end_entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x E x E x 8 x 7
            return start_entity_relation_factor, end_entity_relation_factor
        else:
            entity_relation_factor = contract('bsk,bek,nk,rk->bsenr', er_start_entity_reps, er_end_entity_reps,
                                          entity_type_reps, relation_type_reps)
        
            # if self.valid_relation_entity:
            #     er_valid_pattern_mask = self.entity_relation_factor_mask()
            #     er_valid_pattern_mask = er_valid_pattern_mask.to(entity_relation_factor.device) 
            #     entity_relation_factor = entity_relation_factor * torch.unsqueeze(
            #                                 torch.unsqueeze(torch.unsqueeze(er_valid_pattern_mask, 0), 0), 0)

            # er_pairs_mask: B x E x E
            ere_pairs_mask = torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1)
            entity_relation_factor = entity_relation_factor * torch.unsqueeze(torch.unsqueeze(ere_pairs_mask, -1), -1)

            entity_relation_factor = entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x 8 x 7 x E x E
            entity_relation_factor = entity_relation_factor * \
                                    torch.unsqueeze(torch.unsqueeze(
                                        torch.unsqueeze(torch.ones(entity_num, entity_num).fill_diagonal_(0).to
                                                        (entity_relation_factor.device), 0), 0), 0)

            if not self.config.relation_directional:
                # symmetry
                entity_relation_factor = torch.reshape(entity_relation_factor, (-1, entity_num, entity_num))
                entity_relation_factor = torch.triu(entity_relation_factor) + torch.transpose(
                    torch.triu(entity_relation_factor), 1, 2)
                entity_relation_factor = torch.reshape(entity_relation_factor,
                                                    (batch_size, entity_type_num, relation_type_num,
                                                        entity_num, entity_num))

            entity_relation_factor = entity_relation_factor.permute(0, 3, 4, 1, 2)  # B x E x E x 8 x 7

            return entity_relation_factor


    # ------------------------------------------------------------------------------------------------
    def entity_relation_gp_factor_score(self, entity_reps):
        batch_size, entity_num, _ = entity_reps.shape
        entity_start_gp_reps = self.entity_start_re_gp_W(entity_reps)  # B x E x k
        entity_start_gp_reps = self.gp_start_dropout(entity_start_gp_reps)
        entity_mid_gp_reps = self.entity_mid_re_gp_W(entity_reps) # B x E x k
        entity_mid_gp_reps = self.gp_mid_dropout(entity_mid_gp_reps)
        entity_end_gp_reps = self.entity_end_re_gp_W(entity_reps) # B x E x k
        entity_end_gp_reps = self.gp_end_dropout(entity_end_gp_reps)
        if self.config.decomp:
            entity_start_gp_reps = self.entity_start_re_gp_decomp_ffn(entity_start_gp_reps)
            entity_mid_gp_reps = self.entity_mid_re_gp_decomp_ffn(entity_mid_gp_reps)
            entity_end_gp_reps = self.entity_end_re_gp_decomp_ffn(entity_end_gp_reps)

        if self.config.share_relation_type_reps:
            relation_gp_reps = self.relation_type_re_gp_W(self.unary_relation_type_reps)
        else:
            relation_gp_reps = self.relation_type_re_gp_W
        relation_num = relation_gp_reps.shape[0]

        # B x S x M x E x L1 x L2
        entity_relation_gp_factor = contract('bsk,bmk,bek,ik,jk-> bsmeij', entity_start_gp_reps, entity_mid_gp_reps, entity_end_gp_reps,
                                            relation_gp_reps, relation_gp_reps)
        eee_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.entity_masks, 1) * torch.unsqueeze(self.entity_masks, -1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.entity_masks, -1), -1)
        
        # B x S x M x E x L1 x L2
        entity_relation_gp_factor = entity_relation_gp_factor * torch.unsqueeze(torch.unsqueeze(eee_triples_mask, -1), -1)                                        
     
        # B x S x L1 x L2 x M x E
        entity_relation_gp_factor = entity_relation_gp_factor.permute((0,1,4,5,2,3)) * \
                                                   torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                                       torch.ones(entity_num, entity_num).to(
                                                            entity_relation_gp_factor.device).fill_diagonal_(0), 0), 0), 0), 0)

        # B x E x L1 x L2 x S x M
        entity_relation_gp_factor = entity_relation_gp_factor.permute((0,5,2,3,1,4)) * \
                                                        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                                                torch.ones(entity_num, entity_num).to(
                                                                    entity_relation_gp_factor.device).fill_diagonal_(0), 0), 0), 0), 0)

        entity_relation_gp_factor = entity_relation_gp_factor.permute((0,4,5,1,2,3))

        # # B x S x L1 x L2 x M x E
        # entity_relation_gp1_factor = torch.permute(entity_relation_gp_factor,(0,1,4,5,2,3)) * \
        #                                            torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
        #                                                torch.ones(entity_num, entity_num).to(
        #                                                     entity_relation_gp_factor.device).fill_diagonal_(0), 0), 0), 0), 0)
        # # B x S x M x E x L1 x L2
        # entity_relation_gp1_factor = torch.permute(entity_relation_gp1_factor,(0,1,4,5,2,3))
        # B x S x M x E
        
        # entity_relation_gp1_factor = entity_relation_gp1_factor * torch.unsqueeze(torch.unsqueeze(tee_triples_mask, -1), -1)                                        

        # # B x E x L1 x L2 x S x M         
        # entity_relation_gp2_factor = torch.permute(entity_relation_gp_factor,(0,3,4,5,1,2)) * \
        #                                            torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
        #                                                 torch.ones(entity_num, entity_num).to(
        #                                                     entity_relation_gp_factor.device).fill_diagonal_(0), 0), 0), 0), 0)
        # # B x S x M x E x L1 x L2
        # entity_relation_gp2_factor = torch.permute(entity_relation_gp2_factor,(0,4,5,1,2,3))
        
        # entity_relation_gp2_factor = entity_relation_gp2_factor * torch.unsqueeze(torch.unsqueeze(tte_triples_mask, -1), -1)

        return entity_relation_gp_factor


    # ------------------------------------------------------------------------------------------------
    def entity_relation_sib_factor_score(self, entity_reps):    
        batch_size, entity_num, _ = entity_reps.shape
        # breakpoint()
        entity_start_sib_reps = self.entity_start_re_sib_W(entity_reps)  # B x E x k
        entity_start_sib_reps = self.sib_start_dropout(entity_start_sib_reps)
        
        entity_end_sib_reps = self.entity_end_re_sib_W(entity_reps) # B x E x k
        entity_end_sib_reps = self.sib_end_dropout(entity_end_sib_reps)
        if self.config.decomp:
            entity_start_sib_reps = self.entity_start_re_sib_decomp_ffn(entity_start_sib_reps)
            entity_end_sib_reps = self.entity_end_re_sib_decomp_ffn(entity_end_sib_reps)


        if self.config.share_relation_type_reps:
            relation_sib_reps = self.relation_type_re_sib_W(self.unary_relation_type_reps)
        else:
            relation_sib_reps = self.relation_type_re_sib_W
            #---------------------------------------------------------------------------
            # relation_sib_reps = self.relation_type_sib_dropout(relation_sib_reps)
            #---------------------------------------------------------------------------
        # self.debug['relation_reps'] = relation_reps
        relation_num = relation_sib_reps.shape[0]
        # (B x E x R x R x E x E)
        # 
        entity_relation_sib_factor = contract('bnk,rk,ek,bik,bjk-> bnreij', entity_start_sib_reps, relation_sib_reps, relation_sib_reps,
                                            entity_end_sib_reps, entity_end_sib_reps)
        # 
        # event_role_entity_factor = torch.reshape(event_role_entity_factor, (-1,entity_num,entity_num))
        # B x T x R x R x E x E
        entity_relation_sib_factor = entity_relation_sib_factor * \
                                   torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                       torch.ones(entity_num, entity_num).to(
                                           entity_relation_sib_factor.device).fill_diagonal_(0), 0), 0), 0), 0)

        # B x E x R x E x R x E
        entity_relation_sib_factor = torch.transpose(entity_relation_sib_factor, 3, 4)
        # B*E x R*E x R*E
        entity_relation_sib_factor = torch.reshape(entity_relation_sib_factor,
                                                 (-1, entity_num * relation_num, entity_num * relation_num))
        # btr_1e_1r_2e_2 = btr_2e_2r_1e_1 : bt3122 = bt2231
        entity_relation_sib_factor = torch.triu(entity_relation_sib_factor) + torch.transpose(
            torch.triu(entity_relation_sib_factor), 1, 2)
        # B x E x R x E x R x E
        entity_relation_sib_factor = torch.reshape(entity_relation_sib_factor,
                                                 (batch_size, entity_num, relation_num, entity_num, relation_num, entity_num))

        # B x E x R x E x R x E -> B x E x E x E x R x R
        entity_relation_sib_factor = entity_relation_sib_factor.permute(0, 1, 3, 5, 2, 4)
        # 
        # tre_pairs_mask: B x E x E x E
        tee_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.entity_masks, 1) * torch.unsqueeze(self.entity_masks, -1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.entity_masks, -1), -1)
        entity_relation_sib_factor = entity_relation_sib_factor * torch.unsqueeze(torch.unsqueeze(tee_triples_mask, -1), -1)

        return entity_relation_sib_factor


    # ------------------------------------------------------------------------------------------------
    def entity_relation_cop_factor_score(self, entity_reps): 
        batch_size, entity_num, _ = entity_reps.shape
        entity_start_cop_reps = self.entity_start_re_cop_W(entity_reps)  # B x E x k
        entity_start_cop_reps = self.cop_start_dropout(entity_start_cop_reps)
        entity_end_cop_reps = self.entity_end_re_cop_W(entity_reps) # B x E x k
        entity_end_cop_reps = self.cop_end_dropout(entity_end_cop_reps)
        if self.config.decomp:
            entity_start_cop_reps = self.entity_start_re_cop_decomp_ffn(entity_start_cop_reps)
            entity_end_cop_reps = self.entity_end_re_cop_decomp_ffn(entity_end_cop_reps)


        if self.config.share_relation_type_reps:
            relation_cop_reps = self.relation_type_re_cop_W(self.unary_relation_type_reps)
        else:
            relation_cop_reps = self.relation_type_re_cop_W
        relation_num = relation_cop_reps.shape[0]
        # (B x E x R x R x T x T)
        # 
        entity_relation_cop_factor = contract('bek,mk,nk,bik,bjk-> bemnij', entity_end_cop_reps, relation_cop_reps, relation_cop_reps,
                                            entity_start_cop_reps, entity_start_cop_reps)
        
        # B x E x R x R x T x T
        entity_relation_cop_factor = entity_relation_cop_factor * torch.unsqueeze(torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(
                torch.ones(entity_num, entity_num).to(entity_relation_cop_factor.device).fill_diagonal_(0), 0), 0), 0),
            0)
        # B x E x R x T x R x T
        entity_relation_cop_factor = torch.transpose(entity_relation_cop_factor, 3, 4)
        # B*E x R*T x R*T
        entity_relation_cop_factor = torch.reshape(entity_relation_cop_factor,
                                                 (-1, entity_num * relation_num, entity_num * relation_num))
        # btr_1e_1r_2e_2 = btr_2e_2r_1e_1 : bt3122 = bt2231
        entity_relation_cop_factor = torch.triu(entity_relation_cop_factor) + torch.transpose(
            torch.triu(entity_relation_cop_factor), 1, 2)

        # B x E x R x T x R x T
        entity_relation_cop_factor = torch.reshape(entity_relation_cop_factor,
                                                 (batch_size, entity_num, relation_num, entity_num, relation_num, entity_num))

        # B x E x R x T x R x T -> B x T x T x E x R x R
        entity_relation_cop_factor = entity_relation_cop_factor.permute(0, 3, 5, 1, 2, 4)
        # 
        # tre_pairs_mask: B x T x T x E
        tte_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.entity_masks, -1), -1)
        entity_relation_cop_factor = entity_relation_cop_factor * torch.unsqueeze(torch.unsqueeze(tte_triples_mask, -1), -1)

        return entity_relation_cop_factor


    # ------------------------------------------------------------------------------------------------
    def event_relation_cop_factor_score(self, trigger_reps, entity_reps): 
        batch_size, entity_num, _ = entity_reps.shape
        batch_size, trigger_num, _ = trigger_reps.shape
        trigger_start_cop_reps = self.trigger_start_re_cop_W(trigger_reps)  # B x T x k
        trigger_start_cop_reps = self.cop_trigger_start_dropout(trigger_start_cop_reps)
        entity_start_cop_reps = self.entity_start_re_cop_W(entity_reps)  # B x E x k
        entity_start_cop_reps = self.cop_start_dropout(entity_start_cop_reps)
        entity_end_cop_reps = self.entity_end_re_cop_W(entity_reps) # B x E x k
        entity_end_cop_reps = self.cop_end_dropout(entity_end_cop_reps)
        if self.config.decomp:
            entity_start_cop_reps = self.entity_start_re_cop_decomp_ffn(entity_start_cop_reps)
            entity_end_cop_reps = self.entity_end_re_cop_decomp_ffn(entity_end_cop_reps)

        relation_cop_reps = self.relation_type_re_cop_W
        role_cop_reps = self.role_type_re_cop_W
        # if self.config.share_relation_type_reps:
        #     relation_cop_reps = self.relation_type_re_cop_W(self.unary_relation_type_reps)
        # else:
        #     relation_cop_reps = self.relation_type_re_cop_W
        relation_num = relation_cop_reps.shape[0]
        role_num = role_cop_reps.shape[0]
        # (B x T x E x E x R1 x R2 )
        
        # breakpoint()
        event_relation_cop_factor = contract('bik,bjk,bek,mk,nk->bijemn',trigger_start_cop_reps, entity_start_cop_reps, entity_end_cop_reps,role_cop_reps, relation_cop_reps)
    
        # tre_pairs_mask: B x T x E x E
        tee_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.entity_masks, 1) * torch.unsqueeze(self.entity_masks, -1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.trigger_masks, -1), -1)
        event_relation_cop_factor = event_relation_cop_factor * torch.unsqueeze(torch.unsqueeze(tee_triples_mask, -1), -1)

        return event_relation_cop_factor


    # ------------------------------------------------------------------------------------------------
    def event_relation_gp_factor_score(self, trigger_reps, entity_reps): 
        batch_size, entity_num, _ = entity_reps.shape
        batch_size, trigger_num, _ = trigger_reps.shape
        trigger_start_gp_reps = self.trigger_start_re_gp_W(trigger_reps)  # B x T x k
        trigger_start_gp_reps = self.gp_trigger_start_dropout(trigger_start_gp_reps)
        entity_start_gp_reps = self.entity_start_re_gp_W(entity_reps)  # B x E x k
        entity_start_gp_reps = self.gp_start_dropout(entity_start_gp_reps)
        entity_end_gp_reps = self.entity_end_re_gp_W(entity_reps) # B x E x k
        entity_end_gp_reps = self.gp_end_dropout(entity_end_gp_reps)

        relation_gp_reps = self.relation_type_re_gp_W
        role_gp_reps = self.role_type_re_gp_W

        relation_num = relation_gp_reps.shape[0]
        role_num = role_gp_reps.shape[0]
        # (B x T x E x E x R1 x R2 )
        # 
        event_relation_gp_factor = contract('bik,bsk,bek,mk,nk-> bisemn',trigger_start_gp_reps, entity_start_gp_reps, entity_end_gp_reps, 
                                            role_gp_reps, relation_gp_reps)
    
        # tre_pairs_mask: B x T x E x E
        tee_triples_mask = torch.unsqueeze(
            torch.unsqueeze(self.entity_masks, 1) * torch.unsqueeze(self.entity_masks, -1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(self.trigger_masks, -1), -1)
        event_relation_gp_factor = event_relation_gp_factor * torch.unsqueeze(torch.unsqueeze(tee_triples_mask, -1), -1)

        return event_relation_gp_factor

    # ------------------------------------------------------------------------------------------------
    def mfvi(self, event_logits, role_logits, entity_logits, relation_logits, event_role_factor=None,
             role_entity_factor=None,event_role_entity_factor=None, event_role_sibling_factor=None, 
             event_role_cop_factor=None,entity_relation_entity_factor=None, entity_relation_factor=None,relation_entity_factor=None,
             entity_relation_sib_factor=None,entity_relation_cop_factor=None,entity_relation_gp_factor=None,start_entity_relation_factor=None,
             end_entity_relation_factor=None,event_relation_cop_factor = None, event_relation_gp_factor = None):
        """
                :param event_logits: B x |T| x (33+1)
                :param role_logits: B x |T|*|E| x (22+1)
                :param entity_logits: B x |E| x (7+1)
        """

        batch_size, trigger_num, _ = event_logits.shape
        _, entity_num, _ = entity_logits.shape
        role_logits = torch.reshape(role_logits, (batch_size, trigger_num, entity_num, -1))
        relation_logits = torch.reshape(relation_logits, (batch_size, entity_num, entity_num, -1))
        entity_q = entity_logits
        event_q = event_logits
        role_q = role_logits
        relation_q = relation_logits
        # self.debug['relation_unary'] = relation_logits

        for i in range(self.config.mfvi_iter):
        # for i in range(0):
            if self.config.asynchronous:
                
                entity_q = torch.nn.functional.softmax(entity_q, dim=-1)* torch.unsqueeze(self.entity_masks, -1)
        
                # Relation Extraction----------------------------------------------------
                if self.use_high_order_ere and entity_relation_entity_factor is not None:
                    relation_ere_F = contract('bijern,bie,bjn->bijr', entity_relation_entity_factor, entity_q, entity_q)
                    #---------sib or cop to refine relation---------------------------------
                    if self.use_high_order_re_sibling and entity_relation_sib_factor is not None:
                        relation_sib_F = contract('btnmlr, btnl->btmr', entity_relation_sib_factor, relation_q)
                    else:
                        relation_sib_F = 0
                    if self.use_high_order_re_coparent and entity_relation_cop_factor is not None:
                        relation_cop_F = contract('btselr, bser->btel', entity_relation_cop_factor, relation_q)
                    else:
                        relation_cop_F = 0

                    relation_q = relation_logits+relation_ere_F + relation_sib_F + relation_cop_F
                    relation_q_p = torch.nn.functional.softmax(relation_q, dim=-1) * torch.unsqueeze(
                                torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                    entity_start_ere_F = contract('bijern,bjn,bijr->bie', entity_relation_entity_factor, entity_q,
                                                    relation_q_p)
                    entity_end_ere_F = contract('bijern,bie,bijr->bjn', entity_relation_entity_factor, entity_q,
                                                    relation_q_p)
                    entity_q = (entity_logits + 0.5*entity_start_ere_F + 0.5*entity_end_ere_F)
                if self.use_high_order_er and (entity_relation_factor is not None or start_entity_relation_factor is not None):
                    if self.config.new_potential:
                        relation_er_F1 = contract('bijer,bie->bijr', entity_relation_factor, entity_q) # B x E x E x EL x RL
                        relation_er_F2 = contract('bijre,bje->bijr', relation_entity_factor, entity_q) # B x E x E x RL x EL
                        relation_q = (relation_logits + relation_er_F1 + relation_er_F2)
                        relation_q_p = torch.nn.functional.softmax(relation_q, dim=-1) * torch.unsqueeze(
                            torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                        entity_er_F1 = contract('bijer,bijr->bie', entity_relation_factor, relation_q_p)
                        entity_er_F2 = contract('bijre,bijr->bje', relation_entity_factor, relation_q_p)
                        entity_q = (entity_logits + entity_er_F1 + entity_er_F2)
                    else:
                        if self.config.test_er:
                            # breakpoint()
                            relation_er_F1 = contract('bijer,bie->bijr', start_entity_relation_factor, entity_q)
                            relation_er_F2 = contract('bijer,bje->bijr', end_entity_relation_factor, entity_q)
                            relation_q = relation_logits + relation_er_F1 + relation_er_F2
                            relation_q_p = torch.nn.functional.softmax(relation_q, dim=-1) * torch.unsqueeze(
                                torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                            entity_er_F1 = contract('bijer,bijr->bie', start_entity_relation_factor, relation_q_p)
                            entity_er_F2 =  contract('bijer,bijr->bje', end_entity_relation_factor, relation_q_p)
                            entity_q = (entity_logits + entity_er_F1 + entity_er_F2)
                        else:
                            relation_er_F1 = contract('bijer,bie->bijr', entity_relation_factor, entity_q)
                            # relation_er_F2 = contract('bijer,bje->bijr', entity_relation_factor, entity_q)
                            relation_q = relation_logits + relation_er_F1 #+ relation_er_F2)
                            relation_q_p = torch.nn.functional.softmax(relation_q, dim=-1) * torch.unsqueeze(
                                torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                            entity_er_F1 = contract('bijer,bijr->bie', entity_relation_factor, relation_q_p)
                            entity_er_F2 =  contract('bijer,bijr->bje', entity_relation_factor, relation_q_p)
                            entity_q = (entity_logits + entity_er_F1 + entity_er_F2)
                        # breakpoint()
            

                # Event Extraction----------------------------------------------------
                if self.use_high_order_tre and event_role_entity_factor is not None:
                    role_tre_F = contract('bnmtre, bnt, bme -> bnmr', event_role_entity_factor, event_q,
                                            entity_q)  # B x T x E x 23
                    
                    #---------sib to refine role---------------------------------
                    if self.use_high_order_sibling and event_role_sibling_factor is not None:
                        role_tre_sib_F = contract('btnmlr, btnl->btmr', event_role_sibling_factor, role_q)
                    else:
                        role_tre_sib_F = 0

                    role_q = role_logits + self.alpha_role_tre * role_tre_F + self.alpha_role_sib * role_tre_sib_F
                    # role_q = role_logits + self.config.alpha_role_sib * role_tre_sib_F
                    role_q_p = torch.nn.functional.softmax(role_q, dim=-1) * torch.unsqueeze(
                            torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                    event_tre_F = contract('bnmtre, bnmr, bme -> bnt', event_role_entity_factor,
                                                # *torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.event_role_entity_factor_mask_nonrole().to(event_role_entity_factor.device),0),0),0), 
                                                role_q_p,
                                                entity_q)  # B x T x 34
                    entity_tre_F = contract('bnmtre, bnt, bnmr -> bme', event_role_entity_factor,
                                                # *torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.event_role_entity_factor_mask_nonrole().to(event_role_entity_factor.device),0),0),0), 
                                                event_q,
                                                role_q_p)  # B x E x 8
                    event_q = event_logits + self.alpha_event_tre*event_tre_F
                    entity_q = entity_logits + self.alpha_entity_tre*entity_tre_F
                    # breakpoint()

            else:
                if self.config.prob_damp:
                    if i == 0:
                        former_entity_q = torch.nn.functional.softmax(entity_q, dim=-1)* torch.unsqueeze(self.entity_masks, -1)
                        former_role_q = torch.nn.functional.softmax(role_q, dim=-1) * torch.unsqueeze(
                                        torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)

                    entity_q = 0.5*torch.nn.functional.softmax(entity_q, dim=-1)* torch.unsqueeze(self.entity_masks, -1)+0.5*former_entity_q
                    role_q = 0.5*torch.nn.functional.softmax(role_q, dim=-1) * torch.unsqueeze(
                            torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)+0.5*former_role_q

                    former_entity_q = entity_q
                    former_role_q = role_q
                    # breakpoint()
                else:
                    entity_q = torch.nn.functional.softmax(entity_q, dim=-1)* torch.unsqueeze(self.entity_masks, -1)
                    # entity_q = Sparsemax()(entity_q)
                    role_q = torch.nn.functional.softmax(role_q, dim=-1) * torch.unsqueeze(
                            torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                    # role_q = Sparsemax()(role_q) * torch.unsqueeze(
                    #     torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                event_q = torch.nn.functional.softmax(event_q, dim=-1) * torch.unsqueeze(self.trigger_masks, -1)
                relation_q = torch.nn.functional.softmax(relation_q, dim=-1)* torch.unsqueeze(
                    torch.unsqueeze(self.entity_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
                

                if self.use_high_order_tl and event_role_factor is not None:
                    # breakpoint()
                    event_tl_F = contract('bnmtl, bnml -> bnt', event_role_factor, role_q) # B x T x 34
                    role_tl_F = contract('bnmtl, bnt -> bnml', event_role_factor, event_q) # B x T x E x 23
                else:
                    event_tl_F = 0 
                    role_tl_F = 0

                if self.use_high_order_le and role_entity_factor is not None:
                    role_le_F = contract('bnmle, bme -> bnml', role_entity_factor, entity_q) # B x T x E x 23
                    entity_le_F = contract('bnmle, bnml -> bme', role_entity_factor, role_q) # B x E x 8
                    # breakpoint()     
                else:
                    role_le_F = 0
                    entity_le_F = 0
                
                # self.use_high_order_tre = False
                if self.use_high_order_tre and event_role_entity_factor is not None:
                    event_tre_F = contract('bnmtre, bnmr, bme -> bnt', event_role_entity_factor, role_q,
                                        entity_q)  # B x T x 34
                    role_tre_F = contract('bnmtre, bnt, bme -> bnmr', event_role_entity_factor, event_q,
                                        entity_q)  # B x T x E x 23
                    entity_tre_F = contract('bnmtre, bnt, bnmr -> bme', event_role_entity_factor, event_q,
                                            role_q)  # B x E x 8
                    # 
                else:
                    event_tre_F = 0
                    role_tre_F = 0
                    entity_tre_F = 0

                # self.use_high_order_sibling = False
                if self.use_high_order_sibling and event_role_sibling_factor is not None:
                    # breakpoint()
                    role_tre_sibling_F = contract('btnmlr, btnl->btmr', event_role_sibling_factor, role_q)
                else:
                    role_tre_sibling_F = 0

                if self.use_high_order_coparent and event_role_cop_factor is not None:
                    # 
                    role_tre_cop_F = contract('btselr, bser->btel', event_role_cop_factor, role_q)                
                else:
                    role_tre_cop_F = 0

                if self.use_high_order_ere and entity_relation_entity_factor is not None:
                    relation_ere_F = contract('bijern,bie,bjn->bijr', entity_relation_entity_factor, entity_q, entity_q)
                    if self.config.relation_directional:
                        entity_start_ere_F = contract('bijern,bjn,bijr->bie', entity_relation_entity_factor, entity_q,
                                                    relation_q)
                        entity_end_ere_F = contract('bijern,bie,bijr->bjn', entity_relation_entity_factor, entity_q,
                                                    relation_q)
                    else:
                        entity_start_ere_F = contract('bijern,bjn,bijr->bie', entity_relation_entity_factor, entity_q,
                                                    relation_q)
                        entity_end_ere_F = 0
                else:
                    relation_ere_F = 0
                    entity_start_ere_F = 0
                    entity_end_ere_F = 0

                if self.use_high_order_er and entity_relation_factor is not None:
                    if self.config.new_potential:
                        relation_er_F1 = contract('bijer,bie->bijr', entity_relation_factor, entity_q) # B x E x E x EL x RL
                        relation_er_F2 = contract('bijre,bje->bijr', relation_entity_factor, entity_q) # B x E x E x RL x EL
                        entity_er_F1 = contract('bijer,bijr->bie', entity_relation_factor, relation_q)
                        entity_er_F2 = contract('bijre,bijr->bje', relation_entity_factor, relation_q)
                    else:
                        relation_er_F1 = contract('bijer,bie->bijr', entity_relation_factor, entity_q)
                        relation_er_F2 = contract('bijer,bje->bijr', entity_relation_factor, entity_q)
                        entity_er_F1 = contract('bijer,bijr->bie', entity_relation_factor, relation_q)
                        entity_er_F2 =  contract('bijer,bijr->bje', entity_relation_factor, relation_q)
                        # breakpoint()
                        # entity_er_F1 = 0
                        # entity_er_F2 = 0
                    # breakpoint()
                else:
                    relation_er_F1 = 0
                    relation_er_F2 = 0
                    entity_er_F1 = 0
                    entity_er_F2 = 0

                if self.use_high_order_re_sibling and entity_relation_sib_factor is not None:
                    # breakpoint()
                    relation_sib_F = contract('btnmlr, btnl->btmr', entity_relation_sib_factor, relation_q)
                    # if i == 0:
                    #     self.debug['sib_message_1'] = relation_sib_F
                    # elif i == 1:
                    #     self.debug['sib_message_2'] = relation_sib_F
                    # elif i == 2:
                    #     self.debug['sib_message_3'] = relation_sib_F
                else:
                    relation_sib_F = 0

                if self.use_high_order_re_coparent and entity_relation_cop_factor is not None:
                    relation_cop_F = contract('btselr, bser->btel', entity_relation_cop_factor, relation_q)
                    # breakpoint()
                    # if i == 0:
                    #     self.debug['cop_message_1'] = relation_cop_F
                    # elif i == 1:
                    #     self.debug['cop_message_2'] = relation_cop_F
                    # elif i == 2:
                    #     self.debug['cop_message_3'] = relation_cop_F
                else:
                    relation_cop_F = 0

                if self.use_high_order_re_grandparent and entity_relation_gp_factor is not None:
                    relation_gp1_F = contract('bsmeij, bsmi->bmej', entity_relation_gp_factor, relation_q)
                    relation_gp2_F = contract('bsmeij, bmej->bsmi', entity_relation_gp_factor, relation_q)
                    # breakpoint()
                else:
                    relation_gp1_F = 0
                    relation_gp2_F = 0

                if self.use_high_order_rr_coparent and event_relation_cop_factor is not None:
                    # breakpoint()
                    role_rel_cop_F = contract('btserl, bsel->bter', event_relation_cop_factor, relation_q) 
                    rel_role_cop_F = contract('btserl, bter->bsel', event_relation_cop_factor, role_q)
                else:
                    role_rel_cop_F = 0
                    rel_role_cop_F = 0

                if self.use_high_order_rr_grandparent and event_relation_gp_factor is not None:
                    role_rel_gp_F = contract('btserl, bsel->btsr', event_relation_gp_factor, relation_q) 
                    rel_role_gp_F = contract('btserl, btsr->bsel', event_relation_gp_factor, role_q)
                else:
                    role_rel_gp_F = 0
                    rel_role_gp_F = 0

                if self.config.score_damp:
                    if i == 0:
                        former_role_le_F = role_le_F
                        former_entity_le_F = entity_le_F

                    event_q = event_logits + event_tre_F + event_tl_F 
                    role_q = role_logits + role_tre_F + role_tl_F + (0.5*role_le_F+0.5*former_role_le_F) + role_tre_sibling_F + role_tre_cop_F 
                    entity_q = entity_logits + entity_tre_F + (0.5*entity_le_F+0.5*former_entity_le_F) + entity_start_ere_F + entity_end_ere_F + entity_er_F
                    relation_q = relation_logits + relation_ere_F + relation_er_F1 + relation_er_F2 + relation_sib_F + relation_cop_F
                    
                    former_role_le_F = role_le_F
                    former_entity_le_F = entity_le_F
                else:
                    # breakpoint()
                    event_q = event_logits + event_tre_F + event_tl_F 
                    role_q = role_logits + role_rel_cop_F + role_rel_gp_F + role_tre_F + role_tl_F + role_le_F + self.config.alpha_role_sib*role_tre_sibling_F + self.config.alpha_role_cop*role_tre_cop_F 
                    entity_q = entity_logits + entity_tre_F + entity_le_F + entity_start_ere_F + entity_end_ere_F + entity_er_F1 + entity_er_F2
                    relation_q = relation_logits + rel_role_cop_F + rel_role_gp_F + relation_ere_F + relation_er_F1 + relation_er_F2 + relation_sib_F + relation_cop_F + relation_gp1_F + relation_gp2_F


            # if self.config.scaled:
            #     breakpoint()
            #     entity_q = entity_q / 5
            #     role_q = role_q / 5
            #     event_q = event_q / 5
            #     relation_q = relation_q / 5
        
        # role_q_test = torch.nn.functional.softmax(role_q, dim=-1) * torch.unsqueeze(
        #     torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1)
        # entity_q_test= torch.nn.functional.softmax(entity_q, dim=-1)* torch.unsqueeze(self.entity_masks, -1)
        
        # test_potential = torch.sum(
        #     role_entity_factor*torch.unsqueeze(role_q_test,-1)
        #     *torch.unsqueeze(torch.unsqueeze(entity_q_test,1),-2),dim=(0,1,2))/torch.sum(torch.unsqueeze(
        #     torch.unsqueeze(self.trigger_masks, -1) * torch.unsqueeze(self.entity_masks, 1), -1))
        # self.test_potential.append(test_potential)
        # breakpoint()
       
        # breakpoint()
        role_q = torch.reshape(role_q, (batch_size, trigger_num * entity_num, -1))
        relation_q = torch.reshape(relation_q, (batch_size, entity_num * entity_num, -1))
        return event_q, role_q, entity_q, relation_q


class Ident(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()

        # vocabularies
        self.config = config
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.trigger_label_stoi = vocabs['trigger_label']
        # self.mention_type_stoi = vocabs['mention_type']
        # self.entity_type_stoi = vocabs['entity_type']
        # self.event_type_stoi = vocabs['event_type']
        # self.relation_type_stoi = vocabs['relation_type']
        # self.role_type_stoi = vocabs['role_type']
        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.trigger_label_itos = {i: s for s, i in self.trigger_label_stoi.items()}
        # self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        # self.event_type_itos = {i: s for s, i in self.event_type_stoi.items()}
        # self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        # self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.trigger_label_num = len(self.trigger_label_stoi)
        # self.mention_type_num = len(self.mention_type_stoi)
        # self.entity_type_num = len(self.entity_type_stoi)
        # self.event_type_num = len(self.event_type_stoi)
        # self.relation_type_num = len(self.relation_type_stoi)
        # self.role_type_num = len(self.role_type_stoi)

        # BERT encoder
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        if 'albert' in config.bert_model_name:
            self.bert = AlbertModel(bert_config)
        elif 'roberta' in config.bert_model_name:
            self.bert = RobertaModel(bert_config)
        elif 'scibert' in config.bert_model_name:
            # breakpoint()
            self.bert = BertModel(bert_config) 
        else:
            self.bert = BertModel(bert_config)
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        # local classifiers
        # self.use_entity_type = config.use_entity_type
        # self.binary_dim = self.bert_dim * 2
        linear_bias = config.linear_bias

        self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num,
                                          bias=linear_bias)
        self.trigger_label_ffn = nn.Linear(self.bert_dim, self.trigger_label_num,
                                           bias=linear_bias)


        # others
        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)
        self.trigger_crf = CRF(self.trigger_label_stoi, bioes=False)
        self.pad_vector = nn.Parameter(torch.randn(1, 1, self.bert_dim))



    def load_bert(self, name, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('Loading pre-trained BERT model {}'.format(name))
        if 'scibert' in name:
            self.bert = AutoModel.from_pretrained(name,cache_dir=cache_dir)
        elif 'albert' in name:
            self.bert = AlbertModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                                )
        elif 'roberta' in name:
            self.bert = RobertaModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                                )
        else:
            self.bert = BertModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                              output_hidden_states=True)

    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks,output_hidden_states=True)
        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def forward(self, batch):
        # encoding
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()
        # entity type indices -> one hot
        # entity_types = batch.entity_type_idxs.view(batch_size, -1)
        # entity_types = torch.clamp(entity_types, min=0)
        # entity_types_onehot = bert_outputs.new_zeros(*entity_types.size(),
        #                                              self.entity_type_num)
        # entity_types_onehot.scatter_(2, entity_types.unsqueeze(-1), 1)
        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)

        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        entity_label_loglik = self.entity_crf.loglik(entity_label_scores,
                                                     batch.entity_label_idxs,
                                                     batch.token_nums)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        trigger_label_loglik = self.trigger_crf.loglik(trigger_label_scores,
                                                       batch.trigger_label_idxs,
                                                       batch.token_nums)

        loss = - entity_label_loglik.mean() - trigger_label_loglik.mean()
        # loss = - trigger_label_loglik.mean()
        # print(loss)
        return loss

    def predict(self, batch):
        self.eval()

        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()


        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)
        _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                 batch.token_nums)
        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)
        triggers = tag_paths_to_spans(trigger_label_preds,
                                      batch.token_nums,
                                      self.trigger_label_stoi)

        # 
        batch_graphs = [{'entity': entities[i], 'trigger': triggers[i]} for i in range(len(entities))]


        self.train()
        return batch_graphs

    def provide(self, batch):
        self.eval()
        # breakpoint()
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()


        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)
        _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                 batch.token_nums)
        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)
        triggers = tag_paths_to_spans(trigger_label_preds,
                                      batch.token_nums,
                                      self.trigger_label_stoi)

        # 
        # batch_graphs = [{'entity': entities[i], 'trigger': triggers[i]} for i in range(len(entities))]

        return entities, triggers

