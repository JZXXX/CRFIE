import copy
import json
import os
from re import T

from typing import Dict, Any

from transformers import (BertConfig, AutoConfig, RobertaConfig, XLMRobertaConfig,AlbertConfig,
                          PretrainedConfig)

class Config(object):
    def __init__(self, **kwargs):
        self.coref = kwargs.pop('coref', True)
        # bert
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-large-cased')
        self.bert_cache_dir = kwargs.pop('bert_cache_dir', None)
        self.extra_bert = kwargs.pop('extra_bert', -1)
        self.use_extra_bert = kwargs.pop('use_extra_bert', False)
        # global features
        self.use_global_features = kwargs.get('use_global_features', False)
        self.global_features = kwargs.get('global_features', [])
        # model
        self.multi_piece_strategy = kwargs.pop('multi_piece_strategy', 'first')
        self.bert_dropout = kwargs.pop('bert_dropout', .5)
        self.linear_dropout = kwargs.pop('linear_dropout', .4)
        self.linear_bias = kwargs.pop('linear_bias', True)
        self.linear_activation = kwargs.pop('linear_activation', "")
        self.entity_hidden_num = kwargs.pop('entity_hidden_num', 150)
        self.mention_hidden_num = kwargs.pop('mention_hidden_num', 150)
        self.event_hidden_num = kwargs.pop('event_hidden_num', 600)
        self.relation_hidden_num = kwargs.pop('relation_hidden_num', 150)
        self.role_hidden_num = kwargs.pop('role_hidden_num', 600)
        self.use_entity_type = kwargs.pop('use_entity_type', False)
        self.beam_size = kwargs.pop('beam_size', 5)
        self.beta_v = kwargs.pop('beta_v', 2)
        self.beta_e = kwargs.pop('beta_e', 2)
        self.relation_mask_self = kwargs.pop('relation_mask_self', True)
        self.relation_directional = kwargs.pop('relation_directional', False)
        self.symmetric_relations = set(kwargs.pop('symmetric_relations', ['PER-SOC']))
        # files
        self.train_file = kwargs.pop('train_file', None)
        self.dev_file = kwargs.pop('dev_file', None)
        self.test_file = kwargs.pop('test_file', None)
        self.valid_pattern_path = kwargs.pop('valid_pattern_path', None)
        self.log_path = kwargs.pop('log_path', None)
        # training
        self.accumulate_step = kwargs.pop('accumulate_step', 1)
        self.batch_size = kwargs.pop('batch_size', 10)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 5)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.bert_learning_rate = kwargs.pop('bert_learning_rate', 1e-5)
        self.weight_decay = kwargs.pop('weight_decay', 0.001)
        self.bert_weight_decay = kwargs.pop('bert_weight_decay', 0.00001)
        self.warmup_epoch = kwargs.pop('warmup_epoch', 5)
        self.grad_clipping = kwargs.pop('grad_clipping', 5.0)
        # others
        self.use_gpu = kwargs.pop('use_gpu', True)
        self.gpu_device = kwargs.pop('gpu_device', -1)

        #adding
        self.use_ere_biaffine = kwargs.pop('use_ere_biaffine',False)
        # self.use_high_order = kwargs.pop('use_high_order',False)
        self.split_train = kwargs.pop('split_train',True)
        self.use_high_order_tl = kwargs.pop('use_high_order_tl', False)
        self.use_high_order_le = kwargs.pop('use_high_order_le', False)
        self.use_high_order_tre = kwargs.pop('use_high_order_tre',False)
        self.use_high_order_sibling = kwargs.pop('use_high_order_sibling',False)
        self.use_high_order_coparent = kwargs.pop('use_high_order_coparent',False)
        self.use_high_order_ere = kwargs.pop('use_high_order_ere',False)
        self.use_high_order_er = kwargs.pop('use_high_order_er',False)
        self.use_high_order_re_sibling = kwargs.pop('use_high_order_re_sibling',False)
        self.use_high_order_re_coparent = kwargs.pop('use_high_order_re_coparent',False)
        self.use_high_order_re_grandparent = kwargs.pop('use_high_order_re_grandparent',False)
        self.use_high_order_rr_coparent = kwargs.pop('use_high_order_rr_coparent',False)
        self.use_high_order_rr_grandparent = kwargs.pop('use_high_order_rr_grandparent',False)
        
        self.decomp_size = kwargs.pop('decomp_size',300)
        self.tre_decomp_size = kwargs.pop('tre_decomp_size',150)
        self.mfvi_iter = kwargs.pop('mfvi_iter',1)
        self.event_classification = kwargs.pop('event_classification',True)
        self.relation_classification = kwargs.pop('relation_classification',True)
        self.rebatch = kwargs.pop('rebatch',False)
        self.entity_classification = kwargs.pop('entity_classification',True)
        self.trigger_maxent = kwargs.pop('trigger_maxent', False)
        self.new_potential = kwargs.pop('new_potential',False)
        self.penalized = kwargs.pop('penalized',False)
        self.score_damp = kwargs.pop('score_damp',False)
        self.prob_damp = kwargs.pop('prob_damp',False)
        self.scaled = kwargs.pop('scaled', False)
        self.use_guideliens = kwargs.pop('use_guideliens', False)
        self.guideline_path = kwargs.pop('guideline_path', '')
        self.asynchronous = kwargs.pop('asynchronous',False)
        self.split_rel_ident = kwargs.pop('split_rel_ident',False)
        self.new_score = kwargs.pop('new_score',False)
        self.share_relation_type_reps = kwargs.pop('share_relation_type_reps',False)
        self.test_er = kwargs.pop('test_er',False)
        self.decomp = kwargs.pop('decomp',False)
        self.alpha_role_sib = kwargs.pop('alpha_role_sib',1)
        self.alpha_role_cop = kwargs.pop('alpha_role_sib',1)
        self.alpha_entity_tre = kwargs.pop('alpha_entity_tre',1)
        self.alpha_event_tre = kwargs.pop('alpha_event_tre',1)
        self.alpha_role_tre = kwargs.pop('alpha_role_tre',1)
        self.train_alpha = kwargs.pop('train_alpha',False)
        self.gold_ent = kwargs.pop('gold_ent',False)
        
        
    @classmethod
    def from_dict(cls, dict_obj):
        """Creates a Config object from a dictionary.
        Args:
            dict_obj (Dict[str, Any]): a dict where keys are
        """
        config = cls()
        for k, v in dict_obj.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r', encoding='utf-8') as r:
            return cls.from_dict(json.load(r))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def save_config(self, path):
        """Save a configuration object to a file.
        :param path (str): path to the output file or its parent directory.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        print('Save config to {}'.format(path))
        with open(path, 'w', encoding='utf-8') as w:
            w.write(json.dumps(self.to_dict(), indent=2,
                               sort_keys=True))
    @property
    def bert_config(self):
        if self.bert_model_name.startswith('bert-'):
            return BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('roberta-'):
            return RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('xlm-roberta-'):
            return XLMRobertaConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('albert-'):
            return AlbertConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir)
        elif 'scibert' in self.bert_model_name:
            return AutoConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir)
        else:
            raise ValueError('Unknown model: {}'.format(self.bert_model_name))