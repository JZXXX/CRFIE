
import sys
sys.path.append('/data1/zhangsc/deepN/highIE-event')
import json

class Graph(object):
    def __init__(self, entities, triggers, relations, roles, tokens):
        """
        :param entities (list): A list of entities represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param triggers (list): A list of triggers represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param relations (list): A list of relations represented as a tuple of
        (entity_idx_1, entity_idx_2, label_idx). As we do not consider the
        direction of relations (list), it is better to have entity_idx_1 <
        entity_idx2.
        :param roles: A list of roles represented as a tuple of (trigger_idx_1,
        entity_idx_2, label_idx).
        :param tokens: Sentence tokens.
        """
        self.entities = entities
        self.triggers = triggers
        self.relations = relations
        self.roles = roles
        self.tokens = tokens




def read_gold_json_file(gold_file):
    gold_graphs = []
    with open(gold_file, 'r') as gf:
        total_gold_arguments = 0
        total_gold_triggers = 0

        for line in gf:
            gold_entities = []
            gold_triggers = []
            gold_roles = []
            gold_relations = []
            entities_id = []
            sent = json.loads(line)
            # breakpoint()
            entities = sent['entity_mentions']

            # id2entities = {entity['id']: entity for entity in entities}
            for entity in entities:
                entities_id.append(entity['id'])
                gold_entities.append([entity['start'], entity['end'], entity['entity_type']])

            events = sent['event_mentions']
            # if len(events) > 1:
            #     pdb.set_trace()
            for event in events:
                trigger = event['trigger']
                event_type = event['event_type']
                gold_triggers.append([trigger['start'], trigger['end'], event_type])

                arguments = event['arguments']
                for argument in arguments:
                    gold_roles.append(
                        [gold_triggers[len(gold_triggers) - 1][0], gold_triggers[len(gold_triggers) - 1][1],gold_triggers[len(gold_triggers) - 1][2], gold_entities[entities_id.index(argument['entity_id'])][0],gold_entities[entities_id.index(argument['entity_id'])][1],gold_entities[entities_id.index(argument['entity_id'])][2], argument['role']])
                    # gold_entities.append((trigger['start'], id2entities[argument['entity_id']]['start'],id2entities[argument['entity_id']]['end'],argument['role']))

            relations = sent['relation_mentions']
            for relation in relations:
                relation_type = relation['relation_type']
                start_entities = entities_id.index(relation['arguments'][0]['entity_id'])
                end_entities = entities_id.index(relation['arguments'][1]['entity_id'])
                gold_relations.append([gold_entities[start_entities][0], gold_entities[start_entities][1],gold_entities[start_entities][2],gold_entities[end_entities][0],gold_entities[end_entities][1],gold_entities[end_entities][2],relation_type])

            total_gold_arguments += len(gold_roles)
            # # breakpoint()
            total_gold_triggers += len(gold_triggers)

            gold_graph = Graph(gold_entities, gold_triggers, gold_relations, gold_roles, sent['tokens'])

            gold_graphs.append(gold_graph)
    # print(total_gold_arguments)
    # print(total_gold_triggers)
    return gold_graphs


def read_eval_json_file(eval_file):
    eval_graphs = []
    with open(eval_file, 'r') as ef:
        total_eval_arguments = 0
        total_eval_triggers = 0

        sent_num = 0

        for line in ef:
            sent = json.loads(line)

            try:
                pred = sent['pred']
            except:
                pred = sent['graph']
            eval_entities = [entity[0:3] for entity in pred['entities']]
            eval_triggers = [trigger[0:3] for trigger in pred['triggers']]
            eval_roles = []
            for role in pred['roles']:
                eval_role = []
                eval_role.extend(eval_triggers[role[0]])
                eval_role.extend(eval_entities[role[1]])
                eval_role.append(role[2])
                eval_roles.append(eval_role)
            # eval_roles = [role[0:3] for role in pred['roles']]
            eval_relations = []
            for relation in pred['relations']:
                eval_relation = []
                eval_relation.extend(eval_entities[relation[0]])
                eval_relation.extend(eval_entities[relation[1]])
                eval_relation.append(relation[2])
                eval_relations.append(eval_relation)
            # eval_relations = [relation[0:3] for relation in pred['relations']]

            total_eval_arguments += len(eval_roles)
            total_eval_triggers += len(eval_triggers)

            eval_graph = Graph(eval_entities, eval_triggers, eval_relations, eval_roles, sent['tokens'])

            eval_graphs.append(eval_graph)

    # print(total_eval_arguments)
    # print(total_eval_triggers)

    return eval_graphs


def case_analysis():


    # base_graphs = read_eval_json_file('log/dygie/baseline-gold-3/final.result.test.json')
    # high_graphs = read_eval_json_file('log/dygie/tre-gold-syn-noshare-1/final.result.test.json')
    # gold_graphs = read_gold_json_file('data/dygie/test.oneie.json')

    base_graphs = read_eval_json_file('ace05-R-baseline-test.json')
    high_graphs = read_eval_json_file('../highIE-zsc/log/ace05-R/res_sel/cop-ident2-noshare-150-iter2-3/final.result.test.json')
    gold_graphs = read_gold_json_file('data/ace05-R/test.albert.json')

    # base_graphs = read_eval_json_file('log/dygie/baseline-gold-3/final.result.test.json')
    # high_graphs = read_eval_json_file('log/dygie/rr-cop-gold-noshare-2/final.result.test.json')
    # gold_graphs = read_gold_json_file('data/dygie/test.oneie.json')


    res = {}
    idx = 0
    for base_graph, high_graph, gold_graph in zip(base_graphs, high_graphs, gold_graphs):
        assert base_graph.tokens == high_graph.tokens and base_graph.tokens == gold_graph.tokens, 'Not Align!'
        #------------role----------------------------------------------------------
        # for base_role, high_role in zip(base_graph.roles, high_graph.roles):
        #     if high_role in gold_graph.roles and (not base_role in gold_graph.roles):
        #         res[idx] = {'tokens': gold_graph.tokens,
        #                     'gold roles': gold_graph.roles,
        #                     'base roles':base_graph.roles,
        #                     'high roles':high_graph.roles
        #                     }
        #         idx += 1
        #         break
        #------------relation-----------------------------------------------------
        for base_rel, high_rel in zip(base_graph.relations, high_graph.relations):
            if high_rel in gold_graph.relations and (not base_rel in gold_graph.relations):
                res[idx] = {'tokens': gold_graph.tokens,
                            'gold rels': gold_graph.relations,
                            'base rels':base_graph.relations,
                            'high rels':high_graph.relations
                            }
                idx += 1
                break
        #------------role+relation-----------------------------------------------------
        # for base_role, high_role in zip(base_graph.roles, high_graph.roles):
        #     if high_role in gold_graph.roles and (not base_role in gold_graph.roles):
        #         arg_ent = high_role[3:6]
        #         high_rel_tails = [high_rel[3:6] for high_rel in high_graph.relations]
        #         # breakpoint()
        #         if arg_ent in high_rel_tails:
        #             res[idx] = {'tokens': gold_graph.tokens,
        #                         'gold roles': gold_graph.roles,
        #                         'base roles':base_graph.roles,
        #                         'high roles':high_graph.roles,
        #                         'gold rels': gold_graph.relations,
        #                         'base rels':base_graph.relations,
        #                         'high rels':high_graph.relations,
        #                         'tail ent':arg_ent
        #                         }
        #             idx += 1
        #             break

    json.dump(res, open('analysis_files/rel_case.json', 'w'), indent=4)
    return


def statistic_error(gold_graphs, eval_graphs):
    wrong_triggers_num = 0
    wrong_argument_num = 0
    wrong_only_role_num  = 0
    total_right_args_set = 0
    total_right_role_set = 0
    for gold_graph, eval_graph in zip(gold_graphs, eval_graphs):
        # event type (gold trigger identification)
        gold_triggers = gold_graph.triggers
        eval_triggers = eval_graph.triggers
        for eval_trigger in eval_triggers:
            if not eval_trigger in gold_triggers:
                wrong_triggers_num += 1
       
        # argument identification (trigger+args)
        gold_arguments_set = set([tuple(gold_role[:5]) for gold_role in gold_graph.roles])
        eval_arguments_set = set([tuple(eval_role[:5]) for eval_role in eval_graph.roles])
        right_args_set = gold_arguments_set&eval_arguments_set
        total_right_args_set += len(right_args_set)
        wrong_argument_num += max(len(gold_arguments_set)-len(right_args_set),len(eval_arguments_set)-len(right_args_set))


        gold_arguments_set = set([tuple(gold_role[:5]+[gold_role[6]]) for gold_role in gold_graph.roles])
        eval_arguments_set = set([tuple(eval_role[:5]+[eval_role[6]]) for eval_role in eval_graph.roles])
        right_args_set = gold_arguments_set&eval_arguments_set
        total_right_role_set += len(right_args_set)
        wrong_only_role_num += max(len(gold_arguments_set)-len(right_args_set),len(eval_arguments_set)-len(right_args_set))
       
        # # only role type (argument identification right but wrong role type)
        # gold_roles = gold_graph.roles
        # eval_roles = eval_graph.roles
        # for eval_role in eval_roles:
        #     for gold_role in gold_roles:
        #         if eval_role[:5] == gold_role[:5]:
        #             if eval_role[6] != gold_role[6]:
        #                 wrong_only_role_num += 1
        #         break
        # if gold_graph.roles:
        #     breakpoint()
    print(total_right_args_set)
    print(total_right_role_set)
    return wrong_triggers_num, wrong_argument_num, wrong_only_role_num


def error_analysis(gold_graphs,base_graphs,high_graphs):
    
    refine_triggers_num = 0
    refine_argument_num = 0
    refine_only_role_num  = 0
    for gold_graph, base_graph, high_graph in zip(gold_graphs, base_graphs, high_graphs):
        # event type
        gold_triggers = sorted(gold_graph.triggers)
        base_triggers = sorted(base_graph.triggers)
        high_triggers = sorted(high_graph.triggers)
        for i in range(len(gold_triggers)):
            if (base_triggers != gold_triggers) and (high_triggers == gold_triggers):
                refine_triggers_num += 1
       
        # argument identification
        gold_args = [gold_role[:5] for gold_role in gold_graph.roles]
        base_args = [base_role[:5] for base_role in base_graph.roles]
        high_args = [high_role[:5] for high_role in high_graph.roles]
        for gold_arg in gold_args:
            if (not gold_arg in base_args) and (gold_arg in high_args):
                refine_argument_num += 1

        # only role type
        # gold_roles = gold_graph.roles
        # base_roles = base_graph.roles
        # high_roles = high_graph.roles
        # for base_role in base_roles:
        #     for gold_role in gold_roles:
        #         for high_role in high_roles:
        #             if base_role[:5] == gold_role[:5] == high_role[:5]:
        #                 if (base_role[6] != gold_role[6]) and (high_role == gold_role):
        #                     refine_only_role_num += 1        
        #             break
        #         break
        gold_args = [gold_role[:5]+[gold_role[6]] for gold_role in gold_graph.roles]
        base_args = [base_role[:5]+[base_role[6]] for base_role in base_graph.roles]
        high_args = [high_role[:5]+[high_role[6]] for high_role in high_graph.roles]
        for gold_arg in gold_args:
            if (not gold_arg in base_args) and (gold_arg in high_args):
                refine_only_role_num += 1

    return refine_triggers_num, refine_argument_num, refine_only_role_num




# base_graphs = read_eval_json_file('log/dygie/baseline-gold-3/final.result.test.json')
# high_graphs = read_eval_json_file('log/dygie/tre+sib-gold-asyn-alpha-noshare-2/final.result.test.json')
# gold_graphs = read_gold_json_file('data/dygie/test.oneie.json')

# base_wrong = statistic_error(gold_graphs, base_graphs)
# high_wrong = statistic_error(gold_graphs, high_graphs)
# refine = error_analysis(gold_graphs,base_graphs,high_graphs)
# print(base_wrong)
# print(high_wrong)
# print(refine)

def sel_sent(file):
    sel_sents = {}
    idx = 1
    with open(file, 'r') as gf:
        total_gold_arguments = 0
        total_gold_triggers = 0

        for line in gf:
            gold_entities = []
            gold_triggers = []
            gold_roles = []
            gold_relations = []
            entities_id = []
            sent = json.loads(line)

            entities = sent['entity_mentions']

            # id2entities = {entity['id']: entity for entity in entities}
            for entity in entities:
                entities_id.append(entity['id'])
                gold_entities.append([entity['start'], entity['end'], entity['entity_type']])

            events = sent['event_mentions']
            # if len(events) > 1:
            #     pdb.set_trace()
            for event in events:
                trigger = event['trigger']
                event_type = event['event_type']
                gold_triggers.append([trigger['start'], trigger['end'], event_type])

                arguments = event['arguments']
                for argument in arguments:
                    gold_roles.append(
                        [gold_triggers[len(gold_triggers) - 1][0], gold_triggers[len(gold_triggers) - 1][1],gold_triggers[len(gold_triggers) - 1][2], gold_entities[entities_id.index(argument['entity_id'])][0],gold_entities[entities_id.index(argument['entity_id'])][1],gold_entities[entities_id.index(argument['entity_id'])][2], argument['role']])
                    # gold_entities.append((trigger['start'], id2entities[argument['entity_id']]['start'],id2entities[argument['entity_id']]['end'],argument['role']))

            relations = sent['relation_mentions']
            for relation in relations:
                relation_type = relation['relation_type']
                start_entities = entities_id.index(relation['arguments'][0]['entity_id'])
                end_entities = entities_id.index(relation['arguments'][1]['entity_id'])
                gold_relations.append([gold_entities[start_entities][0], gold_entities[start_entities][1],gold_entities[start_entities][2],gold_entities[end_entities][0],gold_entities[end_entities][1],gold_entities[end_entities][2],relation_type])

            if len(sent['tokens']) < 15:
                if len(gold_relations)>2:
                    sel_sents[idx] = {'tokens':" ".join(sent['tokens']),'roles':gold_roles,'relations':gold_relations}
                    idx += 1
    json.dump(sel_sents, open('analysis_files/sel_sents.json', 'w'), indent=4)
    return

sel_sent('data/dygie/train.oneie.json')