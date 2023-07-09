import sys
sys.path.append('/data1/zhangsc/deepN/highIE-zsc')
import json
import graph
import scorer

def score_relations(gold_graphs, pred_graphs,
                 relation_directional=True, symmetric_relations = ["PER-SOC"]):
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = rel_match_num_plus = 0
    
    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
        # Entity
        gold_entities = gold_graph.entities
        pred_entities = pred_graph.entities
        true_pred_entities = []
        # if gold_entities:
        #     breakpoint()
        for entity in pred_entities:
            if not entity[-1] == 'O':
                # breakpoint()
                true_pred_entities.append(entity)
        gold_ent_num += len(gold_entities)
        pred_ent_num += len(true_pred_entities)
        ent_match_num += len([entity for entity in true_pred_entities
                              if entity in gold_entities])


        # Relation
        gold_relations = gold_graph.relations
        pred_relations = pred_graph.relations
        gold_rel_num += len(gold_relations)
        pred_rel_num += len(pred_relations)
        for arg1, arg2, rel_type in pred_relations:
            arg1_start, arg1_end, arg1_type = pred_entities[arg1]
            arg2_start, arg2_end, arg2_type = pred_entities[arg2]
            for arg1_gold, arg2_gold, rel_type_gold in gold_relations:
                arg1_start_gold, arg1_end_gold, arg1_gold_type = gold_entities[arg1_gold]
                arg2_start_gold, arg2_end_gold, arg2_gold_type = gold_entities[arg2_gold]
                if relation_directional:
                    if rel_type in symmetric_relations:
                        if ((arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold) or (
                            arg1_start == arg2_start_gold and
                            arg1_end == arg2_end_gold and
                            arg2_start == arg1_start_gold and
                            arg2_end == arg1_end_gold
                            )) and rel_type == rel_type_gold:
                            rel_match_num += 1
                        if ((arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and arg1_type == arg1_gold_type and 
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold and arg2_type == arg2_gold_type) or (
                            arg1_start == arg2_start_gold and
                            arg1_end == arg2_end_gold and arg1_type == arg2_gold_type and 
                            arg2_start == arg1_start_gold and
                            arg2_end == arg1_end_gold and arg2_type == arg1_gold_type
                            )) and rel_type == rel_type_gold:
                            rel_match_num_plus += 1
                            # breakpoint()
                            break
                    else:
                        if (arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold
                        ) and rel_type == rel_type_gold:
                            rel_match_num += 1
                            # break
                        if (arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and arg1_type == arg1_gold_type and 
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold and arg2_type == arg2_gold_type
                            ) and rel_type == rel_type_gold:
                            rel_match_num_plus += 1
                            break

                else:
                    if ((arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold) or (
                        arg1_start == arg2_start_gold and
                        arg1_end == arg2_end_gold and
                        arg2_start == arg1_start_gold and
                        arg2_end == arg1_end_gold
                    )) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break
    entity_prec, entity_rec, entity_f = scorer.compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    relation_prec, relation_rec, relation_f = scorer.compute_f1(
        pred_rel_num, gold_rel_num, rel_match_num)
    relation_prec_plus, relation_rec_plus, relation_f_plus = scorer.compute_f1(
        pred_rel_num, gold_rel_num, rel_match_num_plus)
    # print('Entity: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #     entity_prec * 100.0, entity_rec * 100.0, entity_f * 100.0))
    # print('Relation: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #     relation_prec * 100.0, relation_rec * 100.0, relation_f * 100.0))
    # print('Relation+: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #     relation_prec_plus * 100.0, relation_rec_plus * 100.0, relation_f_plus * 100.0))

    # breakpoint()
    score = [str('%.2f'%(entity_f * 100.0)), str(round((relation_f * 100.0),2)),str(format((relation_f_plus * 100.0),'.2f')) ]
    return score


def eval_relation(gold_file, eval_file):
    gold_graphs = []
    eval_graphs = []
    eval_sent_idxes = []
    with open(eval_file, 'r') as ef:
        total_eval_entities = 0
        total_eval_triggers = 0

        for line in ef:
            sent = json.loads(line)
            eval_sent_idxes.append(sent['sent_id'])
            try:
                pred = sent['pred']
            except:
                pred = sent['graph']
            eval_entities = [entity[:3] for entity in pred['entities']]
            eval_relations = [relation[:3] for relation in pred['relations']]


            eval_graphs.append(graph.Graph(eval_entities,[],eval_relations,[],{}))
    with open(gold_file, 'r') as gf:
        total_gold_entities = 0
        total_gold_relations = 0

        for line in gf:
            gold_entities = []
            gold_relations = []
            entities_id = []
            sent = json.loads(line)
            # if not sent['sent_id'] in eval_sent_idxes:
            #     print(sent['sent_id'])
            #     print(len(sent['pieces']))
            #     continue
            if len(sent['pieces'])>=128:
                # breakpoint()
                continue
            entities = sent['entity_mentions']

            # id2entities = {entity['id']: entity for entity in entities}
            for entity in entities:
                entities_id.append(entity['id'])
                gold_entities.append([entity['start'],entity['end'],entity['entity_type']])

            relations = sent['relation_mentions']
            # if len(events) > 1:
            #     pdb.set_trace()
            for relation in relations:
                relation_type = relation['relation_type']
                arguments = relation['arguments']
                gold_relations.append([entities_id.index(arguments[0]['entity_id']),entities_id.index(arguments[1]['entity_id']),relation_type])
                total_gold_relations += 1
            total_gold_entities += len(gold_entities)
                # pdb.set_trace()
            gold_graphs.append(graph.Graph(gold_entities,[],gold_relations,[],{}))

    # breakpoint()
    
   
    # score = score_relations(gold_graphs,eval_graphs, relation_directional=True, symmetric_relations = ["PER-SOC"])
    score = score_relations(gold_graphs,eval_graphs,relation_directional=True,symmetric_relations =["COMPARE","CONJUNCTION"])
    return score

if __name__ == '__main__':
    import sys
    _ = eval_relation(sys.argv[1],sys.argv[2])