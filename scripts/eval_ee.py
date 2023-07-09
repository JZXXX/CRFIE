import sys
from tkinter import E
sys.path.append('/data1/zhangsc/deepN/highIE-event')
import json
import graph
import scorer
def eval_ee_same_file(predicted_file):
    gold_graphs = []
    eval_graphs = []
    with open(predicted_file, 'r') as pf:
        for line in pf:
            sent = json.loads(line)
            
            pred = sent['pred']
            eval_entities = [entity[0:3] for entity in pred['entities']]
            eval_triggers = pred['triggers']#[trigger for trigger in pred['triggers']]
            eval_roles = pred['roles']#[role for role in pred['roles']]
            eval_relations = pred['relations']

            gold = sent['gold']
            gold_entities = [entity[0:3] for entity in gold['entities']]
            gold_triggers = gold['triggers']#[trigger for trigger in gold['triggers']]
            gold_roles = gold['roles']#[role for role in gold['roles']]
            gold_relations = gold['relations']

            # breakpoint()
           
            eval_graphs.append(graph.Graph(eval_entities,eval_triggers,eval_relations,eval_roles,{}))
            gold_graphs.append(graph.Graph(gold_entities,gold_triggers,gold_relations,gold_roles,{}))
   
    scorer.score_graphs(gold_graphs,eval_graphs)
    return gold_graphs, eval_graphs


def eval_ee(gold_file, eval_file):
    gold_graphs = []
    eval_graphs = []
    
    compare_read_gold = []
    with open(gold_file, 'r') as gf:
        # total_gold_entities = 0
        # total_gold_triggers = 0

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
                gold_entities.append([entity['start'],entity['end'],entity['entity_type']])

            events = sent['event_mentions']
            # if len(events) > 1:
            #     pdb.set_trace()
            for event in events:
                trigger = event['trigger']
                event_type = event['event_type']
                gold_triggers.append([trigger['start'],trigger['end'],event_type])

                arguments = event['arguments']
                for argument in arguments:
                    gold_roles.append([len(gold_triggers)-1,entities_id.index(argument['entity_id']),argument['role']])
                    # gold_entities.append((trigger['start'], id2entities[argument['entity_id']]['start'],id2entities[argument['entity_id']]['end'],argument['role']))
            
            relations = sent['relation_mentions']
            for relation in relations:
                relation_type = relation['relation_type']
                start_entities = entities_id.index(relation['arguments'][0]['entity_id'])
                end_entities = entities_id.index(relation['arguments'][1]['entity_id'])
                gold_relations.append([start_entities,end_entities,relation_type])


            
            # total_gold_entities += len(gold_roles)
            # # breakpoint()
            # total_gold_triggers += len(gold_triggers)




            gold_graphs.append(graph.Graph(gold_entities,gold_triggers,gold_relations,gold_roles,{}))
            
            compare_read_gold.append({'sent_id':sent['sent_id'],'triggers':gold_triggers,'roles':gold_roles})
           
   


    with open(eval_file, 'r') as ef:
        total_eval_entities = 0
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
            eval_roles = [role[0:3] for role in pred['roles']]
            eval_relations = [relation[0:3] for relation in pred['relations']]

            # for i in sent['gold']['roles']:
            #     if not i in compare_read_gold[sent_num]['roles']:
            #         print(sent['sent_id'])
            #         print(compare_read_gold[sent_num]['sent_id'])
            #         print(sent['gold']['roles'])
            #         print(compare_read_gold[sent_num]['roles'])
            #         print(sent['pred']['roles'])
            #         print('\n')
            #         break
            # for i in compare_read_gold[sent_num]['roles']:
            #     if not i in sent['gold']['roles']:
            #         print(sent['sent_id'])
            #         # print(sent['sentence'])
            #         print(compare_read_gold[sent_num]['sent_id'])
            #         print(sent['gold']['roles'])
            #         print(compare_read_gold[sent_num]['roles'])
            #         print(sent['pred']['roles'])
            #         print('\n')
            #         break
            # sent_num += 1

            # breakpoint()
            # total_eval_entities += len(eval_roles)
            # total_eval_triggers += len(eval_triggers)
            eval_graphs.append(graph.Graph(eval_entities,eval_triggers,eval_relations,eval_roles,{}))
    # print(total_eval_entities)
    # print(total_gold_entities)
    # print(total_eval_triggers)
    # print(total_gold_triggers)

    # sent_idx = 0
    # another_gold_graphs, another_eval_graphs = eval_ee_same_file(eval_file)
    # for eval_graph, another_eval_graph in zip(eval_graphs, another_eval_graphs):
    #     if len(eval_graph.relations) != len(another_eval_graph.relations):
    #         breakpoint()
    # for gold_graph, another_gold_graph in zip(gold_graphs, another_gold_graphs):
    #     sent_idx += 1
    #     if len(gold_graph.relations) != len(another_gold_graph.relations):
    #         breakpoint()
    # breakpoint()
    scores = scorer.score_graphs(gold_graphs,eval_graphs)
    return scores



def data_analysis(gold_file):
    with open(gold_file, 'r') as gf:
        total_event_roles = {}
        total_role_entities = {}

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
                gold_entities.append([entity['start'],entity['end'],entity['entity_type']])

            events = sent['event_mentions']
            # if len(events) > 1:
            #     pdb.set_trace()
            for event in events:
                trigger = event['trigger']
                event_type = event['event_type']

                if not event_type in total_event_roles:
                    total_event_roles[event_type] = {'nums':1}
                else:
                    total_event_roles[event_type]['nums']+=1

                gold_triggers.append([trigger['start'],trigger['end'],event_type])

                arguments = event['arguments']
                for argument in arguments:
                    # if event_type == 'Movement:Transport' and argument['role'] == 'Place':
                    #     breakpoint()
                    if not argument['role'] in total_event_roles[event_type]:
                        total_event_roles[event_type][argument['role']] = 1
                    else:
                        total_event_roles[event_type][argument['role']] += 1



                    if not argument['role'] in total_role_entities:
                        total_role_entities[argument['role']] = {'nums':1}
                    else:
                        total_role_entities[argument['role']]['nums'] += 1
                    argument_type = gold_entities[entities_id.index(argument['entity_id'])][-1]
                    if not argument_type in total_role_entities[argument['role']]:
                        # breakpoint()
                        total_role_entities[argument['role']][argument_type] = 1
                    else:
                        total_role_entities[argument['role']][argument_type] += 1


                    gold_roles.append([len(gold_triggers)-1,entities_id.index(argument['entity_id']),argument['role']])
                    # gold_entities.append((trigger['start'], id2entities[argument['entity_id']]['start'],id2entities[argument['entity_id']]['end'],argument['role']))
            
            relations = sent['relation_mentions']
            for relation in relations:
                relation_type = relation['relation_type']
                start_entities = entities_id.index(relation['arguments'][0]['entity_id'])
                end_entities = entities_id.index(relation['arguments'][1]['entity_id'])
                gold_relations.append([start_entities,end_entities,relation_type])

    return total_event_roles, total_role_entities

def eval_analysis(eval_file):
    with open(eval_file, 'r') as ef:
        total_eval_event_roles = {}
        total_eval_role_entities = {}
        sent_num = 0
        for line in ef:
            sent = json.loads(line)
            try:
                pred = sent['pred']
            except:
                pred = sent['graph']
            eval_entities = [entity[0:3] for entity in pred['entities']]
            eval_triggers = [trigger[0:3] for trigger in pred['triggers']]
            eval_roles = [role[0:3] for role in pred['roles']]
            eval_relations = [relation[0:3] for relation in pred['relations']]
            for role in eval_roles:
                event_type = eval_triggers[role[0]][-1]
                if not event_type in total_eval_event_roles:
                    total_eval_event_roles[event_type] = {'nums':1}
                else:
                    total_eval_event_roles[event_type]['nums']+=1
                if not role[-1] in total_eval_event_roles[event_type]:
                        total_eval_event_roles[event_type][role[-1]] = 1
                else:
                    total_eval_event_roles[event_type][role[-1]] += 1
                entity_type = eval_entities[role[1]][-1]
                role_type = role[-1]
                if not role_type in total_eval_role_entities:
                    total_eval_role_entities[role_type] = {'nums':1}
                else:
                    total_eval_role_entities[role_type]['nums'] += 1
                if not entity_type in total_eval_role_entities[role_type]:
                    total_eval_role_entities[role_type][entity_type] = 1
                else:
                    total_eval_role_entities[role_type][entity_type] += 1
    return total_eval_event_roles, total_eval_role_entities



def show_gold_event_role(total_event_roles, patterns):
    
    new_patterns = {}
    constrained_roles = []
    wrong_roles = []
    total_events = 0
    total_roles = 0
    for event in patterns:
        new_patterns[event] = {}
        if event in total_event_roles:
            new_patterns[event]['event_nums'] = total_event_roles[event]['nums']
            total_events += total_event_roles[event]['nums']
            for role in total_event_roles[event]:
                if (not role in patterns[event]) and role != 'nums':
                    breakpoint()
                    constrained_roles.append(patterns[event])
                    wrong_roles.append(total_event_roles[event])
                    break
        for role in patterns[event]:
            new_patterns[event][role] = 0
            if event in total_event_roles:
                if role in total_event_roles[event]:
                    new_patterns[event][role] = total_event_roles[event][role]
                    total_roles += total_event_roles[event][role]
    print(constrained_roles)
    print(wrong_roles)
    print(total_events)
    print(total_roles)
    
    json.dump(new_patterns, open('scripts/analysis-dev.json','w'), indent = 6)



def show_gold_role_entity(total_role_entities, patterns):
    new_patterns = {}
    
    total_roles = 0
    total_arguments = 0
    for role in patterns:
        new_patterns[role] = {}
        if role in total_role_entities:
            new_patterns[role]['role_nums'] = total_role_entities[role]['nums']
            total_roles += total_role_entities[role]['nums']
            for entity in total_role_entities[role]:
                if (not entity in patterns[role]) and entity != 'nums':
                    breakpoint()
                    
        for arg in patterns[role]:
            new_patterns[role][arg] = 0
            if role in total_role_entities:
                if arg in total_role_entities[role]:
                    new_patterns[role][arg] = total_role_entities[role][arg]
                    total_arguments += total_role_entities[role][arg]
    
    print(total_arguments)
    print(total_roles)
    
    json.dump(new_patterns, open('scripts/role_entity/analysis-test.json','w'), indent = 6)



def show_eval_event_role(total_eval_event_roles, patterns):
    constrained_wrong = {}
    for event in total_eval_event_roles:
        constrained_wrong[event] = {}
        for role in total_eval_event_roles[event]:
            if (not role in patterns[event]) and (role != 'nums'):
                constrained_wrong[event][role] = total_eval_event_roles[event][role]
    print(constrained_wrong)

def show_eval_role_entity(total_eval_role_entities, patterns):
    constrained_wrong = {}
    for role in total_eval_role_entities:
        constrained_wrong[role] = {}
        for entity in total_eval_role_entities[role]:
            if (not entity in patterns[role]) and (entity != 'nums'):
                constrained_wrong[role][entity] = total_eval_role_entities[role][entity]
    print(constrained_wrong)

def confusion(gold,eval):
    import numpy as np
    import matplotlib.pyplot as plt

    role_type = {1: 'Beneficiary', 2: 'Person', 3: 'Agent', 4: 'Recipient', 5: 'Buyer', 6: 'Giver', 7: 'Adjudicator', 8: 'Defendant', 9: 'Seller', 10: 'Instrument', 11: 'Destination', 12: 'Place', 13: 'Attacker', 14: 'Victim', 15: 'Org', 16: 'Target', 17: 'Entity', 18: 'Vehicle', 19: 'Origin', 20: 'Artifact', 21: 'Prosecutor', 22: 'Plaintiff', 0: 'O'}
    entity_type = {1: 'FAC', 2: 'WEA', 3: 'PER', 4: 'GPE', 5: 'LOC', 6: 'VEH', 7: 'ORG', 0: 'O'}
    


    def plot_confusion_matrix(first_matrix):
        fig, ax = plt.subplots()
        im = ax.imshow(first_matrix)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(entity_type.keys())))
        ax.set_yticks(np.arange(len(role_type.keys())))
        # ... and label them with the respective list entries
        ax.set_xticklabels(entity_type.keys())
        ax.set_yticklabels(role_type.keys())

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(entity_type.keys())):
            for j in range(len(role_type.keys())):
                text = ax.text(j, i, first_matrix[i, j],
                            ha="center", va="center", color="w")

        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plt.show()


    def draw():
        
        first_matrix = np.zeros([8,23], dtype=int)
        for i in entity_type:
            for j in role_type:
                first_matrix[i][j] = gold[j][i]
                # if not i==j:
                #     first_matrix[i][j] = label_dict[label_name1[i]][label_name1[j]]

        print(first_matrix)
        plot_confusion_matrix(first_matrix)

        # second_matrix = np.zeros([10,10],dtype=int)
        # for i in range(10):
        #     for j in range(10):
        #         if not i==j:
        #             try:
        #                 second_matrix[i][j] = label_dict[label_name2[i]][label_name2[j]]
        #             except:
        #                 second_matrix[i][j] = 0

        # print(second_matrix)
        # plot_confusion_matrix(second_matrix,label_name2)

    draw()

if __name__ == '__main__':
    # eval_ee_same_file(sys.argv[2])
    eval_ee(sys.argv[1],sys.argv[2])
    # eval_ee_same_file(sys.argv[1])
    # total_event_roles, total_role_entities = data_analysis(sys.argv[1])
    # # print(total_event_roles)
    # pattern_event_role = json.load(open(sys.argv[2]))
    # show_gold_role_entity(total_role_entities, patterns)

    # total_eval_event_roles, total_eval_role_entities = eval_analysis(sys.argv[1])
    # show_eval_event_role(total_eval_event_roles, json.load(open('resource/valid_patterns/event_role.json')))
    # show_eval_role_entity(total_eval_role_entities, json.load(open('resource/valid_patterns/role_entity.json')))
    # print(total_role_entities)
    # print(total_eval_role_entities)
    # confusion(total_role_entities,total_eval_role_entities)