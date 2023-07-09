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
            tokens[i] = entity[2]
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

# a = [(1,3,3),(4,6,4),(8,9,1)]
# b = [(1,2,3),(5,6,2),(7,8,3)]
# res = remove_overlap_entities(a,b)
# print(res)
import torch
entity_masks = torch.tensor([[1,1,1,0]])
entity_num = 4
tee_triples_mask = torch.unsqueeze(
            torch.unsqueeze(entity_masks, 1) * torch.unsqueeze(entity_masks, -1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(entity_masks, -1), -1)
tte_triples_mask = torch.unsqueeze(
            torch.unsqueeze(entity_masks, -1) * torch.unsqueeze(entity_masks, 1), 1) \
                           * torch.unsqueeze(torch.unsqueeze(entity_masks, -1), -1)
print(tee_triples_mask)
print(tte_triples_mask)
entity_relation_gp_factor = torch.ones(1,4,4,4,2,2)
# B x S x M x E x L1 x L2
entity_relation_gp_factor  = torch.unsqueeze(torch.unsqueeze(tte_triples_mask, -1),-1)*entity_relation_gp_factor
# B x S x L1 x L2 x M x E
entity_relation_gp_factor = entity_relation_gp_factor.permute((0,1,4,5,2,3)) * \
                                                   torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                                       torch.ones(entity_num, entity_num).to(
                                                            entity_relation_gp_factor.device).fill_diagonal_(0), 0), 0), 0), 0)

# entity_relation_gp_factor = entity_relation_gp_factor.permute((0,1,4,5,2,3))
# B x E x L1 x L2 x S x M
entity_relation_gp_factor = entity_relation_gp_factor.permute((0,5,2,3,1,4)) * \
                                                   torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                                                        torch.ones(entity_num, entity_num).to(
                                                            entity_relation_gp_factor.device).fill_diagonal_(0), 0), 0), 0), 0)

entity_relation_gp_factor = entity_relation_gp_factor.permute((0,4,5,1,2,3))
print(entity_relation_gp_factor)