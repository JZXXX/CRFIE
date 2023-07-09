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

a = [(1,3,3),(4,6,4),(8,9,1)]
b = [(1,2,3),(5,6,2),(7,8,3)]
res = remove_overlap_entities(a,b)
print(res)