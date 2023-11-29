PERSPECTIVE_API_ATTRIBUTES = [
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
]
PERSPECTIVE_API_ATTRIBUTES_LOWER = [a.lower() for a in PERSPECTIVE_API_ATTRIBUTES]

import random

def select_max_toxicity_outputs(outputs: list = None, toxicity_list: list = None, p_num: int = None, direct: str = None):
    disagree_enable = False
    # if direct == 'max_value':
    #     indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)
    #     try_max_indexs = indexs[:p_num]
    #     re_max_indexs = []
    #     saved_toxic_types = []
    #     for index in try_max_indexs:
    #         sample = outputs[index]
    #         sub_score = sample['sub_toxicity']
    #         filter_sub_score = dict()
    #         for key in sub_score.keys():
    #             if key in PERSPECTIVE_API_ATTRIBUTES_LOWER:
    #                 filter_sub_score[key] = sub_score[key]
    #             else:
    #                 continue
    #
    #         max_toxic_type = max(filter_sub_score, key=filter_sub_score.get)
    #         if max_toxic_type not in saved_toxic_types:
    #             saved_toxic_types.append(max_toxic_type)
    #             re_max_indexs.append(index)
    #         else:
    #             continue
    #
    #     if len(saved_toxic_types) < p_num:
    #         for index in indexs[p_num:]:
    #             sample = outputs[index]
    #             sub_score = sample['sub_toxicity']
    #             filter_sub_score = dict()
    #             for key in sub_score.keys():
    #                 if key in PERSPECTIVE_API_ATTRIBUTES_LOWER:
    #                     filter_sub_score[key] = sub_score[key]
    #                 else:
    #                     continue
    #
    #             max_toxic_type = max(filter_sub_score, key=filter_sub_score.get)
    #             if max_toxic_type not in saved_toxic_types:
    #                 saved_toxic_types.append(max_toxic_type)
    #                 re_max_indexs.append(index)
    #                 # disagree_enable = True
    #                 if len(saved_toxic_types) == p_num:
    #                     disagree_enable = True
    #                     break
    #             else:
    #                 continue
    #
    #     return re_max_indexs, disagree_enable
    # if direct == 'max_value':
    #     indexes = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)  # 根据毒性从大到校排列 字典编号
    #     best_max_index = indexes[0]
    #     keys_to_keep = PERSPECTIVE_API_ATTRIBUTES_LOWER
    #     org_dict = outputs[best_max_index]['sub_toxicity']
    #     # 去除除指定键以外的其他键值对
    #     max_toxicity_sub_score = {key: value for key, value in org_dict.items() if key in keys_to_keep}
    #     # 找到最大值对应的键
    #     max_key = max(max_toxicity_sub_score.items(), key=lambda x: x[1])[0]
    #     PERSPECTIVE_API_ATTRIBUTES_LOWER_copy = PERSPECTIVE_API_ATTRIBUTES_LOWER[:]
    #     PERSPECTIVE_API_ATTRIBUTES_LOWER_copy.remove(max_key)
    #     try_indexes = indexes[1:] if len(indexes) > 1 else None
    #
    #     re_max_indexes = [best_max_index]
    #
    #     if try_indexes is not None:
    #         filter_outputs = []
    #         for i in try_indexes:
    #             sub_score = outputs[i]['sub_toxicity']
    #             filter_sub_score = dict()
    #             for key in sub_score.keys():
    #                 if key in PERSPECTIVE_API_ATTRIBUTES_LOWER_copy:
    #                     filter_sub_score[key] = sub_score[key]
    #                 else:
    #                     continue
    #             filter_outputs.append(filter_sub_score)
    #         # filter_outputs.remove(filter_outputs[max_index])
    #         # 选择每个键对应的最大值所在的字典编号
    #
    #         for key in filter_outputs[0].keys():
    #             max_value = float('-inf')
    #             max_index = None
    #             for i in range(len(filter_outputs)):
    #                 if filter_outputs[i][key] > max_value and try_indexes[i] not in re_max_indexes:
    #                     max_value = filter_outputs[i][key]
    #                     max_index = try_indexes[i]
    #             if max_index is not None:
    #                 re_max_indexes.append(max_index)
    #
    #     return re_max_indexes, disagree_enable
    #
    #
    #
    # elif direct == 'random':
    #     indexes = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)
    #     # 设置随机种子
    #     random.seed(2023)
    #     # 随机选择元素并组合
    #     selected_elements = random.sample(indexes, k=p_num) if len(indexes) > p_num else indexes
    #     return selected_elements, disagree_enable


