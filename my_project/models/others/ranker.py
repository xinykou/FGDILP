from typing import List, Optional, Dict, Any, Tuple

class Ranker_Base():
    def __init__(self, config):
        self.config = config
    # 直接选择索引最大的位置
    def toxic_topk_nontoxic_bottom_select(self, toxicity_list: List[float] = None, nontoxicity_list: List[float] = None,
                                          p_num=None, direct='max',
                                          combine_list: List[Dict[str, Any]] = None):

        # if direct == 'max':
        #     if len(toxicity_list) == 0:
        #         return None
        #     end_index = p_num if len(toxicity_list) >= p_num else len(toxicity_list)
        #     indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)[:end_index]
        # elif direct == 'min':
        #     if len(nontoxicity_list) == 0:
        #         return None
        #     end_index = p_num if len(nontoxicity_list) >= p_num else len(nontoxicity_list)
        #     indexs = sorted(range(len(nontoxicity_list)), key=lambda i: nontoxicity_list[i], reverse=False)[:end_index]

        if direct == 'max':
            indexs = sorted(range(len(combine_list)), key=lambda i: combine_list[i]['toxicity'], reverse=True)[:p_num]
        elif direct == 'min':
            indexs = sorted(range(len(combine_list)), key=lambda i: combine_list[i]['toxicity'], reverse=False)[:1]

        return indexs

    # 这里从25个生成中选择，选择"toxicity"最大的索引位置+ sexually_explicit', 'threat', 'identity_attack', 'profanity', 'insult'最大的位置
    # 如果索引出现重复，则删除重复的索引
    def toxic_subtopk_nontoxic_bottom(self, toxicity_list: List[float] = None, nontoxicity_list: List[float] = None,
                                      p_num=None, direct='max',
                                      combine_list: List[Dict[str, Any]] = None,
                                      ):

        key = 'toxicity' if len(self.config.toxicity_attribute) > 1 else self.config.toxicity_attribute[0]

        if direct == 'min':
            indexs = sorted(range(len(combine_list)), key=lambda i: combine_list[i][key], reverse=False)[:p_num]
            return indexs
        elif direct == 'max':

            max_index = sorted(range(len(combine_list)), key=lambda i: combine_list[i][key], reverse=True)[:1]   # 选择毒性最大的那个
            other_attributes = self.config.toxicity_attribute[1:]
            select_indexs = max_index

            for attribute in other_attributes:
                other_scores = []
                for i in range(len(combine_list)):
                    other_scores.append(combine_list[i][attribute])

                sub_max_index = sorted(range(len(other_scores)), key=lambda i: other_scores[i], reverse=True)[:1]
                select_indexs.append(sub_max_index[0])

            return list(set(select_indexs))

    # 这里从25个生成中选择，选择"toxicity"最大的索引位置+ sexually_explicit', 'threat', 'identity_attack', 'profanity', 'insult'最大的位置
    # 如果索引出现重复，则删除重复的索引
    def toxic_subtopksubsimilar_nontoxic_bottom(self, toxicity_list: List[float] = None, nontoxicity_list: List[float] = None,
                                      p_num=None, direct='max',
                                      combine_list: List[Dict[str, Any]] = None):

        if direct == 'min':
            indexs = sorted(range(len(combine_list)), key=lambda i: combine_list[i]['toxicity'], reverse=False)[
                     :p_num]
            return indexs
        elif direct == 'max':

            max_index = sorted(range(len(combine_list)), key=lambda i: combine_list[i]['toxicity'], reverse=True)[
                        :1]  # 选择毒性最大的那个
            other_attributes = self.config.toxicity_attribute[1:]
            select_indexs = max_index

            for attribute in other_attributes:
                other_scores = []
                for i in range(len(combine_list)):
                    other_scores.append(combine_list[i][attribute])

                sub_max_index = sorted(range(len(other_scores)), key=lambda i: other_scores[i], reverse=True)[:1]
                select_indexs.append(sub_max_index[0])

            return list(set(select_indexs))



