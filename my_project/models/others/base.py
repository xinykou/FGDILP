
class Base_M:

    def __init__(self, config):
        self.config = config

    # 构建 incontext learning before prompt，负样本和正样本都是局部查找的
    def prefix_build(self, filter_results, max_toxic_index_list, min_nontoxic_index_list, p_num, pad_neg_sample=True):
            prepare_res = []
            for ids, samp in enumerate(filter_results):
                sample_dict = dict(prompt=dict(text=None),
                                   instructions=dict(pos_ins=' ', neg_ins=' '))
                instructs = dict(pos_ins=None, neg_ins=None)
                if max_toxic_index_list[ids]['ids'] is not None and min_nontoxic_index_list[ids]['ids'] is not None:
                    position_0 = min_nontoxic_index_list[ids]['ids'][0]  # 毒性最小的那个作为 正样本
                    li = [samp['continuations_nontoxic'][position_0]['text'] for _ in range(p_num)]
                    instructs['pos_ins'] = li
                    position_1 = max_toxic_index_list[ids]['ids']
                    li_1 = [samp['continuations_toxic'][ids]['text'] for ids in position_1]
                    if len(li_1) < p_num:
                        for _ in range(p_num - len(li_1)):
                            li_1.append(' ')
                    instructs['neg_ins'] = li_1

                    sample_dict['prompt']['text'] = samp['prompt']['text']
                    sample_dict['instructions'] = instructs
                    prepare_res.append(sample_dict)
                elif max_toxic_index_list[ids]['ids'] is None:
                    position_0 = min_nontoxic_index_list[ids]['ids'][0]  # 毒性最小的那个作为 正样本
                    li = [samp['continuations_nontoxic'][position_0]['text'] for _ in range(p_num)]
                    instructs['pos_ins'] = li
                    instructs['neg_ins'] = [' ' for _ in range(p_num)]

                    sample_dict['prompt']['text'] = samp['prompt']['text']
                    sample_dict['instructions'] = instructs
                    prepare_res.append(sample_dict)
                elif min_nontoxic_index_list[ids]['ids'] is None:
                    instructs['pos_ins'] = [' ' for _ in range(p_num)]
                    position_1 = max_toxic_index_list[ids]['ids']
                    li_1 = [samp['continuations_toxic'][ids]['text'] for ids in position_1]
                    instructs['neg_ins'] = li_1

                    sample_dict['prompt']['text'] = samp['prompt']['text']
                    sample_dict['instructions'] = instructs
                    prepare_res.append(sample_dict)

            return prepare_res

    # 负样本和正样本都是全局查找的
    def prefix_build_with_global(self, filter_results, max_toxic_index_list, min_nontoxic_index_list, p_num, pad_neg_sample=True):
        prepare_res = []
        pad_if_or_not = pad_neg_sample
        if not pad_if_or_not:
            print('----不需要填充----')
        for ids, samp in enumerate(filter_results):
            sample_dict = dict(prompt=dict(text=None),
                               instructions=dict(pos_ins=' ', neg_ins=' '))
            instructs = dict(pos_ins=None, neg_ins=None)
            global_prompts = samp['continuations_toxic'] + samp['continuations_nontoxic']
            position_0 = min_nontoxic_index_list[ids]['ids'][0]
            # 毒性最小的那个作为 正样本
            li = [global_prompts[position_0]['text'] for _ in range(p_num)]
            instructs['pos_ins'] = li
            # 毒性较大的那些 负样本
            position_1 = max_toxic_index_list[ids]['ids']
            if 'neg_using_max' in self.config:
                if self.config.neg_using_max:
                    max_id = position_1[0]
                    li_1 = [global_prompts[max_id]['text'] for _ in range(p_num)]
                else:
                    li_1 = [global_prompts[ids]['text'] for ids in position_1]
            else:
                li_1 = [global_prompts[ids]['text'] for ids in position_1]

            if len(li_1) < p_num:
                if pad_neg_sample:
                    for _ in range(p_num - len(li_1)):
                        li_1.append(' ')
                else:  # todo: 如果说不需要填充，那么就是说，如果不够的话，就直接使用正样本作为填充，这样在做减法时就不起作用，相当于没有 ”padding“
                    for _ in range(p_num - len(li_1)):
                        li_1.append(global_prompts[position_0]['text'])
            instructs['neg_ins'] = li_1

            sample_dict['prompt']['text'] = samp['prompt']['text']
            sample_dict['instructions'] = instructs
            prepare_res.append(sample_dict)

        return prepare_res