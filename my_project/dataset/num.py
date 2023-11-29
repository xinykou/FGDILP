# 统计生成的 部分 毒性的类型
import jsonlines as jsl
import tqdm as tqdm




PERSPECTIVE_API_ATTRIBUTES = (
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
file_toxic_types_num = dict()
# 读取数据
with jsl.open('/media/data/2/yx/model_toxic/my_project/results/vector_p_muti_gen/toxic_filters_gen_tags.jsonl') as f:
    for index, line in tqdm(enumerate(f), dsc="toxic_types_num"):
        toxic_all = line["continuations_toxic"]
        toxic_type_num = dict()
        for tox in toxic_all:
            tox_score = tox["sub_toxicity"]
            max_key = max(tox_score.items(), key=lambda x: x[1])[0]
            toxic_type_num[max_key] = toxic_type_num.get(max_key, 0) + 1
        num = len(toxic_type_num)
        file_toxic_types_num[num] = file_toxic_types_num.get(num, 0) + 1

print(file_toxic_types_num)