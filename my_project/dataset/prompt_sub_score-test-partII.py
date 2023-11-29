""""
perspective api 评分细粒度的 毒性评分
"""
import sys
import jsonlines as jsl
import json
import requests
import time
from tqdm import tqdm
import os
os.environ["http_proxy"] = "http://127.0.0.1:27999"
os.environ["https_proxy"] = "http://127.0.0.1:27999"

from googleapiclient import discovery

# Use perspective API
qps = 1
rate = 1/qps

# PUT YOUR API KEY HERE
api_key = "AIzaSyCH7xhMBsW_iE6a-6UcaltwburL77b6p_s"

import socket

import requests

url = "https://www.google.com"

try:
    response = requests.get(url)
    if response.status_code == 200:
        print("成功连接到 Google 网页！")
    else:
        print("连接到 Google 网页失败。状态码：", response.status_code)
except requests.exceptions.RequestException as e:
    print("连接到 Google 网页时发生异常：", e)

PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)


def _make_request(text: str, service):
    analyze_request = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes':{attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES}
    }
    return service.comments().analyze(body=analyze_request)


def main():
    prompts_file = "/media/data/2/yx/model_toxic/my_project/results/vector_p_muti_select_gen/nontoxic_filters_gen.jsonl"
    tags_file = "/media/data/2/yx/model_toxic/my_project/results/vector_p_muti_select_gen/nontoxic_filters_gen_tags-partII.jsonl"
    created_li = []

    service = discovery.build('commentanalyzer',
                              'v1alpha1',
                              developerKey=api_key,
                              # discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                              static_discovery=False)
    start_line = 3300
    if os.path.exists(tags_file):
        with jsl.open(tags_file, 'r') as file:
            for index, line in enumerate(file):
                created_li.append(line)
        start_line += len(created_li)
        print("已经处理了", len(created_li), "条数据")
    else:
        print("没有处理过的数据")

    try:
        with jsl.open(prompts_file, 'r') as file:
            for index, line in enumerate(file):
                if index < start_line:
                    continue
                dict_line = dict()

                dict_line['prompt'] = line['prompt']
                tox_li = line["continuations_toxic"]
                nontox_li = line["continuations_nontoxic"]

                dict_line["continuations_toxic"] = []
                dict_line["continuations_nontoxic"] = []
                map_id_to_name = {'0': "continuations_toxic", '1': 'continuations_nontoxic'}


                for flag, named_li in tqdm(enumerate([tox_li, nontox_li]), total=2, desc="toxicity_score"):
                    if len(named_li) > 0:
                        for tox in named_li:
                            dict_line_sub = dict()
                            if tox["text"] == "":
                                continue
                            dict_line_sub["text"] = tox["text"]
                            dict_line_sub["toxicity"] = tox["toxicity"]
                            texts = dict_line_sub["text"]

                            ids = 0
                            responses = {str(ids): None}
                            batch_request = service.new_batch_http_request()

                            def response_callback(request_id, response, exception):
                                nonlocal responses
                                responses[request_id] = (response, exception)

                            # 第 1 次请求
                            while 1:
                                batch_request.add(_make_request(texts, service), callback=response_callback, request_id=str(ids))
                                batch_request.execute()
                                res = list(responses.values())
                                try:
                                    attribute_scores = res[0][0]["attributeScores"]
                                    break
                                except Exception as e:
                                    print("第1次：请求超时, 再来一次", e)
                                    ids += 1
                                    time.sleep(ids)
                                    if ids > 100:
                                        print("第1次请求超时, 跳过")
                                        ids = 0
                                        break
                                    continue

                            # 第 2 次请求
                            ids = 0
                            batch_request = service.new_batch_http_request()
                            while 1:
                                batch_request.add(_make_request(texts, service), callback=response_callback, request_id=str(ids))
                                batch_request.execute()
                                res = list(responses.values())
                                try:
                                    attribute_scores = res[0][0]["attributeScores"]
                                    break
                                except Exception as e:
                                    print("第2次：请求超时, 再来一次", e)
                                    ids += 1
                                    time.sleep(ids)
                                    if ids > 100:
                                        print("第2次请求超时, 跳过")
                                        jsl.Writer(open(tags_file, 'w', encoding='utf-8')).write_all(created_li)
                                        sys.exit()
                                    continue

                            summary_scores = {}
                            for attribute, scores in attribute_scores.items():
                                attribute = attribute.lower()  # 代表毒性类型
                                summary_scores[attribute] = scores['summaryScore']['value']
                            dict_line_sub["sub_toxicity"] = summary_scores

                            dict_line[map_id_to_name[str(flag)]].append(dict_line_sub)

                            print("texts:", texts, "sub_toxicity:", dict_line_sub)


                created_li.append(dict_line)
                print("line:", index)

        jsl.Writer(open(tags_file, 'w', encoding='utf-8')).write_all(created_li)


    except Exception as e:
        print("出现异常, 保存数据")
        jsl.Writer(open(tags_file, 'w', encoding='utf-8')).write_all(created_li)
        sys.exit()

if __name__ == '__main__':
    main()
