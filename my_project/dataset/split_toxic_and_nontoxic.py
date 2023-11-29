import jsonlines as jsl
import json
def main():
    prepare_for_split_path = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-train-90k.jsonl"
    out_nontoxic_path = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/created_nontoxic_from_90k.json"
    dataset_toxic = []
    toxic_num = 0
    p_o_toxic_num = 0
    with jsl.open(prepare_for_split_path, 'r') as reader:
        for line in reader:
            instructions = line['prompt']['text']
            o_toxicity_score = line['continuation']['toxicity']
            toxicity_score = line['prompt']['toxicity']
            output = line['continuation']['text']
            if toxicity_score > 0.5:
                toxic_num += 1
                if o_toxicity_score > 0.5:
                    p_o_toxic_num += 1
                    sample = {'instruction': instructions,
                              'output': output}
                    dataset_toxic.append(sample)

    res = p_o_toxic_num // 1000

    out_toxic_path = f"/media/data/2/yx/model_toxic/data/RealToxicityPrompts/created_toxic_from_90k_to_{res}k.json"

    with open(out_toxic_path, 'w') as writer:
        json.dump(dataset_toxic, writer, indent=4)

    print(f"toxic sample num: {toxic_num}")
    print(f"out-toxic sample num: {p_o_toxic_num}")



if __name__ == "__main__":
    main()