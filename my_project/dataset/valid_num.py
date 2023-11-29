import jsonlines as jsl
import json
from tqdm import tqdm
def main():
    prepare_for_split_path = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-train-90k.jsonl"
    out_nontoxic_path = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/created_nontoxic_from_90k.json"
    dataset_toxic = []
    toxic_prompt = []
    nontoxic_prompt = []
    toxic_num = 0
    nontoxic_num = 0
    total_lines = sum(1 for _ in open(prepare_for_split_path))
    with jsl.open(prepare_for_split_path, 'r') as reader:
        for index, line in tqdm(enumerate(reader, start=1), total=total_lines, desc="Processing lines"):
            instructions = line['prompt']['text']
            toxicity_score = line['continuation']['toxicity']
            output = line['continuation']['text']
            if toxicity_score > 0.5:
                if not instructions in toxic_prompt:
                    toxic_prompt.append(instructions)
                    sample = {'instruction': instructions,
                              'output': output}
                    dataset_toxic.append(sample)
                toxic_num += 1

            elif toxicity_score < 0.5:
                if not instructions in nontoxic_prompt:
                    nontoxic_prompt.append(instructions)
                nontoxic_num +=1

    res = len(toxic_prompt) // 1000
    out_toxic_path = f"/media/data/2/yx/model_toxic/data/RealToxicityPrompts/created_toxic_from_90k_to_{res}k.json"
    with open(out_toxic_path, 'w') as writer:
        json.dump(dataset_toxic, writer, indent=4)

    print(f"toxic sample num: {toxic_num}")
    print(f"toxic prompt num: {len(toxic_prompt)}")
    print(f"nontoxic sample num: {nontoxic_num}")
    print(f"nontoxic prompt num: {len(nontoxic_prompt)}")


if __name__ == "__main__":
    main()