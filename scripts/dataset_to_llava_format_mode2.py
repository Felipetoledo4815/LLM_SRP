from typing import List, Tuple
import argparse
import json
import os
import re
from tqdm import tqdm
from dataset.llm_srp_dataset import LLMSRPDataset, ImageFormat, TripletsFormat, BoundingBoxFormat


def get_json_sample(idx: int, img: str, question: str, answer: str) -> dict:
    # Return sample in Llava format https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
    return {
        "id": idx,
        "image": img,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question}"
            },
            {
                "from": "gpt",
                "value": answer
            },
        ]
    }


def get_entities_relationship_questions(entities, relationships) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    questions = []
    triplets = []
    for entity in entities:
        if entity == "ego":
            continue
        for relationship in relationships:
            questions.append(f"{entity} {relationship} of ego")
            triplets.append((entity, relationship, "ego"))
    return triplets, questions


def get_rel_text(rel: str):
    # Replace underscores with spaces
    text_rel = rel.replace("_", " ")

    # Replace 'm' following a number with 'meters'
    text_rel = re.sub(r'(\d+)m', r'\1 meters', text_rel)

    return text_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./hf_llm_srp", help="Dataset output folder.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Dataset split to process.")
    args = parser.parse_args()

    # Check if output_dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    llm_srp_dataset = LLMSRPDataset(["nuscenes", "kitti", "waymo_training", "waymo_validation"], output_format=(
        ImageFormat.DEFAULT, TripletsFormat.DEFAULT, BoundingBoxFormat.DEFAULT), configs={
            "nuscenes": "nuscenes",
            "kitti": "kitti_training",
            "waymo_training": "waymo_training",
            "waymo_validation": "waymo_validation",
            })
    idx = 0
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    entities = llm_srp_dataset.get_entity_names()
    relations = llm_srp_dataset.get_relationship_names()
    for split in splits:
        print("Processing", split)
        list_of_json_samples = []
        llm_srp_dataset.set_split(split)
        for i in tqdm(range(len(llm_srp_dataset))):   # pylint: disable=C0200
            img, sg_triplets, _ = llm_srp_dataset[i]
            all_triplets, all_rel_questions = get_entities_relationship_questions(entities, relations)
            for triplet, rel_question in zip(all_triplets, all_rel_questions):
                n_triplets = sg_triplets.count(triplet)
                question = f"Is there a {get_rel_text(rel_question)}"
                if n_triplets > 0:
                    list_of_json_samples.append(get_json_sample(idx, str(img), question, f"Yes. {n_triplets}"))
                else:
                    list_of_json_samples.append(get_json_sample(idx, str(img), question, "No."))
                idx += 1

        # Save to file
        with open(f"{args.output_dir}/llava_dataset_{split}_mode2.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(list_of_json_samples))


if __name__ == "__main__":
    main()
