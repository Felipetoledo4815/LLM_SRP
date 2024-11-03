from typing import List, Tuple
import argparse
import json
import os
import re
from tqdm import tqdm
from dataset.llm_srp_dataset import LLMSRPDataset, ImageFormat, TripletsFormat, BoundingBoxFormat


def get_json_sample(idx: int, img: str, bb: str, question: str, answer: str) -> dict:
    # Return sample in Llava format https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
    return {
        "id": idx,
        "image": img,
        "bbox": bb[0],
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nIs the object in the red bounding box {question}?"
            },
            {
                "from": "gpt",
                "value": answer
            },
        ]
    }


def get_rel_text(rel: str):
    # Replace underscores with spaces
    text_rel = rel.replace("_", " ")

    # Replace 'm' following a number with 'meters'
    text_rel = re.sub(r'(\d+)m', r'\1 meters', text_rel)

    return text_rel


def get_relationship_questions(relationships) -> Tuple[List[str], List[str]]:
    questions = []
    relations = []
    for relationship in relationships:
        questions.append(f"{relationship} of ego")
        relations.append(relationship)
    return questions, relations


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

    relations = llm_srp_dataset.get_relationship_names()
    all_rel_questions, all_relationships = get_relationship_questions(relations)
    for split in splits:
        print("Processing", split)
        list_of_json_samples = []
        llm_srp_dataset.set_split(split)
        for i in tqdm(range(len(llm_srp_dataset))):   # pylint: disable=C0200
            img, _, bbs = llm_srp_dataset[i]
            for bb in bbs:
                bb_relations = [triplet[1] for triplet in bb[1]]
                for rel_question, relationship in zip(all_rel_questions, all_relationships):
                    if relationship in bb_relations:
                        list_of_json_samples.append(get_json_sample(idx, str(img), bb, get_rel_text(rel_question), "yes"))
                    else:
                        list_of_json_samples.append(get_json_sample(idx, str(img), bb, get_rel_text(rel_question), "no"))
                    idx += 1

        # Save to file
        with open(f"{args.output_dir}/llava_dataset_{split}_mode4.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(list_of_json_samples))


if __name__ == "__main__":
    main()
