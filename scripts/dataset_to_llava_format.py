import json
from tqdm import tqdm
from dataset.llm_srp_dataset import LLMSRPDataset, ImageFormat, TripletsFormat, BoundingBoxFormat


def get_json_sample(idx: int, img: str, sg_triplets: str) -> dict:
    # Return sample in Llava format https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
    return {
        "id": idx,
        "image": img,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nGive me the triplets."
            },
            {
                "from": "gpt",
                "value": sg_triplets
            },
        ]
    }

def main():
    llm_srp_dataset = LLMSRPDataset(["nuscenes"], output_format=(
        ImageFormat.DEFAULT, TripletsFormat.DEFAULT, BoundingBoxFormat.DEFAULT), configs={"nuscenes": "nuscenes"})
    idx = 0
    for split in ["train", "val", "test"]:
        print("Processing", split)
        list_of_json_samples = []
        llm_srp_dataset.set_split(split)
        for i in tqdm(range(len(llm_srp_dataset))):   # pylint: disable=C0200
            img, sg_triplets, _ = llm_srp_dataset[i]
            list_of_json_samples.append(get_json_sample(idx, str(img), str(sg_triplets)))
            idx += 1

        # Save to file
        with open(f"llava_dataset_{split}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(list_of_json_samples))


if __name__ == "__main__":
    main()
