from torch.utils.data import DataLoader
from dataset.llm_srp_dataset import LLMSRPDataset


def main():
    llm_srp_dataset = LLMSRPDataset(["nuscenes"], output_format="load_image",)
    dataloader = DataLoader(llm_srp_dataset, batch_size=10, shuffle=True, collate_fn=llm_srp_dataset.collate_fn)
    for i, (img, sg_triplets) in enumerate(dataloader):
        print(f"Batch {i}: {img.shape}, {len(sg_triplets)}")
        break


if __name__ == "__main__":
    main()
