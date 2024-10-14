from dataset.llm_srp_dataset import LLMSRPDataset
import random
def main():
    llm_srp_dataset = LLMSRPDataset(['nuscenes'], configs={"nuscenes":"nuscenes_mini"})
    # plot entities in a 2d plane
    i_id = random.randint(0, len(llm_srp_dataset) - 1)
    img_path, triplets, bb_triplets = llm_srp_dataset.__getitem__(i_id)
    llm_srp_dataset.plot_data_point(i_id)
    print(triplets)
    #plot bounding boxes
    img_path, triplets, bb_triplets = llm_srp_dataset.__getitem__(i_id)
    print(bb_triplets)
    triplet_sample = bb_triplets[0]
    print(triplet_sample)
    llm_srp_dataset.plot_bounding_box(i_id, bbs=[triplet_sample[0]], entity_types=[triplet_sample[1][0][0]])

if __name__ == "__main__":
    main()