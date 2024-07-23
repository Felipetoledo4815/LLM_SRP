from dataset.nuscenes_dataset import NuscenesDataset
from dataset import config


def main():
    nuscene = NuscenesDataset(config.nuscenes_mini)
    entities = nuscene.get_entities(0)
    bb = entities[0].get_2d_bounding_box()
    print(bb)

    bb_triplets = nuscene.get_bb_triplets(0)
    print(bb_triplets)

if __name__ == "__main__":
    main()
