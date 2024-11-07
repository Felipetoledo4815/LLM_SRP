from dataset.waymo_dataset import WaymoDataset
from dataset.nuscenes_dataset import NuscenesDataset
from dataset.config import waymo_validation
from dataset.config import nuscenes_mini


def main():
    d = WaymoDataset(waymo_validation)
    # d = NuscenesDataset(nuscenes_mini)
    # ent = d.get_entities(250)
    d.plot_data_point(250, "test.jpg")
    # d.relationship_extractor.is_occluded(ent[0], ent)


if __name__ == "__main__":
    main()
