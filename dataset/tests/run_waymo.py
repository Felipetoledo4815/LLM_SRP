from dataset.waymo_dataset import WaymoDataset
from dataset.config import waymo_validation


def main():
    d = WaymoDataset(waymo_validation)
    # triplets = d.get_sg_triplets(0)
    # img_path = d.get_image(10)
    # e = d.get_ego_vehicle(120)
    # ent = d.get_entities(10)

    d.plot_data_point(0, "test.jpg")

    # entities = d.get_entities(0)
    # image_path = d.get_image(0)
    # d.scene_plot.plot_2d_bounding_boxes([entities[2]], image_path, out_path="test_bb.jpg")


if __name__ == "__main__":
    main()
