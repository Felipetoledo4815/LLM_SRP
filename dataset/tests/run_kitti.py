from dataset.kitti_dataset import KittiDataset
from dataset.config import kitti_3d_dataset


def main():
    d = KittiDataset(kitti_3d_dataset)
    # triplets = d.get_sg_triplets(0)
    # img_path = d.get_image(10)
    # e = d.get_ego_vehicle(120)
    # ent = d.get_entities(10)

    d.plot_data_point(4631, "test.jpg")

    # entities = d.get_entities(0)
    # image_path = d.get_image(0)
    # d.scene_plot.plot_2d_bounding_boxes([entities[2]], image_path, out_path="test_bb.jpg")


if __name__ == "__main__":
    main()
