from dataset.kitti_dataset import KittiDataset
from dataset.nuscenes_dataset import NuscenesDataset
from dataset.utils.plot import ScenePlot
from dataset import config


def main():
    kitti = KittiDataset(config.kitti_3d_dataset)
    idx = 4631
    ego_vehicle = kitti.get_ego_vehicle(idx)
    entities = kitti.get_entities(idx)
    image_path = kitti.get_image(idx)
    print('\n'.join(map(str, kitti.get_sg_triplets(idx))))
    bb_triplets = kitti.get_bb_triplets(idx)
    print('BB Triplets')
    print(bb_triplets)

    scene_plot = ScenePlot(field_of_view=kitti.field_of_view)
    scene_plot.render_scene(ego_vehicle, entities, image_path)

    scene_plot.plot_2d_bounding_boxes([entities[1]], image_path)
    # print("we are here")
    # scene_plot.plot_2d_bounding_boxes_from_corners([bb_triplets[0][0]], image_path)
    # scene_plot.plot_2d_bounding_boxes_from_corners([bb_triplets[0][0]],
    #                                                image_path,
    #                                                 entity_types=[bb_triplets[0][1][0][0]])


if __name__ == "__main__":
    main()
