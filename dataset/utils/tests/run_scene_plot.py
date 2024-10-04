from dataset.nuscenes_dataset import NuscenesDataset
from dataset.utils.plot import ScenePlot
from dataset import config


def main():
    nuscene = NuscenesDataset(config.nuscenes_mini)
    ego_vehicle = nuscene.get_ego_vehicle(0)
    entities = nuscene.get_entities(0)
    image_path = nuscene.get_image(0)

    scene_plot = ScenePlot(field_of_view=nuscene.field_of_view)
    scene_plot.render_scene(ego_vehicle, entities, image_path)
    scene_plot.plot_2d_bounding_boxes([entities[2]], image_path)

    bb_triplets = nuscene.get_bb_triplets(380)
    scene_plot.plot_2d_bounding_boxes_from_corners([bb_triplets[0][0]], image_path)
    scene_plot.plot_2d_bounding_boxes_from_corners([bb_triplets[0][0]],
                                                   image_path,
                                                   entity_types=[bb_triplets[0][1][0][0]])


if __name__ == "__main__":
    main()
