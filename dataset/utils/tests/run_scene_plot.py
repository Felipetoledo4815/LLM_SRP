from dataset.nuscenes_dataset import NuscenesDataset
from dataset.utils.plot import ScenePlot
from dataset import config


def main():
    nuscene = NuscenesDataset(config.nuscenes)
    ego_vehicle = nuscene.get_ego_vehicle(0)
    entities = nuscene.get_entities(0)
    image_path = nuscene.get_image(0)

    scene_plot = ScenePlot()
    scene_plot.render_scene(ego_vehicle, entities, image_path, out_path="output.png")


if __name__ == "__main__":
    main()
