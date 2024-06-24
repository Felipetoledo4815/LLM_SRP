from dataset.nuscenes_dataset import NuscenesDataset
from dataset.utils.plot import ScenePlot
from dataset.utils.relationship_extractor import RelationshipExtractor

def main():
    nuscene = NuscenesDataset()
    ego_vehicle = nuscene.get_ego_vehicle(0)
    entities = nuscene.get_entities(0)
    image_path = nuscene.get_image(0)

    # scene_plot = ScenePlot()
    # scene_plot.render_scene(ego_vehicle, entities, image_path)

    rel_extractor = RelationshipExtractor()
    relationships = rel_extractor.get_all_relationships(entities, ego_vehicle)
    print(relationships)


if __name__ == "__main__":
    main()
