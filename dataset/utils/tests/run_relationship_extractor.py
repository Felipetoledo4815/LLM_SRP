from dataset.nuscenes_dataset import NuscenesDataset
from dataset.utils.relationship_extractor import RelationshipExtractor
from dataset import config


def main():
    nuscene = NuscenesDataset(config.nuscenes)
    ego_vehicle = nuscene.get_ego_vehicle(0)
    entities = nuscene.get_entities(0)

    rel_extractor = RelationshipExtractor(nuscene.field_of_view)
    relationships = rel_extractor.get_all_relationships(entities, ego_vehicle)
    print(relationships)


if __name__ == "__main__":
    main()
