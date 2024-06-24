from typing import List, Tuple
import numpy as np
from dataset.utils.data_clases import Entity, EgoVehicle


class RelationshipExtractor:
    def __init__(self):
        pass

    def get_all_relationships(self, entities: List[Entity], ego_vehicle: EgoVehicle) -> List[Tuple[str, str, str]]:
        relationships = []
        for entity in entities:
            discrete_distance_rel = self.get_discrete_distance_rel(entity, ego_vehicle)
            if discrete_distance_rel:
                relationships.append(discrete_distance_rel)
                relative_position_rel = self.get_relative_position_rel(entity, ego_vehicle)
                relationships.append(relative_position_rel)
        return relationships

    def get_distance(self, entity: Entity, ego_vehicle: EgoVehicle) -> float:
        distance = np.linalg.norm(entity.get_projected_center_point() - ego_vehicle.get_projected_center_point())
        return distance.item()

    def get_discrete_distance_rel(self, entity: Entity, ego_vehicle: EgoVehicle) -> Tuple[str, str, str] | None:
        distance = self.get_distance(entity, ego_vehicle)
        if distance < 10:
            return (entity.entity_type, 'very_near', ego_vehicle.entity_type)
        if distance < 25:
            return (entity.entity_type, 'near', ego_vehicle.entity_type)
        if distance < 50:
            return (entity.entity_type, 'visible', ego_vehicle.entity_type)
        else:
            return None

    def get_relative_position_rel(self, entity: Entity, ego_vehicle: EgoVehicle) -> Tuple[str, str, str]:
        entity_center = entity.get_projected_center_point()
        # Check if entity is in front of ego vehicle
        if entity_center[1] >= abs(entity_center[0]):
            return (entity.entity_type, 'inFrontOf', ego_vehicle.entity_type)
        # Check if entity is to the left of ego vehicle
        if entity_center[0] < 0:
            return (entity.entity_type, 'toLeftOf', ego_vehicle.entity_type)
        # Entity has to be to the left of ego vehicle
        return (entity.entity_type, 'toRightOf', ego_vehicle.entity_type)
