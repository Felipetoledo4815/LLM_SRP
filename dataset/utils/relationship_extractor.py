from typing import List, Tuple
import numpy as np
import math
from dataset.utils.data_clases import Entity, EgoVehicle, RelationshipType


class RelationshipExtractor:
    def __init__(self, field_of_view: float = 180) -> None:
        self.field_of_view = field_of_view

    def get_all_relationships(self, entities: List[Entity], ego_vehicle: EgoVehicle) -> List[Tuple[str, str, str]]:
        relationships = []
        for entity in entities:
            if not self.is_in_field_of_view(entity, ego_vehicle):
                continue
            discrete_distance_rel = self.get_discrete_distance_rel(entity, ego_vehicle)
            if discrete_distance_rel:
                relationships.append(discrete_distance_rel)
                relative_position_rel = self.get_relative_position_rel(entity, ego_vehicle)
                relationships.append(relative_position_rel)
        return relationships

    def get_all_bb_relationships(self, entities: List[Entity],
                                 ego_vehicle: EgoVehicle) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        all_relationships = []
        for entity in entities:
            if not self.is_in_field_of_view(entity, ego_vehicle):
                continue
            relationships = []
            discrete_distance_rel = self.get_discrete_distance_rel(entity, ego_vehicle)
            if discrete_distance_rel:
                relationships.append(discrete_distance_rel)
                relative_position_rel = self.get_relative_position_rel(entity, ego_vehicle)
                relationships.append(relative_position_rel)
            all_relationships.append((str(entity.get_2d_bounding_box()), relationships))
        return all_relationships

    def is_in_field_of_view(self, entity: Entity, ego_vehicle: EgoVehicle) -> bool:
        # Extract the position of the entity and the ego vehicle
        entity_position = entity.get_projected_center_point()
        ego_position = ego_vehicle.get_projected_center_point()

        # Calculate the angle to the entity from the ego vehicle
        dx = entity_position[0] - ego_position[0]
        dy = entity_position[1] - ego_position[1]
        angle_to_entity = math.atan2(dy, dx)

        # Adjust the angle for the ego vehicle's 90 degree rotation
        adjusted_angle_to_entity = angle_to_entity - math.pi / 2

        # Normalize the adjusted angle to the range [-pi, pi]
        adjusted_angle_to_entity = (adjusted_angle_to_entity + math.pi) % (2 * math.pi) - math.pi

        # Convert the field of view from degrees to radians for comparison
        half_fov_rad = math.radians(self.field_of_view / 2)

        # Check if the absolute value of the angle is within the half field of view
        return abs(adjusted_angle_to_entity) <= half_fov_rad

    def get_distance(self, entity: Entity, ego_vehicle: EgoVehicle) -> float:
        distance = np.linalg.norm(entity.get_projected_center_point() - ego_vehicle.get_projected_center_point())
        return distance.item()

    def get_discrete_distance_rel(self, entity: Entity, ego_vehicle: EgoVehicle) -> Tuple[str, str, str] | None:
        """
        Returns a tuple with the relationship between the entity and the ego vehicle based on the distance between them.
        Distances are divided into three categories based on stopping distances by driving speed.
        https://www.dmv.virginia.gov/sites/default/files/forms/dmv39d.pdf
        """
        distance = self.get_distance(entity, ego_vehicle)
        if distance <= 25:  # Speed 25 mph
            return (entity.entity_type, RelationshipType.WITHIN_25M.type_name, ego_vehicle.entity_type)
        if distance <= 40:  # Speed 35 mph
            return (entity.entity_type, RelationshipType.BETWEEN_25M_AND_40M.type_name, ego_vehicle.entity_type)
        if distance <= 60:  # Speed 45 mph
            return (entity.entity_type, RelationshipType.BETWEEN_40M_AND_60M.type_name, ego_vehicle.entity_type)
        return None

    def get_relative_position_rel(self, entity: Entity, ego_vehicle: EgoVehicle) -> Tuple[str, str, str]:
        # Calculate slope starting from 90 since camera is centered
        angle_in_rad = math.radians(90 - (self.field_of_view / 3) / 2)
        slope = math.tan(angle_in_rad)
        entity_center = entity.get_projected_center_point()
        # Check if entity is in front of ego vehicle
        if entity_center[1] >= abs(entity_center[0]) * slope:
            return (entity.entity_type, RelationshipType.IN_FRONT_OF.type_name, ego_vehicle.entity_type)
        # Check if entity is to the left of ego vehicle
        if entity_center[0] < 0:
            return (entity.entity_type, RelationshipType.TO_LEFT_OF.type_name, ego_vehicle.entity_type)
        # Entity has to be to the left of ego vehicle
        return (entity.entity_type, RelationshipType.TO_RIGHT_OF.type_name, ego_vehicle.entity_type)
