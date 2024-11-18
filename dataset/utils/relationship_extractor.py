from typing import List, Tuple
import math
import numpy as np
from dataset.utils.data_clases import Entity, EgoVehicle, RelationshipType


class RelationshipExtractor:
    def __init__(self, field_of_view: float = 180) -> None:
        self.field_of_view = field_of_view

    def is_point_right_of_line_segments(self, line_points, point):
        px, py = point
        x, y, z = line_points
        for i in range(len(x)):
            x1, y1 = x[i], y[i]

            if y1 + 1 < py:
                continue

            if x1 > px:
                # print(f"{x[0]} right {x1}")
                return False
            if y1 > py:
                break

        return True

    def is_point_left_of_line_segments(self, line_points, point):
        px, py = point
        x, y, z = line_points
        for i in range(len(x)):
            x1, y1 = x[i], y[i]

            if y1 + 1 < py:
                continue

            if x1 < px:
                # print(f"{x[0]} left {x1}")
                return False
            if y1 > py:
                break

        return True

    def get_lanes_on_opposing_lane(self, entities, left_most_line):
        for entity in entities:
            center_point = entity.get_projected_center_point()
            if self.is_point_left_of_line_segments(left_most_line, center_point):
                print(f"{entity.entity_type} is in Opposing lane")

    def get_all_lane_entity_relationship(self, entities: List[Entity], lanes):
        entities_not_in_lane = []
        for entity in entities:
            center_point = entity.get_projected_center_point()
            for lane in lanes:
                if (self.is_point_right_of_line_segments(lane['left_line_of_lane'].xyz, center_point) and
                        self.is_point_left_of_line_segments(lane['right_line_of_lane'].xyz, center_point)):
                    print(f"{entity.entity_type} is in lane {lane['lane_position']}")
                    break
            entities_not_in_lane.append(entity)

        if len(entities_not_in_lane) > 0:
            self.get_lanes_on_opposing_lane(entities_not_in_lane, lanes[0]['left_line_of_lane'].xyz)



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

    def is_occluded(self, entity: Entity, all_entities: List[Entity]) -> bool:
        # Ray from (0,0,0) to the top center point of the entity
        entity_top_center_point = entity.top_center_point()
        ray = {"origin": np.array([0, 0, 0]), "direction": entity_top_center_point - np.array([0, 0, 0])}
        for other_entity in all_entities:
            if entity == other_entity:
                continue
            else:
                # Check if the ray intersects with the other_entity planes
                other_entity_planes = other_entity.lateral_planes()
                # for ray in rays:
                for plane in other_entity_planes:
                    denominator = np.dot(plane["normal"], ray["direction"])
                    if np.isclose(denominator, 0):  # Ray is parallel to the plane
                        continue
                    numerator = plane["d"] - np.dot(plane["normal"], ray["origin"])
                    t = numerator / denominator
                    if t < 0:   # Intersection is in the opposite direction of the ray
                        continue
                    intersection_point = ray["origin"] + t * ray["direction"]
                    if self.is_point_in_polygon(intersection_point, plane['points']):
                        return True

        return False

    def is_point_in_polygon(self, point: np.ndarray, quad: List[np.ndarray]) -> bool:
        # Project 3D points to 2D by choosing the two most varying coordinates
        # Find the range of coordinates to determine which axis to ignore
        quad = np.stack(quad, axis=0) # type: ignore
        ranges = np.ptp(quad, axis=0)  # Range (max - min) for x, y, z

        # Ignore the axis with the smallest range
        ignore_axis = np.argmin(ranges)

        # Create 2D projections of the quadrilateral and the point
        quad_2d = np.delete(quad, ignore_axis, axis=1)
        point_2d = np.delete(point, ignore_axis)

        x, y = point_2d
        n = len(quad_2d)

        # Check if point_2d is inside quad_2d
        inside = False
        p1x, p1y = quad_2d[0]
        for i in range(n + 1):
            p2x, p2y = quad_2d[i % n]
            xinters = float('inf')
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

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
