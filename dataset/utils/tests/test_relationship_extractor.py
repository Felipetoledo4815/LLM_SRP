import unittest
import numpy as np
from unittest.mock import patch
from scipy.spatial.transform import Rotation as R
from dataset.utils.relationship_extractor import RelationshipExtractor
from dataset.utils.data_clases import Entity, EgoVehicle


class TestRelationshipExtractor(unittest.TestCase):

    def test_get_distance(self):
        entity = Entity(
            entity_type='car',
            xyz=np.array([1, 1, 1]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        ego_vehicle = EgoVehicle(
            xyz=np.array([0, 0, 0]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        relationship_extractor = RelationshipExtractor()
        distance = relationship_extractor.get_distance(entity, ego_vehicle)
        self.assertAlmostEqual(distance, 1.4142, places=4)

    def test_get_discrete_distance_rel(self):
        relationship_extractor = RelationshipExtractor()
        ego_vehicle = EgoVehicle(
            xyz=np.array([0, 0, 0]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        for i, expectation in [(1, "within_25m"), (25, "between_25m_and_40m"), (40, "between_40m_and_60m"), (1000, None)]:
            entity = Entity(
                entity_type='car',
                xyz=np.array([i, i, i]),
                whl=np.array([0, 0, 0]),
                ypr=R.from_euler('z', 0, degrees=True)
            )
            discrete_distance_rel = relationship_extractor.get_discrete_distance_rel(entity, ego_vehicle)
            if discrete_distance_rel:
                self.assertEqual(discrete_distance_rel, ("car", expectation, "ego"))
            else:
                self.assertIsNone(discrete_distance_rel)

    @patch('dataset.utils.data_clases.EgoVehicle.get_projected_center_point')
    @patch('dataset.utils.data_clases.Entity.get_projected_center_point')
    def test_is_in_field_of_view(self, mock_entity_get_projected_center_point, mock_ego_get_projected_center_point):
        mock_ego_get_projected_center_point.return_value = np.array([0, 0])

        relationship_extractor = RelationshipExtractor()
        ego_vehicle = EgoVehicle(
            xyz=np.array([0, 0, 0]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        entity = Entity(
            entity_type='car',
            xyz=np.array([0, 0, 0]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        for xy, expectation in [([1,1], True), ([-10,-10], False)]:
            mock_entity_get_projected_center_point.return_value = np.array(xy)

            is_in_fov = relationship_extractor.is_in_field_of_view(entity, ego_vehicle)
            self.assertEqual(is_in_fov, expectation)

    @patch('dataset.utils.data_clases.EgoVehicle.get_projected_center_point')
    @patch('dataset.utils.data_clases.Entity.get_projected_center_point')
    def test_is_in_field_of_view_66(self, mock_entity_get_projected_center_point, mock_ego_get_projected_center_point):
        mock_ego_get_projected_center_point.return_value = np.array([0, 0])

        relationship_extractor = RelationshipExtractor(field_of_view=66)
        ego_vehicle = EgoVehicle(
            xyz=np.array([0, 0, 0]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        entity = Entity(
            entity_type='car',
            xyz=np.array([0, 0, 0]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        for xy, expectation in [([0,1], True), ([100,0], False)]:
            mock_entity_get_projected_center_point.return_value = np.array(xy)

            is_in_fov = relationship_extractor.is_in_field_of_view(entity, ego_vehicle)
            self.assertEqual(is_in_fov, expectation)
