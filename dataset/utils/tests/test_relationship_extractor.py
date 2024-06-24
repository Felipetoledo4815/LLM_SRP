import unittest
import numpy as np
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
        for i, expectation in [(1, "very_near"), (10, "near"), (20, "visible"), (1000, None)]:
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
