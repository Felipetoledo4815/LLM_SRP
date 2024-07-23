import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataset.utils.data_clases import Entity

class Test2DBoundingBox(unittest.TestCase):

    def test_entity_bb(self):
        entity = Entity(
            entity_type='car',
            xyz=np.array([1, 1, 1]),
            whl=np.array([0, 0, 0]),
            ypr=R.from_euler('z', 0, degrees=True)
        )
        bb = entity.get_2d_bounding_box()
        self.assertIsInstance(bb, tuple, "Bounding box should be a tuple")
        self.assertEqual(len(bb), 4, "Bounding box should have 4 elements")
        self.assertEqual(bb, (1.0, 1.0, 1.0, 1.0), "Bounding box corners are wrong")
