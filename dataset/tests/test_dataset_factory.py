import unittest
from dataset.dataset_factory import DatasetFactory


class TestDatasetFactory(unittest.TestCase):

    def test_dataset_factory(self):
        dataset_factory = DatasetFactory(["nuscenes"])
        datasets = dataset_factory.get_datasets()
        self.assertEqual(len(datasets), 1)
        self.assertTrue("nuscenes" in datasets)
        self.assertTrue(datasets["nuscenes"] is not None)

    def test_dataset_factory_error(self):
        with self.assertRaises(AssertionError):
            DatasetFactory(["non_existent_dataset"])
