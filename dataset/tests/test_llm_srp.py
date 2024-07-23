import unittest
from dataset.llm_srp_dataset import LLMSRPDataset, ImageFormat, TripletsFormat, BoundingBoxFormat


class TestLLMSRP(unittest.TestCase):

    def test_bb_output(self):
        llm_srp_dataset = LLMSRPDataset(["nuscenes"], configs={"nuscenes": "nuscenes_mini"}, output_format=(
            ImageFormat.DEFAULT, TripletsFormat.DEFAULT, BoundingBoxFormat.DEFAULT))
        _, _, bb_triplets = llm_srp_dataset.__getitem__(0)
        self.assertIsInstance(bb_triplets, list, "Bounding box triplets should be a list")
        if len(bb_triplets) > 0:
            self.assertIsInstance(bb_triplets[0], tuple, "Bounding box triplet should be a tuple")
            self.assertEqual(len(bb_triplets[0]), 2, "Bounding box triplet should have 2 elements")
            self.assertIsInstance(bb_triplets[0][0], str, "Bounding box should be stated as string")
            self.assertIsInstance(bb_triplets[0][1], list, "Relationships should be a list of triplets")
