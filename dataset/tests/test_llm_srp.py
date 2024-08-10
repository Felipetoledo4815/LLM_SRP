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

    def test_splits(self):
        llm_srp_dataset = LLMSRPDataset(["nuscenes"], configs={"nuscenes": "nuscenes_mini"})
        self.assertEqual(len(llm_srp_dataset), llm_srp_dataset.length,
                         msg="All data should be used if no split is specified")
        llm_srp_dataset.set_split("train")
        self.assertAlmostEqual(len(llm_srp_dataset), int(llm_srp_dataset.length*0.75), delta=5,
                         msg="Train split should have 75 % of the data")

        llm_srp_dataset = LLMSRPDataset(["nuscenes"], split="train", configs={"nuscenes": "nuscenes_mini"})
        self.assertAlmostEqual(len(llm_srp_dataset), int(llm_srp_dataset.length*0.75), delta=5,
                         msg="Train split should have 75 % of the data")

        llm_srp_dataset = LLMSRPDataset(["nuscenes"], split="val", configs={"nuscenes": "nuscenes_mini"})
        self.assertAlmostEqual(len(llm_srp_dataset), int(llm_srp_dataset.length*0.05), delta=5,
                         msg="Validation split should have 5 % of the data")

        llm_srp_dataset = LLMSRPDataset(["nuscenes"], split="test", configs={"nuscenes": "nuscenes_mini"})
        self.assertAlmostEqual(len(llm_srp_dataset), int(llm_srp_dataset.length*0.20), delta=5,
                         msg="Test split should have 20 % of the data")

        with self.assertWarns(Warning):
            llm_srp_dataset = LLMSRPDataset(["nuscenes"], split="xyz", configs={"nuscenes": "nuscenes_mini"})
