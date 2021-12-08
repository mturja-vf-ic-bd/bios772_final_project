import unittest
import torch

from src.dataloaders.multi_scale_loaders import schaeferMultiScaleLoader


class testSchaeferMultiScaleLoader(unittest.TestCase):
    def testOutputShape(self):
        loader = schaeferMultiScaleLoader(
            input_template="thick_schaefer_1000_7",
            target_template_list=["thick_schaefer_100_7",  "thick_schaefer_200_7", "thick_schaefer_300_7"],
            batch_size=32,
            split=0.8
        )
        self.assertEqual((1664, 1000), loader.input_train.shape, "Train dataset shape mismatch")
        self.assertEqual((1664, 600), loader.target_train.shape, "Target shape mismatch")
        self.assertEqual((1664, ), loader.y_train.size(), "y shape mismatch")
