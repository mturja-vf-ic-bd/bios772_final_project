import unittest
import torch

from src.dataloaders.ad_data_loaders import adDataLoader, cut_templates_and_join


class testAdDataLoader(unittest.TestCase):
    def testDataShape(self):
        loader = adDataLoader(batch_size=32, split=0.8)
        self.assertEqual(1664, len(loader.train_set), "Train dataset shape mismatch")
        self.assertEqual(416, len(loader.val_set), "Validation dataset shape mismatch")

    def testCutTemplatesAndJoin(self):
        x = torch.rand(3, 1000)
        ind_list = [(50, 100), (130, 200), (200, 202)]
        x_cut = cut_templates_and_join(x, ind_list)
        self.assertEqual(122, x_cut.shape[1], "ShapeMismatch")
