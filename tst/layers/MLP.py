import torch
import unittest

from src.layers.MLP import MLP


class testMLP(unittest.TestCase):
    def testOutputDim(self):
        dimension_list = [10, 32, 32, 2]
        mlp = MLP(dimension_list)
        x = torch.rand(4, 10)
        y = mlp(x)
        self.assertEqual(y.shape, (4, 2), "Output shape doesn't match")

