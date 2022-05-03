# import unittest

# import numpy as np
# import torch
# from meddlr.metrics.classification import dice_score


# class TestDictScore(unittest.TestCase):
#     # Shape: 4 x 3 x 3
#     PRED_2D = torch.tensor([
#         [[0,1,0,], [0,1,0],[0,0,1]],
#         [[0,0,1,], [0,0,0],[0,1,0]],
#         [[1,0,0,], [1,0,0],[0,0,0]],
#         [[0,0,0,], [1,0,0],[1,0,0]],
#     ])
#     TARGET_2D = torch.tensor([
#         [[0,1,0,], [0,1,1],[0,0,1]],
#         [[0,0,1,], [0,0,1],[0,1,0]],
#         [[1,0,0,], [1,0,0],[0,0,0]],
#         [[0,0,0,], [0,1,0],[1,0,0]],
#     ])

#     def test_dice_one_hot(self):
#         pred = self.PRED_2D
#         target = self.TARGET_2D
#         val = dice_score(pred, target, reduction=None)
#         assert torch.allclose(val, torch.tensor([0.8571428571428571, 0.8, 1.0, 0.5])), val

#         # Batch dimension.
#         batch_size = 2
#         pred = torch.stack([self.PRED_2D] * batch_size, dim=0)
#         target = torch.stack([self.TARGET_2D] * batch_size, dim=0)
#         val = dice_score(pred, target, is_batch=True, reduction=None)
#         assert torch.allclose(
#             val,
#             torch.stack([torch.tensor([0.8571428571428571, 0.8, 1.0, 0.5])]*batch_size, dim=0)
#         ), val


# if __name__ == "__main__":
#     unittest.main()
