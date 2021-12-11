import torch


def normalize_and_stack_scores(y_pred_list):
    y_pred_stacked = []
    for y_pred in y_pred_list:
        scores = y_pred[:, 0]
        y_pred_stacked.append(scores)
    return torch.stack(y_pred_stacked, dim=-1)


def compute_triangle_area_score(y_pred):
    return torch.stack([torch.mul(y_pred[:, 0], y_pred[:, 1]),
                        torch.mul(y_pred[:, 2], (1 - y_pred)[:, 1]),
                        torch.mul((1 - y_pred)[:, 0], (1 - y_pred)[:, 2])],
                       dim=-1)

