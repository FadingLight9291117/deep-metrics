from sklearn.metrics import accuracy_score, \
    precision_score, \
    recall_score, \
    f1_score, \
    precision_recall_fscore_support, \
    mean_squared_error, \
    mean_absolute_error
import numpy as np
import torch

from .utils import pretty_print, number_formatter


def accuracy(preds, gts):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    return number_formatter(accuracy_score(gts, preds))


def precision(preds, gts, cls):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    assert cls is not None, 'cls 参数不能为空'
    scores = precision_score(gts, preds, average=None, zero_division=1)
    res = None
    if type(cls) is list:
        res = {c: number_formatter(scores[i]) for i, c in enumerate(cls)}
    elif type(cls) is dict:
        res = {c: number_formatter(scores[i]) for i, c in cls.items()}
    return res


def recall(preds, gts, cls):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    assert cls is not None, 'cls 参数不能为空'
    scores = recall_score(gts, preds, average=None, zero_division=1)
    res = None
    if type(cls) is list:
        res = {c: number_formatter(scores[i]) for i, c in enumerate(cls)}
    elif type(cls) is dict:
        res = {c: number_formatter(scores[i]) for i, c in cls.items()}
    return res


def f1(preds, gts, cls):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    assert cls is not None, 'cls 参数不能为空'
    scores = f1_score(gts, preds, average=None, zero_division=1)
    res = None
    if type(cls) is list:
        res = {c: number_formatter(scores[i]) for i, c in enumerate(cls)}
    elif type(cls) is dict:
        res = {c: number_formatter(scores[i]) for i, c in cls.items()}
    return res


def p_r_f1(preds, gts, cls):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    assert cls is not None, 'cls 参数不能为空'
    scores = precision_recall_fscore_support(gts, preds, average=None, zero_division=1)
    scores = np.array(scores).T[:, :3]
    res = {}
    if type(cls) is list:
        for i, c in enumerate(cls):
            si = {
                'precison': number_formatter(scores[i][0]),
                'recall': number_formatter(scores[i][1]),
                'f1': number_formatter(scores[i][2]),
            }
            res[c] = si
    elif type(res) is dict:
        for i, c in cls.items():
            si = {
                'precison': number_formatter(scores[i][0]),
                'recall': number_formatter(scores[i][1]),
                'f1': number_formatter(scores[i][2]),
            }
            res[c] = si
    return res


def mae(preds, gts):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    return mean_absolute_error(gts, preds)


def mse(preds, gts):
    assert len(preds) == len(gts), 'preds和gts长度必须相同'
    return mean_squared_error(gts, preds)


def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    if type(boxes1) is not torch.Tensor:
        boxes1 = torch.tensor(boxes1)
    if type(boxes2) is not torch.Tensor:
        boxes2 = torch.tensor(boxes2)

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(
        0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    if type(wh1) is not torch.Tensor:
        wh1 = torch.tensor(wh1)
    if type(wh2) is not torch.Tensor:
        wh2 = torch.tensor(wh2)
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)
