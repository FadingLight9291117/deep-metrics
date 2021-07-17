from .general import *


def test_p_r_f():
    preds = [0, 1, 2, 1, 0, 1, 2, 0, 0, 1]
    gts = [0, 1, 1, 2, 0, 1, 0, 2, 0, 0]
    cls_dict = {0: '人', 1: '狗', 2: '猫'}
    print(accuracy(preds, gts))
    pretty_print(precision(preds, gts, cls_dict), False)
    pretty_print(recall(preds, gts, cls_dict), False)
    pretty_print(f1(preds, gts, cls_dict), False)
    pretty_print(p_r_f1(preds, gts, cls_dict))


def test_iou():
    preds = [
        [0, 0, 13, 13],
    ]
    gts = [
        [0, 0, 13, 13],
    ]

    boxes1 = torch.tensor(preds)
    boxes2 = torch.tensor(gts)
    print('iou', box_iou(boxes1, boxes2))


if __name__ == '__main__':
    test_iou()
    test_p_r_f()
