import torch
import torch.nn as nn


class WeightedMaeLossPixel(nn.Module):
    def __init__(self, weights, threshold=0, boundary_bbox=None):
        super().__init__()
        self.balancing_weights = weights
        self.threshold = threshold
        self._bbox = boundary_bbox
        if self._bbox is not None:
            print(f"[{self.__class__.__name__}] Bbox:{self._bbox}")

    def forward(self, predicted, target, mask):
        if self._bbox is not None:
            predicted, target, mask = ignore_boundary(
                self._bbox, predicted, target, mask
            )
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)

        weights = compute_weights(target, mask, self.balancing_weights)
        return weights * (torch.abs((predicted - target)))


class WeightedMaeLoss(WeightedMaeLossPixel):
    def forward(self, predicted, target):
        # build mask
        mask = torch.zeros_like(target)
        mask[target > self.threshold] = 1

        return torch.mean(super().forward(predicted, target, mask))


def ignore_boundary(bbox, predicted, target, mask):
    xl, xr, yl, yr = bbox
    Nx, Ny = predicted.shape[-2:]
    predicted = predicted[..., xl : Nx - xr, yl : Ny - yr]
    target = target[..., xl : Nx - xr, yl : Ny - yr]
    mask = mask[..., xl : Nx - xr, yl : Ny - yr]
    return predicted, target, mask


def compute_weights(target, mask, BALANCING_WEIGHTS=[1, 2, 5, 10, 30]):
    weights = torch.ones_like(mask) * BALANCING_WEIGHTS[0]
    for i, threshold in enumerate(BALANCING_WEIGHTS):
        if i == 0:
            continue
        weights = (
            weights
            + (BALANCING_WEIGHTS[i] - BALANCING_WEIGHTS[i - 1])
            * (target >= threshold).float()
        )
    weights = weights * mask.float()
    return weights
