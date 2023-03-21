import torchmetrics
import torch


class PerPositionAccuracy(torchmetrics.Metric):
    def __init__(self, n_positions: int = 64, n_codes: int = 256):
        super().__init__()
        self.n_positions = n_positions
        self.n_codes = n_codes
        self.add_state("correct", default=torch.full(
            (n_positions, ), 0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.full(
            (n_positions, ), 0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # preds, target shape of (bs, n_positions)
        for i in range(self.n_positions):
            self.correct[i] += torch.sum(preds[:, i] == target[:, i])
            self.total[i] += target[:, i].numel()

    def compute(self):
        return self.correct.float() / self.total
