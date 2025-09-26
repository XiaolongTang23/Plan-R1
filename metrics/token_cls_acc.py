import torch
from torchmetrics import Metric

class TokenClsAcc(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        predictions = predictions.argmax(dim=-1)
        correct = predictions == targets
        correct = correct.sum().item()
        self.sum = self.sum + correct
        self.count = self.count + len(targets)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count