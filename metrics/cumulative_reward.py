import torch
from torchmetrics import Metric

class CumulativeReward(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               reward: torch.Tensor) -> None:
        self.sum = self.sum + reward.sum()
        self.count = self.count + len(reward)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count