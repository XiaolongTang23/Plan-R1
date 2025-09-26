import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_min

from utils import transform_point_to_local_coordinate
from utils import drop_edge_between_samples

class ProgressReward(nn.Module):
    def __init__(self, num_historical_intervals=4, interval=5, progress_threshold=2.0):
        self.progress_threshold = progress_threshold
        self.interval = interval
        self.num_historical_intervals = num_historical_intervals
        super(ProgressReward, self).__init__()

    def forward(self, data):
        # get the point on route
        polyline_batch = data['polyline']['batch']
        polyline_position = data['polyline']['position']
        polyline_heading = data['polyline']['heading']
        polyline_to_polygon_edge_index = data['polyline', 'polygon']['polyline_to_polygon_edge_index']
        polygon_on_route_mask = data['polygon']['on_route_mask']
        polyline_on_route_mask = polygon_on_route_mask[polyline_to_polygon_edge_index[1]]

        # compute the progress of trajectory
        ego_index = data['agent']['ptr'][:-1]
        ego_pos = data['agent']['infer_position'][ego_index, self.num_historical_intervals:].reshape(-1, 2)    # [Num * T, 2]
        ego_pos_pre = data['agent']['infer_position'][ego_index, self.num_historical_intervals-1:-1].reshape(-1, 2)    # [Num * T, 2]
        T = data['agent']['infer_position'].size(1) - self.num_historical_intervals
        ego_batch = data['agent']['batch'][ego_index].repeat_interleave(T)    # [Num * T]

        ego_pos_to_route_mask = torch.ones(ego_pos.size(0), device=ego_pos.device, dtype=torch.bool).unsqueeze(1) & polyline_on_route_mask.unsqueeze(0)    # [Num * T, (R1,...,Rb)]
        ego_pos_to_route_mask = drop_edge_between_samples(ego_pos_to_route_mask, batch=(ego_batch, polyline_batch))
        ego_pos_to_route_edge_index = dense_to_sparse(ego_pos_to_route_mask)[0]
        ego_pos_to_route = transform_point_to_local_coordinate(ego_pos[ego_pos_to_route_edge_index[0]], polyline_position[ego_pos_to_route_edge_index[1]], polyline_heading[ego_pos_to_route_edge_index[1]])
        ego_pos_pre_to_route = transform_point_to_local_coordinate(ego_pos_pre[ego_pos_to_route_edge_index[0]], polyline_position[ego_pos_to_route_edge_index[1]], polyline_heading[ego_pos_to_route_edge_index[1]])
        
        ego_pos_to_route_edge_dist = torch.abs(ego_pos_to_route[:, 1]) * 10 + torch.abs(ego_pos_to_route[:, 0])
        ego_pos_to_route_edge_mask = ego_pos_to_route[:, 0] > 0
        ego_pos_to_route_edge_dist[ego_pos_to_route_edge_mask] += 1000
        _, ego_pos_to_route_dist_min_index = scatter_min(ego_pos_to_route_edge_dist, ego_pos_to_route_edge_index[0], dim_size=ego_pos.size(0))
        valid_index_mask = ego_pos_to_route_dist_min_index != ego_pos_to_route_edge_index.size(1)

        progress = torch.zeros(ego_pos.size(0), device=ego_pos.device)
        progress[valid_index_mask] = ego_pos_to_route[ego_pos_to_route_dist_min_index[valid_index_mask], 0] - ego_pos_pre_to_route[ego_pos_to_route_dist_min_index[valid_index_mask], 0]
        progress = progress.reshape(-1, T).sum(dim=-1)    # [Num]

        # compute the progress of expert trajectory
        expert_pos = data['agent']['position'][ego_index, ::self.interval]
        expert_pos_pre = expert_pos[:, -T-1:-1].reshape(-1, 2)   # [Num * T, 2]
        expert_pos = expert_pos[:, -T:].reshape(-1, 2)           # [Num * T, 2]
        expert_batch = data['agent']['batch'][ego_index].repeat_interleave(T)        # [Num * T]

        expert_pos_to_route_mask = torch.ones(expert_pos.size(0), device=expert_pos.device, dtype=torch.bool).unsqueeze(1) & polyline_on_route_mask.unsqueeze(0)    # [Num * T, (R1,...,Rb)]
        expert_pos_to_route_mask = drop_edge_between_samples(expert_pos_to_route_mask, batch=(expert_batch, polyline_batch))
        expert_pos_to_route_edge_index = dense_to_sparse(expert_pos_to_route_mask)[0]
        expert_pos_to_route = transform_point_to_local_coordinate(expert_pos[expert_pos_to_route_edge_index[0]], polyline_position[expert_pos_to_route_edge_index[1]], polyline_heading[expert_pos_to_route_edge_index[1]])
        expert_pos_pre_to_route = transform_point_to_local_coordinate(expert_pos_pre[expert_pos_to_route_edge_index[0]], polyline_position[expert_pos_to_route_edge_index[1]], polyline_heading[expert_pos_to_route_edge_index[1]])
        
        expert_pos_to_route_edge_dist = torch.abs(expert_pos_to_route[:, 1]) * 10 + torch.abs(expert_pos_to_route[:, 0])
        expert_pos_to_route_mask = expert_pos_to_route[:, 0] > 0
        expert_pos_to_route_edge_dist[expert_pos_to_route_mask] += 1000
        _, expert_pos_to_route_dist_min_index = scatter_min(expert_pos_to_route_edge_dist, expert_pos_to_route_edge_index[0], dim_size=expert_pos.size(0))
        valid_index_mask = expert_pos_to_route_dist_min_index != expert_pos_to_route_edge_index.size(1)

        expert_progress = torch.zeros(expert_pos.size(0), device=expert_pos.device)
        expert_progress[valid_index_mask] = expert_pos_to_route[expert_pos_to_route_dist_min_index[valid_index_mask], 0] - expert_pos_pre_to_route[expert_pos_to_route_dist_min_index[valid_index_mask], 0]
        expert_progress = expert_progress.reshape(-1, T).sum(dim=-1)    # [Num]

        # compute the progress reward
        progress_reward = (progress.clamp(min=self.progress_threshold) / (expert_progress * 1.0).clamp(min=self.progress_threshold)).clamp(max=1.0)

        return progress_reward