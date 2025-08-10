import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_max

from utils import transform_point_to_local_coordinate
from utils import compute_corner_positions
from utils import drop_edge_between_samples

class TTCReward(nn.Module):
    def __init__(self, long_threshold=0.0, lat_threshold=0.0, num_historical_intervals=4, least_min_ttc=0.95, time_span_per_interval=0.5):
        self.long_threshold = long_threshold
        self.lat_threshold = lat_threshold
        self.num_historical_intervals = num_historical_intervals
        self.least_min_ttc = least_min_ttc
        self.time_span_per_interval = time_span_per_interval
        super(TTCReward, self).__init__()

    def forward(self, data):
        # get the ego vehicle's trajectory and surrounding agents' trajectory
        vel = data['agent']['infer_position'][:, self.num_historical_intervals:] - data['agent']['infer_position'][:, self.num_historical_intervals-1:-1]    # [Num, T, 2]
        vel = vel / self.time_span_per_interval    # [Num, T, 2]
        pos_ttc = data['agent']['infer_position'][:, self.num_historical_intervals:] + vel * self.least_min_ttc    # [Num, T, 2]
        yaw_ttc = data['agent']['infer_heading'][:, self.num_historical_intervals:]    # [Num, T]
        box = data['agent']['box']    # [Num, 4]
        corner_pos_ttc = compute_corner_positions(
            pos_ttc,
            yaw_ttc,
            box.unsqueeze(-1)
        )    # [Num, T, 4, 2]

        batch = data['agent']['batch']    # [Num]
        ego_index = data['agent']['ptr'][:-1]    # [Num_ego]
        num_ego = len(ego_index)
        N, T, _ = vel.size()
        device = batch.device

        ego_pos_ttc = pos_ttc[ego_index].transpose(0, 1).reshape(-1, 2)     # [T*Num_ego, 2]
        ego_yaw_ttc = yaw_ttc[ego_index].transpose(0, 1).reshape(-1)        # [T*Num_ego]
        ego_mask = data['agent']['infer_valid_mask'][ego_index, self.num_historical_intervals:].transpose(0, 1)    # [T, Num_ego]
        ego_batch = batch[ego_index]                                        # [Num_ego]
        ego_front = box[ego_index, 0].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num_ego]
        ego_rear = box[ego_index, 1].unsqueeze(0).expand(T, -1).reshape(-1)     # [T*Num_ego]
        ego_left = box[ego_index, 2].unsqueeze(0).expand(T, -1).reshape(-1)     # [T*Num_ego]
        ego_right = box[ego_index, 3].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num_ego]
        ego_corner_pos_ttc = corner_pos_ttc[ego_index].transpose(0, 1).reshape(-1, 4, 2)    # [T*Num_ego, 4, 2]

        agent_pos_ttc = pos_ttc.transpose(0, 1).reshape(-1, 2)    # [T*Num, 2]
        agent_yaw_ttc = yaw_ttc.transpose(0, 1).reshape(-1)    # [T*Num]
        agent_mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:].transpose(0, 1)    # [T, Num]
        agent_batch = batch    # [Num]
        agent_front = box[..., 0].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_rear = box[..., 1].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_left = box[..., 2].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_right = box[..., 3].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_corner_pos_ttc = corner_pos_ttc.transpose(0, 1).reshape(-1, 4, 2)    # [T*Num, 4, 2]

        edge_mask = ego_mask.unsqueeze(2) & agent_mask.unsqueeze(1)    # [T, Num_ego, Num]
        edge_mask = drop_edge_between_samples(edge_mask, batch=(ego_batch, agent_batch))    # [T, Num_ego, Num]
        self_mask = ego_index.unsqueeze(-1) != torch.arange(N, device=device).unsqueeze(0)    # [1, Num_ego, Num]
        edge_mask = edge_mask & self_mask.unsqueeze(0)    # [T, Num_ego, Num]

        ego_in_agent_edge_index = dense_to_sparse(edge_mask)[0]    # [2, E]
        if ego_in_agent_edge_index.size(1) == 0:
            return torch.ones(num_ego, device=device)
        agent_in_ego_edge_index = dense_to_sparse(edge_mask.transpose(1,2))[0]    # [2, E]
        
        ego_corner_in_agent_pos_ttc = torch.stack([transform_point_to_local_coordinate(ego_corner_pos_ttc[ego_in_agent_edge_index[0], i], agent_pos_ttc[ego_in_agent_edge_index[1]], agent_yaw_ttc[ego_in_agent_edge_index[1]]) for i in range(4)], dim=1)    # [E, 4, 2]
        agent_corner_in_ego_pos_ttc = torch.stack([transform_point_to_local_coordinate(agent_corner_pos_ttc[agent_in_ego_edge_index[0], i], ego_pos_ttc[agent_in_ego_edge_index[1]], ego_yaw_ttc[agent_in_ego_edge_index[1]]) for i in range(4)], dim=1)    # [E, 4, 2]

        agent_corner_in_ego_ttc = torch.stack([torch.min(torch.stack([
                torch.clamp((ego_front[agent_in_ego_edge_index[1]] + self.long_threshold - agent_corner_in_ego_pos_ttc[..., i, 0]), min=0),
                torch.clamp((ego_rear[agent_in_ego_edge_index[1]] + self.long_threshold + agent_corner_in_ego_pos_ttc[..., i, 0]), min=0),
                torch.clamp((ego_left[agent_in_ego_edge_index[1]] + self.lat_threshold - agent_corner_in_ego_pos_ttc[..., i, 1]), min=0),
                torch.clamp((ego_right[agent_in_ego_edge_index[1]] + self.lat_threshold + agent_corner_in_ego_pos_ttc[..., i, 1]), min=0)
        ], dim=0), dim=0).values for i in range(4)], dim=0)    # [4, E]
        ego_corner_in_agent_ttc = torch.stack([torch.min(torch.stack([
                torch.clamp((agent_front[ego_in_agent_edge_index[1]] + self.long_threshold - ego_corner_in_agent_pos_ttc[..., i, 0]), min=0),
                torch.clamp((agent_rear[ego_in_agent_edge_index[1]] + self.long_threshold + ego_corner_in_agent_pos_ttc[..., i, 0]), min=0),
                torch.clamp((agent_left[ego_in_agent_edge_index[1]] + self.lat_threshold - ego_corner_in_agent_pos_ttc[..., i, 1]), min=0),
                torch.clamp((agent_right[ego_in_agent_edge_index[1]] + self.lat_threshold + ego_corner_in_agent_pos_ttc[..., i, 1]), min=0)
        ], dim=0), dim=0).values for i in range(4)], dim=0)    # [4, E]

        ttc_loss = torch.max(torch.cat([agent_corner_in_ego_ttc, ego_corner_in_agent_ttc]), dim=0).values    # [E]
        _, ttc_loss_max_index = scatter_max(ttc_loss, ego_in_agent_edge_index[0], dim_size=num_ego * T)
        valid_index_mask = ttc_loss_max_index != len(ego_in_agent_edge_index[0])
        
        ttc_reward = torch.ones(num_ego * T, device=device, dtype=torch.bool)
        ttc_reward[valid_index_mask] = ttc_loss[ttc_loss_max_index[valid_index_mask]] <= 0
        ttc_reward = ttc_reward.reshape(T, -1).all(dim=0).float()
        
        return ttc_reward