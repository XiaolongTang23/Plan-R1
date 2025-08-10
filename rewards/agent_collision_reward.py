import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_max

from utils import transform_point_to_local_coordinate
from utils import compute_corner_positions
from utils import drop_edge_between_samples


class AgentCollisionReward(nn.Module):
    def __init__(self, num_historical_intervals=4, long_threshold=0.0, lat_threshold=0.0):
        self.num_historical_intervals = num_historical_intervals
        self.long_threshold = long_threshold
        self.lat_threshold = lat_threshold
        super(AgentCollisionReward, self).__init__()

    def forward(self, data):
        # get the ego vehicle's trajectory and surrounding agents' trajectory
        pos = data['agent']['infer_position'][:, self.num_historical_intervals:]   # [Num, T, 2]
        yaw = data['agent']['infer_heading'][:, self.num_historical_intervals:]
        mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:]
        box = data['agent']['box']    # [Num, 4]
        corner_pos = compute_corner_positions(
            pos,
            yaw,
            box.unsqueeze(-1)
        )    # [Num, T, 4, 2]

        batch = data['agent']['batch']    # [Num]
        ego_index = data['agent']['ptr'][:-1]    # [Num_ego]
        N, T, _ = pos.size()
        num_ego = ego_index.size(0)
        device = batch.device

        ego_pos = pos[ego_index].transpose(0, 1).reshape(-1, 2)     # [T*Num_ego, 2]
        ego_yaw = yaw[ego_index].transpose(0, 1).reshape(-1)        # [T*Num_ego]
        ego_mask = mask[ego_index].transpose(0, 1)                  # [T, Num_ego]
        ego_batch = batch[ego_index]                                # [Num_ego]
        ego_front = box[ego_index, 0].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num_ego]
        ego_rear = box[ego_index, 1].unsqueeze(0).expand(T, -1).reshape(-1)     # [T*Num_ego]
        ego_left = box[ego_index, 2].unsqueeze(0).expand(T, -1).reshape(-1)     # [T*Num_ego]
        ego_right = box[ego_index, 3].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num_ego]
        ego_corner_pos = corner_pos[ego_index].transpose(0, 1).reshape(-1, 4, 2)    # [T*Num_ego, 4, 2]

        agent_pos = pos.transpose(0, 1).reshape(-1, 2)    # [T*Num, 2]
        agent_yaw = yaw.transpose(0, 1).reshape(-1)    # [T*Num]
        agent_mask = mask.transpose(0, 1)    # [T, Num]
        agent_batch = batch    # [Num]
        agent_front = box[..., 0].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_rear = box[..., 1].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_left = box[..., 2].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_right = box[..., 3].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
        agent_corner_pos = corner_pos.transpose(0, 1).reshape(-1, 4, 2)    # [T*Num, 4, 2]

        edge_mask = ego_mask.unsqueeze(2) & agent_mask.unsqueeze(1)    # [T, Num_ego, Num]
        edge_mask = drop_edge_between_samples(edge_mask, batch=(ego_batch, agent_batch))    # [T, Num_ego, Num]
        self_mask = ego_index.unsqueeze(-1) != torch.arange(N, device=device).unsqueeze(0)    # [1, Num_ego, Num]
        edge_mask = edge_mask & self_mask.unsqueeze(0)    # [T, Num_ego, Num]

        ego_in_agent_edge_index = dense_to_sparse(edge_mask)[0]    # [2, E]
        if ego_in_agent_edge_index.size(1) == 0:
            return torch.zeros(num_ego, T, device=device, dtype=torch.bool), torch.ones(num_ego, device=device)
        ego_corner_in_agent_pos = torch.stack([transform_point_to_local_coordinate(ego_corner_pos[ego_in_agent_edge_index[0], i], agent_pos[ego_in_agent_edge_index[1]], agent_yaw[ego_in_agent_edge_index[1]]) for i in range(4)], dim=1)    # [E, 4, 2]
        agent_in_ego_edge_index = dense_to_sparse(edge_mask.transpose(1,2))[0]    # [2, E]
        agent_corner_in_ego_pos = torch.stack([transform_point_to_local_coordinate(agent_corner_pos[agent_in_ego_edge_index[0], i], ego_pos[agent_in_ego_edge_index[1]], ego_yaw[agent_in_ego_edge_index[1]]) for i in range(4)], dim=1)    # [E, 4, 2]

        agent_corner_in_ego = torch.stack([torch.min(torch.stack([
                torch.clamp((ego_front[agent_in_ego_edge_index[1]] + self.long_threshold - agent_corner_in_ego_pos[..., i, 0]), min=0),
                torch.clamp((ego_rear[agent_in_ego_edge_index[1]] + self.long_threshold + agent_corner_in_ego_pos[..., i, 0]), min=0),
                torch.clamp((ego_left[agent_in_ego_edge_index[1]] + self.lat_threshold - agent_corner_in_ego_pos[..., i, 1]), min=0),
                torch.clamp((ego_right[agent_in_ego_edge_index[1]] + self.lat_threshold + agent_corner_in_ego_pos[..., i, 1]), min=0)
        ], dim=0), dim=0).values for i in range(4)], dim=0)    # [4, E]
        ego_corner_in_agent = torch.stack([torch.min(torch.stack([
                torch.clamp((agent_front[ego_in_agent_edge_index[1]] + self.long_threshold - ego_corner_in_agent_pos[..., i, 0]), min=0),
                torch.clamp((agent_rear[ego_in_agent_edge_index[1]] + self.long_threshold + ego_corner_in_agent_pos[..., i, 0]), min=0),
                torch.clamp((agent_left[ego_in_agent_edge_index[1]] + self.lat_threshold - ego_corner_in_agent_pos[..., i, 1]), min=0),
                torch.clamp((agent_right[ego_in_agent_edge_index[1]] + self.lat_threshold + ego_corner_in_agent_pos[..., i, 1]), min=0)
        ], dim=0), dim=0).values for i in range(4)], dim=0)    # [4, E]

        collision_loss = torch.max(torch.cat([agent_corner_in_ego, ego_corner_in_agent], dim=0), dim=0).values    # [E]
        _, collision_loss_max_index = scatter_max(collision_loss, ego_in_agent_edge_index[0], dim_size=num_ego * T)
        valid_index_mask = collision_loss_max_index != len(ego_in_agent_edge_index[0])

        collision_reward = torch.ones(num_ego * T, device=device, dtype=torch.bool)
        collision_reward[valid_index_mask] = collision_loss[collision_loss_max_index[valid_index_mask]] <= 0
        collision_reward = collision_reward.reshape(T, -1).all(dim=0).float()
        collision_done = torch.zeros(num_ego * T, device=device, dtype=torch.bool)
        collision_done[valid_index_mask] = collision_loss[collision_loss_max_index[valid_index_mask]] > 0
        collision_done = collision_done.reshape(T, -1).transpose(0, 1)
    
        return collision_done, collision_reward