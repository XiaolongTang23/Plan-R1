import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_min
import math

from utils import transform_point_to_local_coordinate
from utils import wrap_angle
from utils import compute_corner_positions
from utils import drop_edge_between_samples
from utils import compute_angles_lengths_2D


class ObstacleCollisionReward(nn.Module):
    def __init__(self, num_historical_intervals=4, threshold=0.0):
        self.threshold = threshold
        self.num_historical_intervals = num_historical_intervals
        super(ObstacleCollisionReward, self).__init__()

    def forward(self, data):
        # Get the ego vehicle's trajectory
        ego_index = data['agent']['ptr'][:-1]
        ego_pos = data['agent']['infer_position'][ego_index, self.num_historical_intervals:]
        ego_yaw = data['agent']['infer_heading'][ego_index, self.num_historical_intervals:]
        ego_box = data['agent']['box'][ego_index] 
        ego_corner_pos = compute_corner_positions(
            ego_pos,
            ego_yaw,
            ego_box.unsqueeze(-1)
        ).reshape(-1, 2)    # [Num * T * 4, 2]
        num_ego, T, _ = ego_pos.size()
        ego_batch = data['agent']['batch'][ego_index].repeat_interleave(4 * T)    # [Num * T * 4]
        device = ego_pos.device

        # Get the static object area boundary
        if data['static_object_area_boundary']['num_nodes'] == 0:
            return torch.zeros(num_ego, T, device=device, dtype=torch.bool), torch.ones(num_ego, device=device)

        soa_boundary_batch = data['static_object_area_boundary']['batch']            # [(S1,...,Sb)], soa: static object area
        soa_boundary_position = data['static_object_area_boundary']['position']      # [(S1,...,Sb), 2]
        soa_boundary_heading = data['static_object_area_boundary']['heading']        # [(S1,...,Sb)]
        soa_boundary_theta = data['static_object_area_boundary']['theta']            # [(S1,...,Sb)]
        soa_boundary_length = data['static_object_area_boundary']['length']          # [(S1,...,Sb)]
        Num_soa_boundary = len(soa_boundary_position)

        # Compute the distance from the ego vehicle to the static object area boundary
        ego_corner_in_soa_mask = torch.ones(num_ego * T * 4, Num_soa_boundary, device=device).bool()         # [Num * T * 4, (S1,...,Sb)]
        ego_corner_in_soa_mask = drop_edge_between_samples(ego_corner_in_soa_mask, batch=(ego_batch, soa_boundary_batch))    # [Num * T * 4, (S1,...,Sb)]

        ego_corner_in_soa_edge_index = dense_to_sparse(ego_corner_in_soa_mask)[0]
        ego_corner_in_soa = transform_point_to_local_coordinate(ego_corner_pos[ego_corner_in_soa_edge_index[0]], soa_boundary_position[ego_corner_in_soa_edge_index[1]], soa_boundary_heading[ego_corner_in_soa_edge_index[1]])
        ego_corner_to_soa_edge_dist = torch.where((ego_corner_in_soa[..., 0] > 0) & (ego_corner_in_soa[..., 0] < soa_boundary_length[ego_corner_in_soa_edge_index[1]]), torch.abs(ego_corner_in_soa[..., 1]), torch.tensor(1000.0).to(device))
        ego_corner_to_soa_node_dist, ego_corner_to_soa_node_theta = compute_angles_lengths_2D(ego_corner_in_soa)
        ego_corner_to_soa_node_theta = wrap_angle(ego_corner_to_soa_node_theta, min_val=0, max_val=2*math.pi)
        ego_corner_to_soa_dist = torch.cat([ego_corner_to_soa_edge_dist, ego_corner_to_soa_node_dist], dim=-1)

        ego_corner_to_soa_edge_loss = torch.clamp(ego_corner_in_soa[..., 1] + self.threshold, min=0.0)
        ego_corner_to_soa_node_loss = torch.clamp((2 * (soa_boundary_theta[ego_corner_in_soa_edge_index[1]] - ego_corner_to_soa_node_theta > 0).float() - 1) * ego_corner_to_soa_node_dist  + self.threshold, min=0.0)
        ego_corner_to_soa_loss = torch.cat([ego_corner_to_soa_edge_loss, ego_corner_to_soa_node_loss], dim=-1)

        ego_corner_to_soa_index = torch.cat([ego_corner_in_soa_edge_index[0], ego_corner_in_soa_edge_index[0]], dim=-1)
        _, ego_corner_to_soa_dist_min_index = scatter_min(ego_corner_to_soa_dist, ego_corner_to_soa_index, dim_size=num_ego * T * 4)
        valid_index_mask = ego_corner_to_soa_dist_min_index != len(ego_corner_to_soa_loss)
        
        obstacle_collision_reward = torch.ones(num_ego * T * 4, device=device, dtype=torch.bool)
        obstacle_collision_reward[valid_index_mask] = ego_corner_to_soa_loss[ego_corner_to_soa_dist_min_index[valid_index_mask]] <= 0
        obstacle_collision_reward = obstacle_collision_reward.reshape(-1, T*4).all(dim=-1).float()    # [Num]
        obstacle_collision_done = torch.zeros(num_ego * T * 4, device=device, dtype=torch.bool)
        obstacle_collision_done[valid_index_mask] = ego_corner_to_soa_loss[ego_corner_to_soa_dist_min_index[valid_index_mask]] > 0
        obstacle_collision_done = obstacle_collision_done.reshape(-1, T, 4).any(dim=-1)        # [Num, T]
        
        return obstacle_collision_done, obstacle_collision_reward