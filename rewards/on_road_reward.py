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


class OnRoadReward(nn.Module):
    def __init__(self, num_historical_intervals=4, threshold=0.0):
        self.threshold = threshold
        self.num_historical_intervals = num_historical_intervals
        super(OnRoadReward, self).__init__()

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
        ).reshape(-1, 2)    # [Num * T, 4, 2]
        num_ego, T, _ = ego_pos.size()
        ego_batch = data['agent']['batch'][ego_index].repeat_interleave(4 * T)    # [Num * T * 4]
        device = ego_pos.device

        if data['drivable_area_boundary']['num_nodes'] == 0:
            return torch.zeros(num_ego, T, device=device, dtype=torch.bool), torch.ones(num_ego, device=device)

        # Get the drivable area boundary
        da_boundary_batch = data['drivable_area_boundary']['batch']            # [(R1,...,Rb)], da: drivable area
        da_boundary_position = data['drivable_area_boundary']['position']      # [(R1,...,Rb), 2]
        da_boundary_heading = data['drivable_area_boundary']['heading']        # [(R1,...,Rb)]
        da_boundary_theta = data['drivable_area_boundary']['theta']            # [(R1,...,Rb)]
        da_boundary_length = data['drivable_area_boundary']['length']          # [(R1,...,Rb)]
        Num_da_boundary = len(da_boundary_position)

        # Compute the distance from the ego vehicle to the drivable area boundary
        ego_corner_in_da_mask = torch.ones(num_ego * T * 4, Num_da_boundary, device=device).bool()         # [Num * T * 4, (R1,...,Rb)]
        ego_corner_in_da_mask = drop_edge_between_samples(ego_corner_in_da_mask, batch=(ego_batch, da_boundary_batch))    # [Num * T * 4, (R1,...,Rb)]

        ego_corner_in_da_edge_index = dense_to_sparse(ego_corner_in_da_mask)[0]
        ego_corner_in_da = transform_point_to_local_coordinate(ego_corner_pos[ego_corner_in_da_edge_index[0]], da_boundary_position[ego_corner_in_da_edge_index[1]], da_boundary_heading[ego_corner_in_da_edge_index[1]])
        ego_corner_to_da_edge_dist = torch.where((ego_corner_in_da[..., 0] > 0) & (ego_corner_in_da[..., 0] < da_boundary_length[ego_corner_in_da_edge_index[1]]), torch.abs(ego_corner_in_da[..., 1]), torch.tensor(1000.0).to(device))
        ego_corner_to_da_node_dist, ego_corner_to_da_node_theta = compute_angles_lengths_2D(ego_corner_in_da)
        ego_corner_to_da_node_theta = wrap_angle(ego_corner_to_da_node_theta, min_val=0, max_val=2*math.pi)
        ego_corner_to_da_dist = torch.cat([ego_corner_to_da_edge_dist, ego_corner_to_da_node_dist], dim=-1)

        ego_corner_to_da_edge_loss = torch.clamp(ego_corner_in_da[..., 1] + self.threshold, min=0.0)
        ego_corner_to_da_node_loss = torch.clamp((2 * (da_boundary_theta[ego_corner_in_da_edge_index[1]] - ego_corner_to_da_node_theta > 0).float() - 1) * ego_corner_to_da_node_dist  + self.threshold, min=0.0)
        ego_corner_to_da_loss = torch.cat([ego_corner_to_da_edge_loss, ego_corner_to_da_node_loss], dim=-1)

        ego_corner_to_da_index = torch.cat([ego_corner_in_da_edge_index[0], ego_corner_in_da_edge_index[0]], dim=-1)
        _, ego_corner_to_da_dist_min_index = scatter_min(ego_corner_to_da_dist, ego_corner_to_da_index, dim_size=num_ego * T * 4)
        valid_index_mask = ego_corner_to_da_dist_min_index != len(ego_corner_to_da_loss)
        
        on_road_reward = torch.ones(num_ego * T * 4, device=device, dtype=torch.bool)
        on_road_reward[valid_index_mask] = ego_corner_to_da_loss[ego_corner_to_da_dist_min_index[valid_index_mask]] <= 0
        on_road_reward = on_road_reward.reshape(-1, T*4).all(dim=-1).float()               # [Num]
        on_road_done = torch.zeros(num_ego * T * 4, device=device, dtype=torch.bool)
        on_road_done[valid_index_mask] = ego_corner_to_da_loss[ego_corner_to_da_dist_min_index[valid_index_mask]] > 0
        on_road_done = on_road_done.reshape(-1, T, 4).any(dim=-1)        # [Num, T]
        
        return on_road_done, on_road_reward