import torch
import torch.nn as nn

from utils import wrap_angle

class SpeedLimitReward(nn.Module):
    def __init__(self, time_span_per_interval=0.5, num_historical_intervals=4, max_overspeed_value_threshold=2.23):
        self.time_span_per_interval = time_span_per_interval
        self.num_historical_intervals = num_historical_intervals
        self.max_overspeed_value_threshold = max_overspeed_value_threshold
        super(SpeedLimitReward, self).__init__()

    def forward(self, data):
        # Get the ego vehicle's speed
        ego_index = data['agent']['ptr'][:-1]
        ego_pos = data['agent']['infer_position'][ego_index]
        ego_vel = (ego_pos[:, 1:] - ego_pos[:, :-1]) / self.time_span_per_interval
        T = ego_pos.size(1) - self.num_historical_intervals
        ego_speed = torch.norm(ego_vel, dim=-1)[:, -T:]

        # Get the speed limit 
        ego_pos = data['agent']['infer_position'][ego_index, self.num_historical_intervals:].reshape(-1, 2)  # [Num * T, 2]
        ego_yaw = data['agent']['infer_heading'][ego_index, self.num_historical_intervals:].reshape(-1)  # [Num * T]
        polyline_position = data['polyline']['position']
        polyline_heading = data['polyline']['heading']
        polyline_batch = data['polyline']['batch']
        ego_pos_to_polyline_dist = torch.norm(ego_pos.unsqueeze(1) - polyline_position.unsqueeze(0), dim=-1)
        ego_pos_to_polyline_angle_dist = torch.abs(wrap_angle(ego_yaw.unsqueeze(1) - polyline_heading.unsqueeze(0)))

        # filter invalid polyline in different batch
        ego_batch = data['agent']['batch'][ego_index].repeat_interleave(T)  # [Num * T]
        batch_mask = ego_batch.unsqueeze(1) == polyline_batch.unsqueeze(0)
        ego_pos_to_polyline_dist[~batch_mask] = float('inf')
        ego_pos_to_polyline_angle_dist[~batch_mask] = float('inf')

        # lane only
        polygon_type = data['polygon']['type']
        polyline_to_polygon_edge_index = data['polyline', 'polygon']['polyline_to_polygon_edge_index']
        polyline_type = torch.zeros(polyline_position.size(0), device=polyline_position.device, dtype=torch.uint8)
        polyline_type[polyline_to_polygon_edge_index[0]] = polygon_type[polyline_to_polygon_edge_index[1]]
        type_mask = polyline_type == 0
        ego_pos_to_polyline_dist[:, ~type_mask] = float('inf')
        ego_pos_to_polyline_angle_dist[:, ~type_mask] = float('inf')

        # get the closest polyline
        ego_pos_to_polyline_min_index = torch.argmin(ego_pos_to_polyline_dist + 5 * ego_pos_to_polyline_angle_dist, dim=1)

        # get the speed limit of the closest polyline
        polygon_speed_limit = data['polygon']['speed_limit']
        polygon_speed_limit_valid_mask = data['polygon']['speed_limit_valid_mask']
        polyline_speed_limit = polygon_speed_limit[polyline_to_polygon_edge_index[1]]
        polyline_speed_limit_valid_mask = polygon_speed_limit_valid_mask[polyline_to_polygon_edge_index[1]]
        closest_polyline_speed_limit = polyline_speed_limit[ego_pos_to_polyline_min_index].reshape(-1, T)
        closest_polyline_speed_limit_valid_mask = polyline_speed_limit_valid_mask[ego_pos_to_polyline_min_index].reshape(-1, T)

        # compute the speed limit reward
        speed_limit_reward = (ego_speed - closest_polyline_speed_limit).clamp(min=0.0) * closest_polyline_speed_limit_valid_mask.float() * self.time_span_per_interval
        speed_limit_reward = 1 - speed_limit_reward.sum(-1) / ((self.max_overspeed_value_threshold * T * self.time_span_per_interval))
        speed_limit_reward = speed_limit_reward.clamp(min=0.0)
        
        return speed_limit_reward