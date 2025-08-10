import torch
import torch.nn as nn

from utils import wrap_angle

class ComfortReward(nn.Module):
    def __init__(self, time_span_per_interval=0.5, num_historical_intervals=4):
        self.time_span_per_interval = time_span_per_interval
        self.num_historical_intervals = num_historical_intervals
        self.min_lon_accel_threshold = -4.05
        self.max_lon_accel_threshold = 2.40
        self.max_abs_lat_accel_threshold = 4.89
        self.max_abs_yaw_accel_threshold = 1.93
        self.max_abs_yaw_rate_threshold = 0.95
        super(ComfortReward, self).__init__()

    def forward(self, data):
        # Get the ego vehicle's position and orientation
        ego_index = data['agent']['ptr'][:-1]
        ego_pos = data['agent']['infer_position'][ego_index]
        ego_yaw = data['agent']['infer_heading'][ego_index]
        T = ego_pos.size(1) - self.num_historical_intervals

        # compute the ego vehicle's acceleration and angular velocity
        ego_vel = (ego_pos[:, 1:] - ego_pos[:, :-1]) / self.time_span_per_interval
        ego_acc = (ego_vel[:, 1:] - ego_vel[:, :-1]) / self.time_span_per_interval
        ego_angular_vel = wrap_angle((ego_yaw[:, 1:] - ego_yaw[:, :-1])) / self.time_span_per_interval
        ego_angular_acc = (ego_angular_vel[:, 1:] - ego_angular_vel[:, :-1]) / self.time_span_per_interval

        ego_acc = ego_acc[:, -T:]
        ego_angular_vel = ego_angular_vel[:, -T:]
        ego_angular_acc = ego_angular_acc[:, -T:]

        # rotation matrix to transform from global to ego's local coordinates
        rotation_matrix = torch.stack([
            torch.cos(ego_yaw[:, -T:]), torch.sin(ego_yaw[:, -T:]),
            -torch.sin(ego_yaw[:, -T:]), torch.cos(ego_yaw[:, -T:])
        ], dim=-1).view(-1, T, 2, 2).to(ego_pos.device)

        # transform to ego's local coordinates
        ego_acc_local = torch.matmul(rotation_matrix, ego_acc.unsqueeze(-1)).squeeze(-1)
        ego_acc_long = ego_acc_local[..., 0]
        ego_acc_lat = ego_acc_local[..., 1]

        # compute the comfort reward
        comfort_reward = (ego_acc_long > self.min_lon_accel_threshold) & \
            (ego_acc_long < self.max_lon_accel_threshold) & \
            (torch.abs(ego_acc_lat) < self.max_abs_lat_accel_threshold) & \
            (torch.abs(ego_angular_acc) < self.max_abs_yaw_accel_threshold) & \
            (torch.abs(ego_angular_vel) < self.max_abs_yaw_rate_threshold)
        comfort_reward = comfort_reward.all(dim=-1).float()    # [Num]

        return comfort_reward