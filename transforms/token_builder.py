import torch
from torch_geometric.transforms import BaseTransform

from utils import transform_point_to_local_coordinate
from utils import transform_point_to_global_coordinate
from utils import wrap_angle
from utils import compute_average_corner_distance

Type = {
    0: 'Vehicle', 
    1: 'Pedestrian', 
    2: 'Bicycle'
}

class TokenBuilder(BaseTransform):
    def __init__(self, token_dict_path, interval, num_historical_steps=20, mode='pred'):
        super(TokenBuilder, self).__init__()
        self.interval = interval
        self.num_historical_steps = num_historical_steps
        self.mode = mode
        
        # load tokens
        self.tokens = torch.load(token_dict_path)

    def __call__(self, data):
        # load data
        position = data['agent']['position']
        heading = data['agent']['heading']
        velocity = data['agent']['velocity']
        visible_mask = data['agent']['visible_mask']
        type = data['agent']['type']

        # process data to ensure one valid token at current step
        interplote_mask = visible_mask[:, self.num_historical_steps] & (~visible_mask[:, self.num_historical_steps - self.interval])
        interplote_position = position[:, self.num_historical_steps] - velocity[:, self.num_historical_steps] * (0.1*self.interval)
        interplote_heading = heading[:, self.num_historical_steps]
        position[:, self.num_historical_steps - self.interval] = torch.where(interplote_mask.unsqueeze(1), interplote_position, position[:, self.num_historical_steps - self.interval])
        heading[:, self.num_historical_steps - self.interval] = torch.where(interplote_mask, interplote_heading, heading[:, self.num_historical_steps - self.interval])
        visible_mask[:, self.num_historical_steps - self.interval] = torch.where(interplote_mask, interplote_mask, visible_mask[:, self.num_historical_steps - self.interval])
        position = position[:,::self.interval]
        heading = heading[:,::self.interval]
        visible_mask = visible_mask[:,::self.interval]
        
        # generate tokens
        recon_token = torch.zeros((position.shape[0], position.shape[1] - 1), dtype=torch.long)
        recon_token_mask = torch.zeros((position.shape[0], position.shape[1] - 1), dtype=torch.bool)
        recon_position = torch.zeros((position.shape[0], position.shape[1], 2), dtype=torch.float)
        recon_heading = torch.zeros((position.shape[0], position.shape[1]), dtype=torch.float)
        recon_valid_mask = torch.zeros((position.shape[0], position.shape[1]), dtype=torch.bool)

        recon_position[:, 0] = position[:, 0]
        recon_heading[:, 0] = heading[:, 0]
        recon_valid_mask[:, 0] = visible_mask[:, 0]

        for step in range(0, position.shape[1] - 1):
            relative_position = transform_point_to_local_coordinate(position[:, step + 1], recon_position[:, step], recon_heading[:, step])
            relative_heading = wrap_angle(heading[:, step + 1] - recon_heading[:, step])
            relative_valid_mask = visible_mask[:, step + 1] & visible_mask[:, step]

            for i in type.unique().tolist():
                mask = type == i
                relative_position_i = relative_position[mask]
                relative_heading_i = relative_heading[mask]
                relative_valid_mask_i = relative_valid_mask[mask]
                tokens_i = self.tokens[Type[i]]
                tokens_to_points_corner_distance = compute_average_corner_distance(tokens_i[:, :2], tokens_i[:, 2], relative_position_i, relative_heading_i)
                target = torch.argmin(tokens_to_points_corner_distance, dim=0)

                recon_token[mask, step] = target
                recon_token_mask[mask, step] = relative_valid_mask_i

                recon_position[mask, step + 1] = torch.where(relative_valid_mask_i.unsqueeze(1), transform_point_to_global_coordinate(tokens_i[target, :2], recon_position[mask, step], recon_heading[mask, step]), position[mask, step + 1])
                recon_heading[mask, step + 1] = torch.where(relative_valid_mask_i, wrap_angle(tokens_i[target, 2] + recon_heading[mask, step]), heading[mask, step + 1])
                recon_valid_mask[mask, step + 1] = torch.where(relative_valid_mask_i, relative_valid_mask_i, visible_mask[mask, step + 1])

        data['agent']['recon_token'] = recon_token
        data['agent']['recon_token_mask'] = recon_token_mask
        data['agent']['recon_position'] = recon_position[:, 1:]
        data['agent']['recon_heading'] = recon_heading[:, 1:]
        data['agent']['recon_valid_mask'] = recon_valid_mask[:, 1:]

        data['agent']['infer_token'] = recon_token[:, :self.num_historical_steps // self.interval]
        data['agent']['infer_token_mask'] = recon_token_mask[:, :self.num_historical_steps // self.interval]
        data['agent']['infer_position'] = recon_position[:, 1:self.num_historical_steps // self.interval + 1]
        data['agent']['infer_heading'] = recon_heading[:, 1:self.num_historical_steps // self.interval + 1]
        data['agent']['infer_valid_mask'] = recon_valid_mask[:, 1: self.num_historical_steps // self.interval + 1]

        return data