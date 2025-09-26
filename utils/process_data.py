import torch
import math
from shapely.geometry import Polygon
from typing import Any, List, Optional, Tuple, Union


def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def get_index_of_a_in_B(a: Optional[Any], list_B: Optional[List[Any]]) -> List[int]: 
    if not a or not list_B or a not in list_B:
        return []

    index = [list_B.index(a)]

    return index

def get_index_of_A_in_B(list_A: Optional[List[Any]], list_B: Optional[List[Any]]) -> List[int]: 
    if not list_A or not list_B:
        return []

    set_B = set(list_B)
    indices = [list_B.index(i) for i in list_A if i in set_B]

    return indices
    
def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix

def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix
    
def compute_angles_lengths_3D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    r_xy = torch.norm(vectors[..., :2], dim=-1)
    phi = torch.atan2(vectors[..., 2], r_xy)
    return length, theta, phi

def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta

def drop_edge_between_samples(valid_mask: torch.Tensor, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        batch_matrix = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    else:
        batch_src, batch_dst = batch
        batch_matrix = batch_src.unsqueeze(-1) == batch_dst.unsqueeze(-2)
    if len(valid_mask.shape) == len(batch_matrix.shape):
        valid_mask = valid_mask * batch_matrix
    elif len(valid_mask.shape) == len(batch_matrix.shape) + 1:
        valid_mask = valid_mask * batch_matrix.unsqueeze(0)
    else:
        raise ValueError("The shape of valid_mask and batch does not match.")
    return valid_mask

def transform_traj_to_local_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    traj = traj - position.unsqueeze(-2)
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    return traj

def transform_traj_to_global_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    traj = traj + position.unsqueeze(-2)
    return traj

def transform_point_to_local_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point

def transform_point_to_global_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    point = point + position
    return point

def generate_reachable_matrix(edge_index: torch.Tensor, num_hops: int, max_nodes: int) -> list:
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    sparse_mat = torch.sparse_coo_tensor(edge_index, values, torch.Size([max_nodes, max_nodes]))

    reach_matrices = []
    current_matrix = sparse_mat.clone()
    for _ in range(num_hops):
        current_matrix = current_matrix.coalesce()
        current_matrix = torch.sparse_coo_tensor(current_matrix.indices(), torch.ones_like(current_matrix.values()), current_matrix.size())

        edge_index_now = current_matrix.coalesce().indices()
        reach_matrices.append(edge_index_now)

        next_matrix = torch.sparse.mm(current_matrix, sparse_mat)

        current_matrix = next_matrix
    return reach_matrices

def compute_corner_positions(position: torch.Tensor, heading: torch.Tensor, box_or_shape: torch.Tensor) -> torch.Tensor:
    # box: [length_front, length_rear, width_left, width_right], shape: [length, width]
    if box_or_shape.size(1) == 4:
        longitudinal_shift_front = torch.stack([box_or_shape[:, 0] * torch.cos(heading), box_or_shape[:, 0] * torch.sin(heading)], dim=-1)
        longitudinal_shift_rear = torch.stack([box_or_shape[:, 1] * torch.cos(heading + math.pi), box_or_shape[:, 1] * torch.sin(heading + math.pi)], dim=-1)
        lateral_shift_left = torch.stack([box_or_shape[:, 2] * torch.cos(heading + math.pi/2), box_or_shape[:, 2] * torch.sin(heading + math.pi/2)], dim=-1)
        lateral_shift_right = torch.stack([box_or_shape[:, 3] * torch.cos(heading - math.pi/2), box_or_shape[:, 3] * torch.sin(heading - math.pi/2)], dim=-1)
    elif box_or_shape.size(1) == 2:
        longitudinal_shift_front = torch.stack([box_or_shape[:, 0] * torch.cos(heading) / 2, box_or_shape[:, 0] * torch.sin(heading) / 2], dim=-1)
        longitudinal_shift_rear = torch.stack([box_or_shape[:, 0] * torch.cos(heading + math.pi) / 2, box_or_shape[:, 0] * torch.sin(heading + math.pi) / 2], dim=-1)
        lateral_shift_left = torch.stack([box_or_shape[:, 1] * torch.cos(heading + math.pi/2) / 2, box_or_shape[:, 1] * torch.sin(heading + math.pi/2) / 2], dim=-1)
        lateral_shift_right = torch.stack([box_or_shape[:, 1] * torch.cos(heading - math.pi/2) / 2, box_or_shape[:, 1] * torch.sin(heading - math.pi/2) / 2], dim=-1)
    else:
        raise ValueError("The shape of box_or_shape is not supported.")
    corners = torch.stack([
        position + longitudinal_shift_front + lateral_shift_left,
        position + longitudinal_shift_front + lateral_shift_right,
        position + longitudinal_shift_rear + lateral_shift_right,
        position + longitudinal_shift_rear + lateral_shift_left
    ], dim=-2)
    return corners

def compute_average_corner_distance(position_1, heading_1, position_2, heading_2, box_or_shape=None):
    if box_or_shape is None:   
        box_or_shape = torch.tensor([1, 1, 1, 1], device=position_1.device).unsqueeze(0)
    else:
        box_or_shape = torch.tensor(box_or_shape, device=position_1.device).unsqueeze(0)
    box_or_shape_1 = box_or_shape.repeat(position_1.size(0), 1)
    box_or_shape_2 = box_or_shape.repeat(position_2.size(0), 1)

    corners_1 = compute_corner_positions(position_1, heading_1, box_or_shape_1)
    corners_2 = compute_corner_positions(position_2, heading_2, box_or_shape_2)
    average_corner_distance = torch.norm(corners_1.unsqueeze(1) - corners_2.unsqueeze(0), dim=-1).mean(dim=-1)

    return average_corner_distance

def sample_with_top_k_top_p(logits:torch.Tensor, top_k:int=0, top_p:float=0.0, num_samples:int=1):
    # adapted from https://github.com/FoundationVision/VAR
    B, l, V = logits.shape
    logits = logits.clone()
    if top_k > 0:
        idx_to_remove = logits < logits.topk(top_k, largest=True, sorted=False)[0].min(dim=-1, keepdim=True)[0]
        logits.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    return torch.multinomial(logits.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=True).view(B, l, num_samples)

def interpolate_traj_using_midpoints(start_position, start_visible_mask, traj_position, traj_heading, traj_visible_mask):
    # compute midpoints
    traj_position = torch.cat([start_position.unsqueeze(1), traj_position], dim=1)
    midpoints_position = (traj_position[:, 1:] + traj_position[:, :-1]) / 2
    midpoints_heading = wrap_angle(torch.atan2(traj_position[:, 1:, 1] - traj_position[:, :-1, 1], traj_position[:, 1:, 0] - traj_position[:, :-1, 0]))
    traj_visible_mask = torch.cat([start_visible_mask.unsqueeze(1), traj_visible_mask], dim=1)
    midpoints_visible_mask = traj_visible_mask[:, 1:] & traj_visible_mask[:, :-1]

    # interpolate
    interpolated_traj_position = torch.stack([midpoints_position, traj_position[:, 1:]], dim=2).view(traj_position.size(0), -1, 2)
    interpolated_traj_heading = torch.stack([midpoints_heading, traj_heading], dim=2).view(traj_heading.size(0), -1)
    interpolated_traj_visible_mask = torch.stack([midpoints_visible_mask, traj_visible_mask[:, 1:]], dim=2).view(traj_visible_mask.size(0), -1)

    return interpolated_traj_position, interpolated_traj_heading, interpolated_traj_visible_mask

def move_dict_to_device(d, device):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.to(device)
        elif isinstance(value, dict):
            move_dict_to_device(value, device)
    return d
