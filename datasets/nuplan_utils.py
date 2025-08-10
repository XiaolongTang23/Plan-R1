import logging
import os
from typing import List, Dict, Optional
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import math
import torch
import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from utils.process_data import get_index_of_a_in_B
from utils.process_data import get_index_of_A_in_B
from utils.process_data import compute_angles_lengths_2D
from utils.process_data import wrap_angle
from utils.process_data import generate_counterclockwise_rotation_matrix

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_scenario_map():
    scenario_map = {
        'accelerating_at_crosswalk': [15.0, -3.0],
        'accelerating_at_stop_sign': [15.0, -3.0],
        'accelerating_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'accelerating_at_traffic_light': [15.0, -3.0],
        'accelerating_at_traffic_light_with_lead': [15.0, -3.0],
        'accelerating_at_traffic_light_without_lead': [15.0, -3.0],
        'behind_bike': [15.0, -3.0],
        'behind_long_vehicle': [15.0, -3.0],
        'behind_pedestrian_on_driveable': [15.0, -3.0],
        'behind_pedestrian_on_pickup_dropoff': [15.0, -3.0],
        'changing_lane': [15.0, -3.0],
        'changing_lane_to_left': [15.0, -3.0],
        'changing_lane_to_right': [15.0, -3.0],
        'changing_lane_with_lead': [15.0, -3.0],
        'changing_lane_with_trail': [15.0, -3.0],
        'crossed_by_bike': [15.0, -3.0],
        'crossed_by_vehicle': [15.0, -3.0],
        'following_lane_with_lead': [15.0, -3.0],
        'following_lane_with_slow_lead': [15.0, -3.0],
        'following_lane_without_lead': [15.0, -3.0],
        'high_lateral_acceleration': [15.0, -3.0],
        'high_magnitude_jerk': [15.0, -3.0],
        'high_magnitude_speed': [15.0, -3.0],
        'low_magnitude_speed': [15.0, -3.0],
        'medium_magnitude_speed': [15.0, -3.0],
        'near_barrier_on_driveable': [15.0, -3.0],
        'near_construction_zone_sign': [15.0, -3.0],
        'near_high_speed_vehicle': [15.0, -3.0],
        'near_long_vehicle': [15.0, -3.0],
        'near_multiple_bikes': [15.0, -3.0],
        'near_multiple_pedestrians': [15.0, -3.0],
        'near_multiple_vehicles': [15.0, -3.0],
        'near_pedestrian_at_pickup_dropoff': [15.0, -3.0],
        'near_pedestrian_on_crosswalk': [15.0, -3.0],
        'near_pedestrian_on_crosswalk_with_ego': [15.0, -3.0],
        'near_trafficcone_on_driveable': [15.0, -3.0],
        'on_all_way_stop_intersection': [15.0, -3.0],
        'on_carpark': [15.0, -3.0],
        'on_intersection': [15.0, -3.0],
        'on_pickup_dropoff': [15.0, -3.0],
        'on_stopline_crosswalk': [15.0, -3.0],
        'on_stopline_stop_sign': [15.0, -3.0],
        'on_stopline_traffic_light': [15.0, -3.0],
        'on_traffic_light_intersection': [15.0, -3.0],
        'starting_high_speed_turn': [15.0, -3.0],
        'starting_left_turn': [15.0, -3.0],
        'starting_low_speed_turn': [15.0, -3.0],
        'starting_protected_cross_turn': [15.0, -3.0],
        'starting_protected_noncross_turn': [15.0, -3.0],
        'starting_right_turn': [15.0, -3.0],
        'starting_straight_stop_sign_intersection_traversal': [15.0, -3.0],
        'starting_straight_traffic_light_intersection_traversal': [15.0, -3.0],
        'starting_u_turn': [15.0, -3.0],
        'starting_unprotected_cross_turn': [15.0, -3.0],
        'starting_unprotected_noncross_turn': [15.0, -3.0],
        'stationary': [15.0, -3.0],
        'stationary_at_crosswalk': [15.0, -3.0],
        'stationary_at_traffic_light_with_lead': [15.0, -3.0],
        'stationary_at_traffic_light_without_lead': [15.0, -3.0],
        'stationary_in_traffic': [15.0, -3.0],
        'stopping_at_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_with_lead': [15.0, -3.0],
        'stopping_at_stop_sign_without_lead': [15.0, -3.0],
        'stopping_at_traffic_light_with_lead': [15.0, -3.0],
        'stopping_at_traffic_light_without_lead': [15.0, -3.0],
        'stopping_with_lead': [15.0, -3.0],
        'traversing_crosswalk': [15.0, -3.0],
        'traversing_intersection': [15.0, -3.0],
        'traversing_narrow_lane': [15.0, -3.0],
        'traversing_pickup_dropoff': [15.0, -3.0],
        'traversing_traffic_light_intersection': [15.0, -3.0],
        'waiting_for_pedestrian_to_cross': [15.0, -3.0]
    }

    return scenario_map

def get_plan_scenario_types():
    scenario_types = [
        'behind_long_vehicle',
        'changing_lane',
        'following_lane_with_lead',
        'high_lateral_acceleration',
        'high_magnitude_speed',
        'low_magnitude_speed',
        'near_multiple_vehicles',
        'starting_left_turn',
        'starting_right_turn',
        'starting_straight_traffic_light_intersection_traversal',
        'stationary_in_traffic',
        'stopping_with_lead',
        'traversing_pickup_dropoff',
        'waiting_for_pedestrian_to_cross'
    ]

    return scenario_types

def get_filter_parameters(num_scenarios_per_type=None, 
                          limit_total_scenarios=None,
                          timestamp_threshold_s=None,
                          scenario_types=None,
                          scenario_tokens=None):
    scenario_types = scenario_types                # List of scenario types to include
    scenario_tokens = scenario_tokens              # List of scenario tokens to include
    
    log_names = None                     # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type = num_scenarios_per_type    # Number of scenarios per type
    limit_total_scenarios = limit_total_scenarios       # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = timestamp_threshold_s          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
    shuffle = False                      # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

def set_default_path():
    """
    This function sets the default paths as environment variables if none are set.
    These can then be used by Hydra, unless the user overwrites them from the command line.
    """
    logger = logging.getLogger(__name__)

    DEFAULT_DATA_ROOT = os.path.expanduser('~/nuplan/dataset')
    DEFAULT_EXP_ROOT = os.path.expanduser('~/nuplan/exp')
    DEFAULT_MAPS_ROOT = os.path.expanduser('~/nuplan/dataset/maps')

    if 'NUPLAN_DATA_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_DATA_ROOT: {DEFAULT_DATA_ROOT}')
        os.environ['NUPLAN_DATA_ROOT'] = DEFAULT_DATA_ROOT

    if 'NUPLAN_EXP_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_EXP_ROOT: {DEFAULT_EXP_ROOT}')
        os.environ['NUPLAN_EXP_ROOT'] = DEFAULT_EXP_ROOT

    if 'NUPLAN_MAPS_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_MAPS_ROOT: {DEFAULT_MAPS_ROOT}')
        os.environ['NUPLAN_MAPS_ROOT'] = DEFAULT_MAPS_ROOT

def filter_agents_according_to_type_and_distance(agents_ids: List, agents_position: torch.Tensor, agents_type: List, ego_position: torch.Tensor, max_agents: List) -> List:
    filtered_agents_ids = []
    for i in range(len(max_agents)):
        mask = [agent_type == i for agent_type in agents_type]
        ids = [agents_ids[j] for j in range(len(agents_ids)) if mask[j]]
        if len(ids) > max_agents[i]:
            position = agents_position[mask]
            distance = torch.norm(position - ego_position, dim=-1)
            _, indices = torch.topk(distance, max_agents[i], largest=False)
            ids = [ids[j] for j in indices]
        filtered_agents_ids += ids
    return filtered_agents_ids

def get_features(ego_state_buffer, 
                 observation_buffer,
                 map_api, 
                 traffic_lights, 
                 route_roadblock_ids, 
                 num_historical_steps: int=20, 
                 num_future_steps: int=80,
                 max_agents: int = 60,
                 max_lanes: int = 80,
                 max_crosswalks: int = 5,
                 max_drivable_area_segments: int = 30,
                 max_static_objects: int = 30,
                 radius: float = 120,
                 margin: float = 20,
                 is_simulation: bool=False,
                 static_object_tokens: List[str]=None,
                 dynamic_object_tokens: List[str]=None) -> Dict:
    # define
    _agent_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    _agent_identity = ['EGO', 'AGENT']
    _static_object_types = [TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER, TrackedObjectType.TRAFFIC_CONE,TrackedObjectType.GENERIC_OBJECT]
    _map_features = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR, SemanticMapLayer.CARPARK_AREA, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION]
    _polygon_types = ['LANE', 'CROSSWALK', 'DRIVABLE_AREA_SEGMENT','STATIC_OBJECT']
    _polyline_types = ['CENTERLINE', 'BOUNDARY']
    _traffic_light_types = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN', 'NONE']

    data = {
        'agent': {},
        'polygon': {},
        'polyline': {},
        ('polyline', 'polygon'): {},
        ('polygon', 'polygon'): {},
        'drivable_area_boundary': {},
        'static_object_area_boundary': {},
    }

    present_ego_agent = ego_state_buffer[num_historical_steps]
    present_observation = observation_buffer[num_historical_steps]

    # Agent
    present_neighbor_agents = present_observation.tracked_objects.get_tracked_objects_of_types(_agent_types)

    # filter agents according to type and distance if dynamic_object_tokens is None
    if dynamic_object_tokens is None:
        neighbor_agents_ids = list(neighbor_agent.track_token for neighbor_agent in present_neighbor_agents)
        present_ego_agent_position = torch.tensor([present_ego_agent.rear_axle.x, present_ego_agent.rear_axle.y], dtype=torch.float)
        if len(neighbor_agents_ids) != 0:
            present_neighbor_agents_position = torch.tensor([[agent.center.x, agent.center.y] for agent in present_neighbor_agents], dtype=torch.float)
            present_relative_position_length = torch.norm(present_neighbor_agents_position - present_ego_agent_position, dim=-1)
            if len(neighbor_agents_ids) > max_agents:
                _, present_relative_position_index = torch.topk(present_relative_position_length, max_agents, largest=False)
            else:
                _, present_relative_position_index = torch.topk(present_relative_position_length, len(neighbor_agents_ids), largest=False)
            agents_ids = [neighbor_agents_ids[i] for i in present_relative_position_index]
        else:
            agents_ids = []
    else:
        agents_ids = dynamic_object_tokens
    num_agents = len(agents_ids) + 1
    
    # initialization
    if is_simulation:
        num_future_steps = 0
    agent_ids = [present_ego_agent.agent.track_token] + agents_ids
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_identity = torch.zeros(num_agents, dtype=torch.uint8)
    agent_box = torch.zeros(num_agents, 4, dtype=torch.float32)
    agent_visible_mask = torch.zeros(num_agents, num_historical_steps + num_future_steps + 1, dtype=torch.bool)
    agent_position = torch.zeros(num_agents, num_historical_steps + num_future_steps + 1, 2, dtype=torch.float64)
    agent_heading = torch.zeros(num_agents, num_historical_steps + num_future_steps + 1, dtype=torch.float32)
    agent_velocity = torch.zeros(num_agents, num_historical_steps + num_future_steps + 1, 2, dtype=torch.float32)

    # agent features
    agent_type[0] = torch.tensor(_agent_types.index(TrackedObjectType.VEHICLE), dtype=torch.uint8)
    agent_identity[0] = torch.tensor(_agent_identity.index('EGO'), dtype=torch.uint8)
    agent_box[0] = torch.tensor([present_ego_agent.agent.box.length / 2 + present_ego_agent.car_footprint.rear_axle_to_center_dist,
                                    present_ego_agent.agent.box.length / 2 - present_ego_agent.car_footprint.rear_axle_to_center_dist,
                                    present_ego_agent.agent.box.width / 2, 
                                    present_ego_agent.agent.box.width / 2], dtype=torch.float32)
    for i in range(num_historical_steps + 1 + num_future_steps):
        # ego
        agent_position[0, i, 0] = ego_state_buffer[i].rear_axle.x
        agent_position[0, i, 1] = ego_state_buffer[i].rear_axle.y
        agent_visible_mask[0, i] = True
        agent_heading[0, i] = ego_state_buffer[i].rear_axle.heading
        agent_velocity[0, i, 0] = ego_state_buffer[i].dynamic_car_state.rear_axle_velocity_2d.x
        agent_velocity[0, i, 1] = ego_state_buffer[i].dynamic_car_state.rear_axle_velocity_2d.y

        # neighbor
        for neighbor_agent in observation_buffer[i].tracked_objects.get_tracked_objects_of_types(_agent_types):
            agent_id = neighbor_agent.track_token
            if agent_id not in agent_ids:
                continue
            index = agent_ids.index(agent_id)

            if agent_box[index].sum() == 0:
                agent_box[index] = torch.tensor([neighbor_agent.box.length / 2,
                                                neighbor_agent.box.length / 2,
                                                neighbor_agent.box.width / 2, 
                                                neighbor_agent.box.width / 2], dtype=torch.float32)
                agent_type[index] = torch.tensor(_agent_types.index(neighbor_agent.tracked_object_type), dtype=torch.uint8)
                agent_identity[index] = torch.tensor(_agent_identity.index('AGENT'), dtype=torch.uint8)
            agent_position[index, i, 0] = neighbor_agent.center.x
            agent_position[index, i, 1] = neighbor_agent.center.y
            agent_visible_mask[index, i] = True
            agent_heading[index, i] = neighbor_agent.center.heading
            agent_velocity[index, i, 0] = neighbor_agent.velocity.x
            agent_velocity[index, i, 1] = neighbor_agent.velocity.y
    
    # the velocity of the ego agent is in the ego's local coordinate
    agent_velocity[0] = torch.matmul(generate_counterclockwise_rotation_matrix(agent_heading[0]), agent_velocity[0].unsqueeze(-1)).squeeze(-1)

    # get relative position
    center_position = torch.tensor([present_ego_agent.rear_axle.x, present_ego_agent.rear_axle.y], dtype=torch.float64)
    agent_position = torch.where(agent_visible_mask.unsqueeze(-1), agent_position - center_position, torch.tensor(0.0))

    data['agent']['num_nodes'] = num_agents
    data['agent']['id'] = agent_ids
    data['agent']['type'] = agent_type
    data['agent']['identity'] = agent_identity
    data['agent']['box'] = agent_box
    data['agent']['visible_mask'] = agent_visible_mask
    data['agent']['position'] = agent_position.to(torch.float32)
    data['agent']['heading'] = agent_heading
    data['agent']['velocity'] = agent_velocity

    # map
    if not is_simulation:
        radius = torch.norm(agent_position[agent_visible_mask], dim=-1).max().item() + margin
    map_objects = map_api.get_proximal_map_objects(Point2D(center_position[0].item(), center_position[1].item()), radius, _map_features)
    lanes = map_objects[SemanticMapLayer.LANE] + map_objects[SemanticMapLayer.LANE_CONNECTOR]
    crosswalks = map_objects[SemanticMapLayer.CROSSWALK]
    carparks = map_objects[SemanticMapLayer.CARPARK_AREA]
    intersections = map_objects[SemanticMapLayer.INTERSECTION]
    roadblocks = map_objects[SemanticMapLayer.ROADBLOCK] + map_objects[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    static_objects = present_observation.tracked_objects.get_tracked_objects_of_types(_static_object_types)
    
    # filter map objects
    if not is_simulation:
        max_x = agent_position[agent_visible_mask][:, 0].max().item() + center_position[0].item() + margin
        min_x = agent_position[agent_visible_mask][:, 0].min().item() + center_position[0].item() - margin
        max_y = agent_position[agent_visible_mask][:, 1].max().item() + center_position[1].item() + margin
        min_y = agent_position[agent_visible_mask][:, 1].min().item() + center_position[1].item() - margin
        patch = box(min_x, min_y, max_x, max_y)

        lane_in_patch_mask = [lane.polygon.intersects(patch) for lane in lanes]
        lanes = [lane for lane, mask in zip(lanes, lane_in_patch_mask) if mask]
        crosswalk_in_patch_mask = [crosswalk.polygon.intersects(patch) for crosswalk in crosswalks]
        crosswalks = [crosswalk for crosswalk, mask in zip(crosswalks, crosswalk_in_patch_mask) if mask]
        carpark_in_patch_mask = [carpark.polygon.intersects(patch) for carpark in carparks]
        carparks = [carpark for carpark, mask in zip(carparks, carpark_in_patch_mask) if mask]
        intersection_in_patch_mask = [intersection.polygon.intersects(patch) for intersection in intersections]
        intersections = [intersection for intersection, mask in zip(intersections, intersection_in_patch_mask) if mask]
        roadblock_in_patch_mask = [roadblock.polygon.intersects(patch) for roadblock in roadblocks]
        roadblocks = [roadblock for roadblock, mask in zip(roadblocks, roadblock_in_patch_mask) if mask]
        static_object_in_patch_mask = [static_object.box.geometry.intersects(patch) for static_object in static_objects]
        static_objects = [static_object for static_object, mask in zip(static_objects, static_object_in_patch_mask) if mask]

    # drivalbe area
    drivable_area_boundary = []
    for carpark in carparks:
        boundary = torch.tensor(np.array(carpark.polygon.exterior.coords), dtype=torch.float64) - center_position
        drivable_area_boundary.append(Polygon(boundary.cpu().numpy()))
    for intersection in intersections:
        boundary = torch.tensor(np.array(intersection.polygon.exterior.coords), dtype=torch.float64) - center_position
        drivable_area_boundary.append(Polygon(boundary.cpu().numpy()))
    for roadblock in roadblocks:
        boundary = torch.tensor(np.array(roadblock.polygon.exterior.coords), dtype=torch.float64) - center_position
        drivable_area_boundary.append(Polygon(boundary.cpu().numpy()))

    drivable_area = unary_union(drivable_area_boundary).buffer(0).simplify(0.1, preserve_topology=True).buffer(0)
    if drivable_area.geom_type == 'Polygon':
        drivable_area_exterior_coords = [drivable_area.exterior.coords]
        drivable_area_interior_coords = [interior.coords for interior in drivable_area.interiors if Polygon(interior).area > 5]
        drivable_area_coords = drivable_area_exterior_coords + drivable_area_interior_coords
    elif drivable_area.geom_type == 'MultiPolygon':
        drivable_area_exterior_coords = [polygon.exterior.coords for polygon in drivable_area.geoms]
        drivable_area_interior_coords = [interior.coords for polygon in drivable_area.geoms for interior in polygon.interiors if Polygon(interior).area > 5]
        drivable_area_coords = drivable_area_exterior_coords + drivable_area_interior_coords
    
    num_drivable_area = len(drivable_area_coords)
    num_drivable_area_boundary = torch.zeros(num_drivable_area, dtype=torch.long)
    drivable_area_boundary_position = [None] * num_drivable_area
    drivable_area_boundary_length = [None] * num_drivable_area
    drivable_area_boundary_heading = [None] * num_drivable_area
    drivable_area_boundary_theta = [None] * num_drivable_area
    num_drivable_area_segment = torch.zeros(num_drivable_area, dtype=torch.long)
    num_drivable_area_boundary_processed = 0
    drivable_area_segment_split_index = [None] * num_drivable_area
    for i in range(num_drivable_area):
        drivable_area_boundary = torch.tensor([point for point in drivable_area_coords[i]], dtype=torch.float32)
        drivable_area_boundary_position[i] = drivable_area_boundary[:-1]
        drivable_area_boundary_length[i], drivable_area_boundary_heading[i] = compute_angles_lengths_2D(drivable_area_boundary[1:] - drivable_area_boundary[:-1])
        drivable_area_boundary_theta[i] = wrap_angle(math.pi + torch.roll(drivable_area_boundary_heading[i], 1) - drivable_area_boundary_heading[i], min_val=0.0, max_val=2*math.pi)
        num_drivable_area_boundary[i] = len(drivable_area_boundary) - 1

        num_drivable_area_segment[i] = int(math.ceil(len(drivable_area_boundary_position[i]) / 10.0))
        drivable_area_segment_split_index[i] = torch.arange(0, len(drivable_area_boundary_position[i]), 10) + num_drivable_area_boundary_processed
        num_drivable_area_boundary_processed += len(drivable_area_boundary_position[i])

    drivable_area_boundary_position = torch.cat(drivable_area_boundary_position, dim=0)
    drivable_area_boundary_length = torch.cat(drivable_area_boundary_length, dim=0)
    drivable_area_boundary_heading = torch.cat(drivable_area_boundary_heading, dim=0)
    drivable_area_boundary_theta = torch.cat(drivable_area_boundary_theta, dim=0)

    drivable_area_segment_split_index.append(torch.tensor((num_drivable_area_boundary_processed,), dtype=torch.long))
    drivable_area_segment_split_index = torch.cat(drivable_area_segment_split_index, dim=0)

    data['drivable_area_boundary']['num_nodes'] = num_drivable_area_boundary.sum().item()
    data['drivable_area_boundary']['position'] = drivable_area_boundary_position
    data['drivable_area_boundary']['length'] = drivable_area_boundary_length
    data['drivable_area_boundary']['heading'] = drivable_area_boundary_heading
    data['drivable_area_boundary']['theta'] = drivable_area_boundary_theta

    # polygon
    num_lanes = len(lanes)
    num_crosswalks = len(crosswalks)
    num_drivable_area_segment = num_drivable_area_segment.sum().item()
    num_static_objects = len(static_objects)
    num_polygons = num_lanes + num_crosswalks + num_drivable_area_segment + num_static_objects

    polygon_position = torch.zeros(num_polygons, 2, dtype=torch.float32)
    polygon_heading = torch.zeros(num_polygons, dtype=torch.float32)
    polygon_heading_valid_mask = torch.zeros(num_polygons, dtype=torch.bool)
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_speed_limit = torch.zeros(num_polygons, dtype=torch.float32)
    polygon_speed_limit_valid_mask = torch.zeros(num_polygons, dtype=torch.bool)
    polygon_traffic_light = torch.ones(num_polygons, dtype=torch.uint8) * (len(_traffic_light_types) - 1)
    polygon_on_route_mask = torch.zeros(num_polygons, dtype=torch.bool)

    polygon_static_object_boundary: List[Polygon] = [None] * num_static_objects

    num_polylines = torch.zeros(num_polygons, dtype=torch.long)
    polyline_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    polyline_heading: List[Optional[torch.Tensor]] = [None] * num_polygons
    polyline_length: List[Optional[torch.Tensor]] = [None] * num_polygons
    polyline_type: List[Optional[torch.Tensor]] = [None] * num_polygons

    # route
    route_roadblocks = [roadblock for roadblock in roadblocks if roadblock.id in route_roadblock_ids]
    route_lane_ids = set()
    for route_roadblock in route_roadblocks:
        route_lane_ids.update([lane.id for lane in route_roadblock.interior_edges])
    route_lane_ids = list(route_lane_ids)
    lane_ids = [lane.id for lane in lanes]
    route_lane_index = get_index_of_A_in_B(route_lane_ids, lane_ids)
    polygon_on_route_mask[route_lane_index] = True

    # lanes
    for i, lane in enumerate(lanes):
        center_polyline = (torch.stack([torch.tensor([point.x, point.y], dtype=torch.float64) for point in lane.baseline_path.discrete_path], dim=0) - center_position).to(torch.float32)
        if len(center_polyline) > 10:
            center_polyline = interpolate_points(center_polyline, 10, 'linear')

        num_polylines[i] = len(center_polyline) - 1
        polyline_position[i] = center_polyline[:-1]
        polyline_length[i], polyline_heading[i] = compute_angles_lengths_2D(center_polyline[1:] - center_polyline[:-1])
        polyline_type[i] = torch.ones_like(polyline_length[i], dtype=torch.uint8) * torch.tensor(_polyline_types.index('CENTERLINE'), dtype=torch.uint8)

        center_index = num_polylines[i] // 2
        polygon_position[i] = polyline_position[i][center_index]
        polygon_heading[i] = polyline_heading[i][center_index]
        polygon_heading_valid_mask[i] = True
        polygon_type[i] = torch.tensor(_polygon_types.index('LANE'), dtype=torch.uint8)
        if lane.speed_limit_mps is not None:
            polygon_speed_limit[i] = lane.speed_limit_mps
            polygon_speed_limit_valid_mask[i] = True
    
    # crosswalk
    for i, crosswalk in enumerate(crosswalks):
        boundary = (torch.tensor(np.array(crosswalk.polygon.exterior.coords), dtype=torch.float64) - center_position).to(torch.float32)
        if len(boundary) > 30:
            boundary = interpolate_points(boundary, 30, 'linear')

        num_polylines[num_lanes + i] = len(boundary) - 1
        polyline_position[num_lanes + i] = boundary[:-1]
        polyline_length[num_lanes + i], polyline_heading[num_lanes + i] = compute_angles_lengths_2D(boundary[1:] - boundary[:-1])
        polyline_type[num_lanes + i] = torch.ones_like(polyline_length[num_lanes + i], dtype=torch.uint8) * torch.tensor(_polyline_types.index('BOUNDARY'), dtype=torch.uint8)

        polygon_position[num_lanes + i] = torch.mean(boundary, dim=0)
        polygon_type[num_lanes + i] = torch.tensor(_polygon_types.index('CROSSWALK'), dtype=torch.uint8)

    # drivable area segment
    for i in range(num_drivable_area_segment):
        num_polylines[num_lanes + num_crosswalks + i] = drivable_area_segment_split_index[i+1] - drivable_area_segment_split_index[i]
        polyline_position[num_lanes + num_crosswalks + i] = drivable_area_boundary_position[drivable_area_segment_split_index[i]:drivable_area_segment_split_index[i+1]]
        polyline_length[num_lanes + num_crosswalks + i] = drivable_area_boundary_length[drivable_area_segment_split_index[i]:drivable_area_segment_split_index[i+1]]
        polyline_heading[num_lanes + num_crosswalks + i] = drivable_area_boundary_heading[drivable_area_segment_split_index[i]:drivable_area_segment_split_index[i+1]]
        polyline_type[num_lanes + num_crosswalks + i] = torch.ones_like(polyline_length[num_lanes + num_crosswalks + i], dtype=torch.uint8) * torch.tensor(_polyline_types.index('BOUNDARY'), dtype=torch.uint8)

        center_index = num_polylines[num_lanes + num_crosswalks + i] // 2
        polygon_position[num_lanes + num_crosswalks + i] = polyline_position[num_lanes + num_crosswalks + i][center_index]
        polygon_heading[num_lanes + num_crosswalks + i] = wrap_angle(polyline_heading[num_lanes + num_crosswalks + i][center_index] - math.pi/2)
        polygon_heading_valid_mask[num_lanes + num_crosswalks + i] = True
        polygon_type[num_lanes + num_crosswalks + i] = torch.tensor(_polygon_types.index('DRIVABLE_AREA_SEGMENT'), dtype=torch.uint8)

    # static objects
    for i, static_object in enumerate(static_objects):
        boundary = (torch.tensor(np.array(static_object.box.geometry.exterior.coords), dtype=torch.float64) - center_position).to(torch.float32)

        boundary_polygon = Polygon(boundary.cpu().numpy())
        polygon_static_object_boundary[i] = boundary_polygon

        num_polylines[num_lanes + num_crosswalks + num_drivable_area_segment + i] = len(boundary) - 1
        polyline_position[num_lanes + num_crosswalks + num_drivable_area_segment + i] = boundary[:-1]
        polyline_length[num_lanes + num_crosswalks + num_drivable_area_segment + i], polyline_heading[num_lanes + num_crosswalks + num_drivable_area_segment + i] = compute_angles_lengths_2D(boundary[1:] - boundary[:-1])
        polyline_type[num_lanes + num_crosswalks + num_drivable_area_segment + i] = torch.ones_like(polyline_length[num_lanes + num_crosswalks + num_drivable_area_segment + i], dtype=torch.uint8) * torch.tensor(_polyline_types.index('BOUNDARY'), dtype=torch.uint8)

        polygon_position[num_lanes + num_crosswalks + num_drivable_area_segment + i] = torch.mean(boundary, dim=0)
        polygon_type[num_lanes + num_crosswalks + num_drivable_area_segment + i] = torch.tensor(_polygon_types.index('STATIC_OBJECT'), dtype=torch.uint8)
        
    # traffic light
    traffic_lights = list(traffic_lights)
    for traffic_light in traffic_lights:
        lane_id = str(traffic_light.lane_connector_id)
        if lane_id in lane_ids:
            index = lane_ids.index(lane_id)
            polygon_traffic_light[index] = torch.tensor(traffic_light.status, dtype=torch.uint8)

    # filter polygons
    if num_lanes > max_lanes:
        lane_to_center_distance = torch.norm(polygon_position[:num_lanes], dim=-1)
        _, lane_to_center_index = torch.topk(lane_to_center_distance, max_lanes, largest=False)
    else:
        lane_to_center_index = torch.arange(num_lanes)
    if num_crosswalks > max_crosswalks:
        crosswalk_to_center_distance = torch.norm(polygon_position[num_lanes:num_lanes+num_crosswalks], dim=-1)
        _, crosswalk_to_center_index = torch.topk(crosswalk_to_center_distance, max_crosswalks, largest=False)
    else:
        crosswalk_to_center_index = torch.arange(num_crosswalks)
    if num_drivable_area_segment > max_drivable_area_segments:
        drivable_area_segment_to_center_distance = torch.norm(polygon_position[num_lanes+num_crosswalks:num_lanes+num_crosswalks+num_drivable_area_segment], dim=-1)
        _, drivable_area_segment_to_center_index = torch.topk(drivable_area_segment_to_center_distance, max_drivable_area_segments, largest=False)
    else:
        drivable_area_segment_to_center_index = torch.arange(num_drivable_area_segment)
    if static_object_tokens is not None:
        static_object_to_center_index = torch.tensor([i for i in range(num_static_objects) if static_objects[i].track_token in static_object_tokens],dtype=torch.long)
    elif num_static_objects > max_static_objects:
        static_object_to_center_distance = torch.norm(polygon_position[num_lanes+num_crosswalks+num_drivable_area_segment:], dim=-1)
        _, static_object_to_center_index = torch.topk(static_object_to_center_distance, max_static_objects, largest=False)
        polygon_static_object_boundary = [polygon_static_object_boundary[i] for i in static_object_to_center_index]
    else:
        static_object_to_center_index = torch.arange(num_static_objects)
    index = torch.cat([lane_to_center_index, crosswalk_to_center_index + num_lanes, drivable_area_segment_to_center_index + num_lanes + num_crosswalks, static_object_to_center_index + num_lanes + num_crosswalks + num_drivable_area_segment], dim=0)

    num_polygons = len(index)
    polygon_position = polygon_position[index]
    polygon_heading = polygon_heading[index]
    polygon_heading_valid_mask = polygon_heading_valid_mask[index]
    polygon_type = polygon_type[index]
    polygon_speed_limit = polygon_speed_limit[index]
    polygon_speed_limit_valid_mask = polygon_speed_limit_valid_mask[index]
    polygon_traffic_light = polygon_traffic_light[index]
    polygon_on_route_mask = polygon_on_route_mask[index]

    num_polylines = num_polylines[index]
    polyline_position = [polyline_position[i] for i in index]
    polyline_length = [polyline_length[i] for i in index]
    polyline_heading = [polyline_heading[i] for i in index]
    polyline_type = [polyline_type[i] for i in index]

    data['polygon']['num_nodes'] = num_polygons
    data['polygon']['position'] = polygon_position
    data['polygon']['heading'] = polygon_heading
    data['polygon']['heading_valid_mask'] = polygon_heading_valid_mask
    data['polygon']['type'] = polygon_type
    data['polygon']['speed_limit'] = polygon_speed_limit
    data['polygon']['speed_limit_valid_mask'] = polygon_speed_limit_valid_mask
    data['polygon']['on_route_mask'] = polygon_on_route_mask
    data['polygon']['traffic_light'] = polygon_traffic_light

    data['polyline']['num_nodes'] = num_polylines.sum().item()
    data['polyline']['position'] = torch.cat(polyline_position, dim=0)
    data['polyline']['length'] = torch.cat(polyline_length, dim=0)
    data['polyline']['heading'] = torch.cat(polyline_heading, dim=0)
    data['polyline']['type'] = torch.cat(polyline_type, dim=0)

    polyline_to_polygon_edge_index = torch.stack([torch.arange(num_polylines.sum().item(), dtype=torch.long), torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_polylines)], dim=0)
    data['polyline', 'polygon']['polyline_to_polygon_edge_index'] = polyline_to_polygon_edge_index

    # static object area
    static_object_area = unary_union(polygon_static_object_boundary).buffer(0).simplify(0.1, preserve_topology=True).buffer(0)
    if static_object_area.is_empty:
        data['static_object_area_boundary']['num_nodes'] = 0
        data['static_object_area_boundary']['position'] = torch.zeros(0, 2, dtype=torch.float32)
        data['static_object_area_boundary']['length'] = torch.zeros(0, dtype=torch.float32)
        data['static_object_area_boundary']['heading'] = torch.zeros(0, dtype=torch.float32)
        data['static_object_area_boundary']['theta'] = torch.zeros(0, dtype=torch.float32)
    else:
        if static_object_area.geom_type == 'Polygon':
            static_object_area_coords = [static_object_area.exterior.coords[::-1]]
        elif static_object_area.geom_type == 'MultiPolygon':
            static_object_area_coords = [polygon.exterior.coords[::-1] for polygon in static_object_area.geoms]

        num_static_object_area = len(static_object_area_coords)
        num_static_object_area_boundary = torch.zeros(num_static_object_area, dtype=torch.long)
        static_object_area_boundary_position = [None] * num_static_object_area
        static_object_area_boundary_length = [None] * num_static_object_area
        static_object_area_boundary_heading = [None] * num_static_object_area
        static_object_area_boundary_theta = [None] * num_static_object_area
        for i in range(num_static_object_area):
            static_object_area_boundary = torch.tensor([point for point in static_object_area_coords[i]], dtype=torch.float32)
            static_object_area_boundary_position[i] = static_object_area_boundary[:-1]
            static_object_area_boundary_length[i], static_object_area_boundary_heading[i] = compute_angles_lengths_2D(static_object_area_boundary[1:] - static_object_area_boundary[:-1])
            static_object_area_boundary_theta[i] = wrap_angle(math.pi + torch.roll(static_object_area_boundary_heading[i], 1) - static_object_area_boundary_heading[i], min_val=0.0, max_val=2*math.pi)
            num_static_object_area_boundary[i] = len(static_object_area_boundary) - 1

        data['static_object_area_boundary']['num_nodes'] = num_static_object_area_boundary.sum().item()
        data['static_object_area_boundary']['position'] = torch.cat(static_object_area_boundary_position, dim=0)
        data['static_object_area_boundary']['length'] = torch.cat(static_object_area_boundary_length, dim=0)
        data['static_object_area_boundary']['heading'] = torch.cat(static_object_area_boundary_heading, dim=0)
        data['static_object_area_boundary']['theta'] = torch.cat(static_object_area_boundary_theta, dim=0)

    # lane edge
    polygon_left_edge_index = []
    polygon_right_edge_index = []
    polygon_incoming_edge_index = []
    polygon_outgoing_edge_index = []

    lanes = [lanes[i] for i in lane_to_center_index]
    lane_ids = [lane_ids[i] for i in lane_to_center_index]
    for lane in lanes:
        polygon_index = lane_ids.index(lane.id)
        polygon_left_polygon, polygon_right_polygon = lane.adjacent_edges
        polygon_left_polygon_id = polygon_left_polygon.id if polygon_left_polygon is not None else None
        polygon_right_polygon_id = polygon_right_polygon.id if polygon_right_polygon is not None else None
        polygon_left_polygon_index = get_index_of_a_in_B(polygon_left_polygon_id, lane_ids)
        polygon_right_polygon_index = get_index_of_a_in_B(polygon_right_polygon_id, lane_ids)
        if len(polygon_left_polygon_index) > 0:
            edge_index = torch.stack([torch.tensor(polygon_left_polygon_index, dtype=torch.long), torch.full((len(polygon_left_polygon_index),), polygon_index, dtype=torch.long)], dim=0)
            polygon_left_edge_index.append(edge_index)
        if len(polygon_right_polygon_index) > 0:
            edge_index = torch.stack([torch.tensor(polygon_right_polygon_index, dtype=torch.long), torch.full((len(polygon_right_polygon_index),), polygon_index, dtype=torch.long)], dim=0)
            polygon_right_edge_index.append(edge_index)
        polygon_incoming_polygons, polygon_outgoing_polygons = lane.incoming_edges, lane.outgoing_edges
        polygon_incoming_polygon_ids = [polygon.id for polygon in polygon_incoming_polygons]
        polygon_outgoing_polygon_ids = [polygon.id for polygon in polygon_outgoing_polygons]
        polygon_incoming_polygon_index = get_index_of_A_in_B(polygon_incoming_polygon_ids, lane_ids)
        polygon_outgoing_polygon_index = get_index_of_A_in_B(polygon_outgoing_polygon_ids, lane_ids)
        if len(polygon_incoming_polygon_index) > 0:
            edge_index = torch.stack([torch.tensor(polygon_incoming_polygon_index, dtype=torch.long), torch.full((len(polygon_incoming_polygon_index),), polygon_index, dtype=torch.long)], dim=0)
            polygon_incoming_edge_index.append(edge_index)
        if len(polygon_outgoing_polygon_index) > 0:
            edge_index = torch.stack([torch.tensor(polygon_outgoing_polygon_index, dtype=torch.long), torch.full((len(polygon_outgoing_polygon_index),), polygon_index, dtype=torch.long)], dim=0)
            polygon_outgoing_edge_index.append(edge_index)

    if len(polygon_left_edge_index) != 0:
        polygon_left_edge_index = torch.cat(polygon_left_edge_index, dim=-1)
    else:
        polygon_left_edge_index = torch.tensor([[], []], dtype=torch.long)
    if len(polygon_right_edge_index) != 0:
        polygon_right_edge_index = torch.cat(polygon_right_edge_index, dim=-1)
    else:
        polygon_right_edge_index = torch.tensor([[], []], dtype=torch.long)
    if len(polygon_incoming_edge_index) != 0:
        polygon_incoming_edge_index = torch.cat(polygon_incoming_edge_index, dim=-1)
    else:
        polygon_incoming_edge_index = torch.tensor([[], []], dtype=torch.long)
    if len(polygon_outgoing_edge_index) != 0:
        polygon_outgoing_edge_index = torch.cat(polygon_outgoing_edge_index, dim=-1)
    else:
        polygon_outgoing_edge_index = torch.tensor([[], []], dtype=torch.long)
    
    data['polygon', 'polygon']['left_edge_index'] = polygon_left_edge_index
    data['polygon', 'polygon']['right_edge_index'] = polygon_right_edge_index
    data['polygon', 'polygon']['incoming_edge_index'] = polygon_incoming_edge_index
    data['polygon', 'polygon']['outgoing_edge_index'] = polygon_outgoing_edge_index

    if is_simulation:
        return data, center_position
    else:
        return data