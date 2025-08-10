import torch
from torch_geometric.data import HeteroData, Batch
import numpy as np
import numpy.typing as npt
from typing import Deque
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

import os
from model import PlanR1
from datasets import get_features
from visualization import visualization

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PlanR1Planner(AbstractPlanner):
    def __init__(self, model_path, token_builder, model_mode, visualization):
        self.future_horizon = 8.0
        self.step_interval = 0.5
        self.model_path = model_path
        self.model_mode = model_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = token_builder
        self.visualization = visualization

    def name(self) -> str:
        return "Plan-R1 Planner"

    def observation_type(self):
        return DetectionsTracks
    
    def initialize(self, initialization: PlannerInitialization):
        self.map_api = initialization.map_api
        self.mission_goal = initialization.mission_goal
        self.route_roadblock_ids = initialization.route_roadblock_ids
        self.initialize_model()

    def initialize_model(self):
        token_dict_path = os.path.join(os.environ["PROJECT_ROOT"], "tokens/tokens_1024.pt")
        self.model = PlanR1.load_from_checkpoint(self.model_path, token_dict_path=token_dict_path).to(self.device)
        self.model.eval()

    def compute_planner_trajectory(self, current_input: PlannerInput):
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_lights = current_input.traffic_light_data
            
        # process data
        data, center_position = get_features(list(history.ego_state_buffer)[-21:], list(history.observation_buffer)[-21:], self.map_api, traffic_lights, self.route_roadblock_ids, radius=120, is_simulation=True)
        data = Batch.from_data_list([self.transform(HeteroData(data))]).to(self.device)
        
        # generate output
        with torch.no_grad():
            if self.model_mode == 'plan':
                _, traj_pos, traj_yaw, _ = self.model.plan_inference(data)
            elif self.model_mode == 'pred':
                _, traj_pos, traj_yaw, _ = self.model.pred_inference(data)
            traj_output = torch.cat([traj_pos, traj_yaw.unsqueeze(-1)], dim=-1)
                
            if self.visualization:
                visualization(data, traj_pos, iteration=iteration)

            traj_output = traj_output[0]
            traj_output = traj_output.to(torch.float64)      
            traj_output[:, :2] = traj_output[:, :2] + center_position.to(self.device)
            traj_output = traj_output[:, :3].cpu().numpy()

        states = self.global_trajectory_to_states(traj_output, history.ego_states, len(traj_output)*self.step_interval, self.step_interval)
        trajectory = InterpolatedTrajectory(states)
        return trajectory
    
    def global_trajectory_to_states(
        self,
        global_trajectory: npt.NDArray[np.float32],
        ego_history: Deque[EgoState],
        future_horizon: float,
        step_interval: float,
        include_ego_state: bool = True,
    ):
        ego_state = ego_history[-1]
        timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
        global_states = [StateSE2.deserialize(pose) for pose in global_trajectory]

        velocities, accelerations = _get_velocity_and_acceleration(global_states, ego_history, timesteps)
        agent_states = [_se2_vel_acc_to_ego_state(state, velocity, acceleration, timestep, ego_state.car_footprint.vehicle_parameters)
                        for state, velocity, acceleration, timestep in zip(global_states, velocities, accelerations, timesteps)]

        if include_ego_state:
            agent_states.insert(0, ego_state)

        return agent_states