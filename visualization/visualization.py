import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from utils import compute_corner_positions

def visualization(data:Batch, pred_position:torch.tensor=None, pred_heading=None, iteration:int=None, num_historical_steps:int=20) -> None:
    batch_size = len(data['agent']['ptr']) - 1

    agent_batch = data['agent']['batch']
    agent_position = data['agent']['position'][:, num_historical_steps]
    agent_heading = data['agent']['heading'][:, num_historical_steps]
    agent_box = data['agent']['box']
    agent_type = data['agent']['type']
    agent_identity = data['agent']['identity']
    agent_corner_position = compute_corner_positions(agent_position, agent_heading, agent_box)
    agent_corner_position = torch.roll(agent_corner_position, 1, 1)
    if pred_position is not None and pred_heading is not None:
        pred_corner_position = compute_corner_positions(pred_position, pred_heading, agent_box.unsqueeze(-1))

    polygon_batch = data['polygon']['batch']
    polygon_type = data['polygon']['type']
    polygon_on_route_mask = data['polygon']['on_route_mask']
    polygon_traffic_light = data['polygon']['traffic_light']
    
    polyline_position = data['polyline']['position']
    polyline_heading = data['polyline']['heading']
    polyline_length = data['polyline']['length']

    polyline_to_polygon_edge_index = data['polyline', 'polygon']['polyline_to_polygon_edge_index']
    
    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(20, 20))

        # map
        polygon_indices = (polygon_batch == i).nonzero(as_tuple=False).squeeze()
        for index in polygon_indices:
            # lane
            if polygon_type[index] == 0:
                # traffic light
                if polygon_traffic_light[index] == 0:
                    color = 'green'
                elif polygon_traffic_light[index] == 1:
                    color = 'yellow'
                elif polygon_traffic_light[index] == 2:
                    color = 'red'
                else:
                    color = 'grey'
                # route
                if polygon_on_route_mask[index]:
                    linestyle = '-'
                    linewidth = 3
                else:
                    linestyle = '--'
                    linewidth = 1
                mask = polyline_to_polygon_edge_index[1] == index
                polyline_position_temp = polyline_position[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                polyline_heading_temp = polyline_heading[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                polyline_length_temp = polyline_length[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                for j in range(len(polyline_position_temp)):
                    ax.plot(
                        [polyline_position_temp[j, 0], polyline_position_temp[j, 0] + polyline_length_temp[j] * np.cos(polyline_heading_temp[j])],
                        [polyline_position_temp[j, 1], polyline_position_temp[j, 1] + polyline_length_temp[j] * np.sin(polyline_heading_temp[j])],
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        zorder=0
                    )
            # crosswalk
            elif polygon_type[index] == 1:
                mask = polyline_to_polygon_edge_index[1] == index
                polyline_position_temp = polyline_position[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                ax.add_patch(plt.Polygon(polyline_position_temp, fill=True, edgecolor='gray', facecolor='gray', alpha=0.2, zorder=0))
            # drivable area segment
            elif polygon_type[index] == 2:
                mask = polyline_to_polygon_edge_index[1] == index
                polyline_position_temp = polyline_position[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                polyline_heading_temp = polyline_heading[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                polyline_length_temp = polyline_length[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                for j in range(len(polyline_position_temp)):
                    ax.plot(
                        [polyline_position_temp[j, 0], polyline_position_temp[j, 0] + polyline_length_temp[j] * np.cos(polyline_heading_temp[j])],
                        [polyline_position_temp[j, 1], polyline_position_temp[j, 1] + polyline_length_temp[j] * np.sin(polyline_heading_temp[j])],
                        color='black',
                        linewidth=1,
                        zorder=10
                    )
            # static obstacle
            elif polygon_type[index] == 3:
                mask = polyline_to_polygon_edge_index[1] == index
                polyline_position_temp = polyline_position[polyline_to_polygon_edge_index[0][mask]].cpu().numpy()
                ax.add_patch(plt.Polygon(polyline_position_temp, fill=True, edgecolor='c', facecolor='c', zorder=10))
            else:
                raise ValueError(f"Unknown polygon type: {polygon_type[index]}") 

        # agent
        agent_mask = agent_batch == i
        agent_position_i = agent_position[agent_mask].cpu().numpy()
        agent_type_i = agent_type[agent_mask].cpu().numpy()
        agent_identity_i = agent_identity[agent_mask].cpu().numpy()
        agent_corner_position_i = agent_corner_position[agent_mask].cpu().numpy()
        pred_corner_position_i = pred_corner_position[agent_mask].cpu().numpy() if pred_position is not None and pred_heading is not None else None
        for j in range(len(agent_position_i)-1, -1, -1):
            if agent_type_i[j] == 0:
                color = 'cornflowerblue'
            elif agent_type_i[j] == 1:
                color = 'violet'
            elif agent_type_i[j] == 2:
                color = 'salmon'
            else:
                raise ValueError(f"Unknown agent type: {agent_type_i[j]}")
            if agent_identity_i[j] == 0:
                color = 'lightgreen'
            if j == 0:
                zorder = 3
            else:
                zorder = 2
        
            if pred_corner_position_i is not None:
                for k in range(pred_corner_position_i.shape[1]-1, -1, -1):
                    ax.add_patch(plt.Polygon(pred_corner_position_i[j, k], fill=True, edgecolor=None, facecolor=color, alpha=0.7, zorder=2, linewidth=1))

            ax.plot(agent_corner_position_i[j][:, 0], agent_corner_position_i[j][:, 1], color='black', linewidth=3, zorder=zorder)    
            ax.add_patch(plt.Polygon(agent_corner_position_i[j], fill=True, edgecolor=None, facecolor=color, alpha=0.7, zorder=1, linewidth=1))
                                        
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)

        # save figure
        os.makedirs('visualization/results', exist_ok=True)
        if iteration is not None:
            plt.savefig(f'visualization/results/{iteration}.png', bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(f'visualization/results/{data["scenario_name"][i]}_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
