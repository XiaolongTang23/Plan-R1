import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import transform_point_to_local_coordinate
from utils import generate_reachable_matrix

class MapEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_hops:int, 
                 num_heads: int,
                 dropout: float) -> None:
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self._polygon_types = ['LANE', 'CROSSWALK', 'DRIVABLE_AREA_SEGMENT','STATIC_OBJECT']
        self._polygon_embs = nn.Embedding(len(self._polygon_types), hidden_dim)
        # self._polyline_types = ['CENTERLINE', 'BOUNDARY']
        # self._polyline_embs = nn.Embedding(len(self._polyline_types), hidden_dim)
        self._traffic_light_types = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN', 'NONE']
        self._traffic_light_type_embs = nn.Embedding(len(self._traffic_light_types), hidden_dim)
        self._route_types = ['YES', 'NO']
        self._route_embs = nn.Embedding(len(self._route_types), hidden_dim)

        self.l_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g_emb_layer = TwoLayerMLP(input_dim=2 , hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2g_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g2g_emb_layer = TwoLayerMLP(input_dim=8, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self._g2g_edge_types = ['LEFT', 'RIGHT', 'INCOMING', 'OUTGOING']

        self.l2g_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.g2g_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True)

        self.apply(init_weights)

    def forward(self, data: Batch) -> torch.Tensor:
        # embedding
        l_length = data['polyline']['length']
        l_embs = self.l_emb_layer(input=l_length.unsqueeze(-1))        #[(m1,...,mb),D]
        # l_embs = l_embs + self._polyline_embs.weight[data['polyline']['type'].long()]        #[(m1,...,mb),D]

        g_speed_limit = data['polygon']['speed_limit']
        g_speed_limit_valid_mask = data['polygon']['speed_limit_valid_mask']
        g_embs = self.g_emb_layer(input=torch.cat([g_speed_limit.unsqueeze(-1), g_speed_limit_valid_mask.unsqueeze(-1)], dim=-1))   #[(M1,...,Mb),D]
        g_embs = g_embs + self._polygon_embs.weight[data['polygon']['type'].long()]        #[(M1,...,Mb),D]

        # edge
        # l2g
        l2g_position_l = data['polyline']['position']                     #[(m1,...,mb),2]
        l2g_heading_l = data['polyline']['heading']                       #[(m1,...,mb)]
        l2g_position_g = data['polygon']['position']                      #[(M1,...,Mb),2]
        l2g_heading_g = data['polygon']['heading']                        #[(M1,...,Mb)]
        l2g_edge_index = data['polyline', 'polygon']['polyline_to_polygon_edge_index']   
        l2g_edge_vector = transform_point_to_local_coordinate(l2g_position_l[l2g_edge_index[0]], l2g_position_g[l2g_edge_index[1]], l2g_heading_g[l2g_edge_index[1]])
        l2g_edge_attr_length, l2g_edge_attr_theta = compute_angles_lengths_2D(l2g_edge_vector)
        l2g_edge_attr_heading = wrap_angle(l2g_heading_l[l2g_edge_index[0]] - l2g_heading_g[l2g_edge_index[1]])
        l2g_edge_attr_input = torch.stack([l2g_edge_attr_length, torch.cos(l2g_edge_attr_theta), torch.sin(l2g_edge_attr_theta), torch.cos(l2g_edge_attr_heading), torch.sin(l2g_edge_attr_heading)], dim=-1)
        l2g_edge_attr_embs = self.l2g_emb_layer(input = l2g_edge_attr_input)

        # g2g
        g2g_position = data['polygon']['position']                         #[(M1,...,Mb),2]
        g2g_heading = data['polygon']['heading']                           #[(M1,...,Mb)]
        g2g_edge_index = []
        g2g_edge_attr_type = []
        g2g_edge_attr_hop = []
        device = g2g_position.device

        g2g_left_edge_index = data['polygon', 'polygon']['left_edge_index']
        num_left_edges = g2g_left_edge_index.size(1)
        g2g_edge_index.append(g2g_left_edge_index)
        g2g_edge_attr_type.append(F.one_hot(torch.full((num_left_edges,), self._g2g_edge_types.index('LEFT')).long(), num_classes=len(self._g2g_edge_types)).float().to(device))
        g2g_edge_attr_hop.append(torch.ones(num_left_edges, device=device))

        g2g_right_edge_index = data['polygon', 'polygon']['right_edge_index']
        num_right_edges = g2g_right_edge_index.size(1)
        g2g_edge_index.append(g2g_right_edge_index)
        g2g_edge_attr_type.append(F.one_hot(torch.full((num_right_edges,), self._g2g_edge_types.index('RIGHT')).long(), num_classes=len(self._g2g_edge_types)).float().to(device))
        g2g_edge_attr_hop.append(torch.ones(num_right_edges, device=device))

        num_polygons = data['polygon']['num_nodes']
        g2g_incoming_edge_index = data['polygon', 'polygon']['incoming_edge_index']
        g2g_incoming_edge_index_all = generate_reachable_matrix(g2g_incoming_edge_index, self.num_hops, num_polygons)
        for i in range(self.num_hops):
            num_edges_now = g2g_incoming_edge_index_all[i].size(1)
            g2g_edge_index.append(g2g_incoming_edge_index_all[i])
            g2g_edge_attr_type.append(F.one_hot(torch.full((num_edges_now,), self._g2g_edge_types.index('INCOMING')).long(), num_classes=len(self._g2g_edge_types)).float().to(device))
            g2g_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=device))

        g2g_outgoing_edge_index = data['polygon', 'polygon']['outgoing_edge_index']
        g2g_outgoing_edge_index_all = generate_reachable_matrix(g2g_outgoing_edge_index, self.num_hops, num_polygons)
        for i in range(self.num_hops):
            num_edges_now = g2g_outgoing_edge_index_all[i].size(1)
            g2g_edge_index.append(g2g_outgoing_edge_index_all[i])
            g2g_edge_attr_type.append(F.one_hot(torch.full((num_edges_now,), self._g2g_edge_types.index('OUTGOING')).long(), num_classes=len(self._g2g_edge_types)).float().to(device))
            g2g_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=device))

        g2g_edge_index = torch.cat(g2g_edge_index, dim=1)
        g2g_edge_attr_type = torch.cat(g2g_edge_attr_type, dim=0)
        g2g_edge_attr_hop = torch.cat(g2g_edge_attr_hop, dim=0)
        g2g_edge_vector = transform_point_to_local_coordinate(g2g_position[g2g_edge_index[0]], g2g_position[g2g_edge_index[1]], g2g_heading[g2g_edge_index[1]])
        g2g_edge_attr_length, g2g_edge_attr_theta = compute_angles_lengths_2D(g2g_edge_vector)
        g2g_edge_attr_heading = wrap_angle(g2g_heading[g2g_edge_index[0]] - g2g_heading[g2g_edge_index[1]])
        g2g_edge_attr_input = torch.cat([g2g_edge_attr_length.unsqueeze(-1), g2g_edge_attr_theta.unsqueeze(-1), g2g_edge_attr_heading.unsqueeze(-1), g2g_edge_attr_type, g2g_edge_attr_hop.unsqueeze(-1)], dim=-1)
        g2g_edge_attr_embs = self.g2g_emb_layer(input=g2g_edge_attr_input)

        # attention
        # l2g
        g_embs = self.l2g_attn_layer(x = [l_embs, g_embs], edge_index = l2g_edge_index, edge_attr = l2g_edge_attr_embs)         #[(M1,...,Mb),D]

        # g2g
        g_embs = self.g2g_attn_layer(x = g_embs, edge_index = g2g_edge_index, edge_attr = g2g_edge_attr_embs)                   #[(M1,...,Mb),D]

        # add traffic light
        g_embs = g_embs + self._traffic_light_type_embs.weight[data['polygon']['traffic_light'].long()]        #[(M1,...,Mb),D]

        # add route
        g_embs = g_embs + self._route_embs.weight[data['polygon']['on_route_mask'].long()]        #[(M1,...,Mb),D]

        return g_embs