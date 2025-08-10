import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math
from copy import deepcopy
import os

from layers import TwoLayerMLP
from modules import Backbone, MapEncoder
from metrics import minJointADE, minJointFDE, TokenClsAcc, CumulativeReward
from rewards import AgentCollisionReward, ObstacleCollisionReward, ComfortReward, ProgressReward, SpeedLimitReward, OnRoadReward, TTCReward
from visualization import visualization
from utils import sample_with_top_k_top_p, move_dict_to_device, transform_point_to_global_coordinate, wrap_angle

class PlanR1(pl.LightningModule):
    def __init__(self,
                 mode: str,
                 token_dict_path: str,
                 num_tokens: int = 1024,
                 interval: int = 5,
                 hidden_dim: int = 128,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 agent_radius: float = 60,
                 polygon_radius: float = 30,
                 num_attn_layers: int = 6,
                 pred_top_k: int = 1,
                 plan_top_k: int = 1,
                 rollout_top_k: int = 50,
                 num_samples: int = 4,
                 beta: float = 0.1,
                 num_hops: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0001,
                 warmup_epochs: int = 4,
                 T_max: int = 32,
                 val_visualization: bool = False,
                 comfort_reward_weight: float = 2,
                 ttc_reward_weight: float = 5,
                 speed_limit_reward_weight: float = 4,
                 progress_reward_weight: float = 2) -> None:
        super(PlanR1, self).__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.token_dict = torch.load(token_dict_path)
        self.num_tokens = num_tokens
        self.interval = interval
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_historical_intervals = num_historical_steps // interval
        self.num_future_intervals = num_future_steps // interval
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius
        self.num_attn_layers = num_attn_layers
        self.pred_top_k = pred_top_k
        self.plan_top_k = plan_top_k
        self.rollout_top_k = rollout_top_k
        self.num_samples = num_samples
        self.beta = beta
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max

        # vis in validation
        self.val_visualization = val_visualization

        # reward weights
        self.comfort_reward_weight = comfort_reward_weight
        self.ttc_reward_weight = ttc_reward_weight
        self.speed_limit_reward_weight = speed_limit_reward_weight
        self.progress_reward_weight = progress_reward_weight

        # pred model
        self.pred_backbone = Backbone(
            token_dict=self.token_dict,
            num_tokens=num_tokens,
            interval=interval,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            agent_radius=agent_radius,
            polygon_radius=polygon_radius,
            num_attn_layers=num_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pred_map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pred_decoder_head = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_tokens)

        # plan model
        self.plan_backbone = Backbone(
            token_dict=self.token_dict,
            num_tokens=num_tokens,
            interval=interval,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            agent_radius=agent_radius,
            polygon_radius=polygon_radius,
            num_attn_layers=num_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.plan_map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )
        self.plan_decoder_head = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_tokens)

        # metric
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.token_cls_acc = TokenClsAcc()
        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()
        self.reward = CumulativeReward()

        # reward
        self.on_road_reward = OnRoadReward()
        self.agent_collision_reward = AgentCollisionReward()
        self.obstacle_collision_reward = ObstacleCollisionReward()

        self.speed_limit_reward = SpeedLimitReward()
        self.comfort_reward = ComfortReward()
        self.progress_reward = ProgressReward()
        self.ttc_reward = TTCReward()

    def training_step(self, data: Batch) -> None:
        if self.mode == 'pred':
            # pred token and reward
            polygon_embs = self.pred_map_encoder(data=data) 
            feat = self.pred_backbone(data=data, g_embs=polygon_embs)
            logits = self.pred_decoder_head(feat)
            # compute loss
            target = data['agent']['recon_token'].roll(-1,1)
            target_mask = data['agent']['recon_token_mask'].roll(-1,1)
            target_mask[:, -1] = False
            cls_loss = self.cls_loss(logits[target_mask], target[target_mask])
            self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            return cls_loss
        
        elif self.mode == 'plan':
            advantages, pred_logps, plan_logps, rewards, valid_mask = self.rollout(data)

            ratio = (plan_logps - plan_logps.detach()).exp()
            policy_loss = - (ratio * advantages * valid_mask).sum() / valid_mask.sum()

            kl_loss = torch.exp(pred_logps - plan_logps) - (pred_logps - plan_logps) - 1
            kl_loss = (kl_loss * valid_mask).sum() / valid_mask.sum()

            loss = policy_loss + self.beta * kl_loss

            self.log('train_policy_loss', policy_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_kl_loss', kl_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_reward', rewards.mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

            return loss

    def pred_inference(self, data: Batch):
        map_encoder = self.pred_map_encoder
        backbone = self.pred_backbone
        decoder_head = self.pred_decoder_head

        polygon_embs = map_encoder(data=data)
        agent_embs, k_embs_dict = backbone.pre_inference(data=data)

        for _ in range(self.num_future_intervals):
            k_embs_dict, k_embs_step = backbone.inference(data=data, g_embs=polygon_embs, a_embs=agent_embs, k_embs_dict=k_embs_dict)
            logits_step = decoder_head(k_embs_step)
            action_step = sample_with_top_k_top_p(logits_step.unsqueeze(1), top_k=self.pred_top_k).squeeze(1).squeeze(1)

            data = self.transition(data, action_step)

        position = data['agent']['infer_position'][:, self.num_historical_intervals:]
        heading = data['agent']['infer_heading'][:, self.num_historical_intervals:]
        valid_mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:]

        return data, position, heading, valid_mask

    def plan_inference(self, data: Batch):
        ego_index = data['agent']['ptr'][:-1]

        pred_polygon_embs = self.pred_map_encoder(data=data)
        pred_agent_embs, pred_k_embs_dict = self.pred_backbone.pre_inference(data=data)

        plan_polygon_embs = self.plan_map_encoder(data=data)
        plan_agent_embs, plan_k_embs_dict = self.plan_backbone.pre_inference(data=data)

        for _ in range(self.num_future_intervals):
            pred_k_embs_dict, pred_k_embs_step = self.pred_backbone.inference(data=data, g_embs=pred_polygon_embs, a_embs=pred_agent_embs, k_embs_dict=pred_k_embs_dict)
            plan_k_embs_dict, plan_k_embs_step = self.plan_backbone.inference(data=data, g_embs=plan_polygon_embs, a_embs=plan_agent_embs, k_embs_dict=plan_k_embs_dict)

            pred_logits_step = self.pred_decoder_head(pred_k_embs_step)
            plan_logits_step = self.plan_decoder_head(plan_k_embs_step)
            action_step = sample_with_top_k_top_p(pred_logits_step.unsqueeze(1), top_k=self.pred_top_k).squeeze(1).squeeze(1)
            ego_action_step = sample_with_top_k_top_p(plan_logits_step[ego_index].unsqueeze(1), top_k=self.plan_top_k).squeeze(1).squeeze(1)
            action_step[ego_index] = ego_action_step

            data = self.transition(data, action_step)

        position = data['agent']['infer_position'][:, self.num_historical_intervals:]
        heading = data['agent']['infer_heading'][:, self.num_historical_intervals:]
        valid_mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:]

        return data, position, heading, valid_mask

    def validation_step(self, data: Batch, batch_idx: int) -> None:
        if self.mode == 'pred':
            # pred token and reward
            polygon_embs = self.pred_map_encoder(data=data) 
            feat = self.pred_backbone(data=data, g_embs=polygon_embs)
            logits = self.pred_decoder_head(feat)
            # compute loss
            target = data['agent']['recon_token'].roll(-1,1)
            target_mask = data['agent']['recon_token_mask'].roll(-1,1)
            target_mask[:, -1] = 0
            cls_loss = self.cls_loss(logits[target_mask], target[target_mask])
            self.log('val_token_cls_acc', self.token_cls_acc(logits[target_mask], target[target_mask]), prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            # inference
            _, position, heading, valid_mask = self.pred_inference(data)

        elif self.mode == 'plan':
            # inference
            data, position, heading, valid_mask = self.plan_inference(data)
            # compute rewards
            rewards, _, _ = self.reward_fn(data)
            self.reward.update(rewards)
            self.log('val_reward', self.reward, prog_bar=True, on_step=False, on_epoch=True)

        agent_batch = data['agent']['batch']
        agent_pred_traj = unbatch(position.unsqueeze(1), agent_batch)
        agent_target_traj = unbatch(data['agent']['position'][:, self.num_historical_steps+self.interval::self.interval], agent_batch)
        agent_mask = unbatch(data['agent']['visible_mask'][:, self.num_historical_steps+self.interval::self.interval] & valid_mask, agent_batch)

        self.min_joint_ade.update(agent_pred_traj, agent_target_traj, agent_mask)
        self.min_joint_fde.update(agent_pred_traj, agent_target_traj, agent_mask)
        self.log('val_min_joint_ade', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_min_joint_fde', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True)

        if self.val_visualization:
            visualization(data, position)

    def freeze_pred_model(self):
        # eval mode
        self.pred_map_encoder.eval()
        self.pred_backbone.eval()
        self.pred_decoder_head.eval()
        # freeze
        for p in self.pred_map_encoder.parameters():
            p.requires_grad = False
        for p in self.pred_backbone.parameters():
            p.requires_grad = False
        for p in self.pred_decoder_head.parameters():
            p.requires_grad = False

    def on_train_start(self):
        if self.mode == 'plan':
            self.plan_map_encoder.load_state_dict(self.pred_map_encoder.state_dict())
            self.plan_backbone.load_state_dict(self.pred_backbone.state_dict())
            self.plan_decoder_head.load_state_dict(self.pred_decoder_head.state_dict())
            self.freeze_pred_model()

    def transition(self, data, action):
        next_data = data.clone()

        next_data['agent']['infer_token'] = torch.cat([next_data['agent']['infer_token'], action.unsqueeze(1)], dim=1)
        next_data['agent']['infer_token_mask'] = torch.cat([next_data['agent']['infer_token_mask'], next_data['agent']['infer_token_mask'][:, -1:]], dim=1)

        a_type = data['agent']['type']
        vehicle_mask = a_type == 0
        pedestrian_mask = a_type == 1
        bicycle_mask = a_type == 2

        token = torch.zeros(action.size(0), 3, device=action.device)
        self.token_dict = move_dict_to_device(self.token_dict, action.device)
        token[vehicle_mask] = self.token_dict['Vehicle'][action[vehicle_mask]]
        token[pedestrian_mask] = self.token_dict['Pedestrian'][action[pedestrian_mask]]
        token[bicycle_mask] = self.token_dict['Bicycle'][action[bicycle_mask]]
        token_position = transform_point_to_global_coordinate(token[:, :2], next_data['agent']['infer_position'][:, -1], next_data['agent']['infer_heading'][:, -1])
        token_heading = wrap_angle(token[:, 2] + next_data['agent']['infer_heading'][:, -1])

        next_data['agent']['infer_position'] = torch.cat([next_data['agent']['infer_position'], token_position.unsqueeze(1)], dim=1)
        next_data['agent']['infer_heading'] = torch.cat([next_data['agent']['infer_heading'], token_heading.unsqueeze(1)], dim=1)
        next_data['agent']['infer_valid_mask'] = torch.cat([next_data['agent']['infer_valid_mask'], next_data['agent']['infer_valid_mask'][:, -1:]], dim=1)

        return next_data
        
    def rollout(self, data):
        # copy data
        data_list = data.to_data_list()
        data_list_copy = deepcopy(data_list)
        for _ in range(self.num_samples - 1):
            data_list += data_list_copy
        data = Batch.from_data_list(data_list)
        ego_index = data['agent']['ptr'][:-1]

        # initialize
        pred_polygon_embs = self.pred_map_encoder(data=data)
        pred_agent_embs, pred_k_embs_dict = self.pred_backbone.pre_inference(data=data)
        plan_polygon_embs = self.plan_map_encoder(data=data)
        plan_agent_embs, plan_k_embs_dict = self.plan_backbone.pre_inference(data=data)

        # inference
        for step in range(self.num_future_intervals):
            pred_k_embs_dict, pred_k_embs_step = self.pred_backbone.inference(data=data, g_embs=pred_polygon_embs, a_embs=pred_agent_embs, k_embs_dict=pred_k_embs_dict)
            plan_k_embs_dict, plan_k_embs_step = self.plan_backbone.inference(data=data, g_embs=plan_polygon_embs, a_embs=plan_agent_embs, k_embs_dict=plan_k_embs_dict)
            
            pred_logits_step = self.pred_decoder_head(pred_k_embs_step)
            plan_logits_step = self.plan_decoder_head(plan_k_embs_step)

            action_step = sample_with_top_k_top_p(pred_logits_step.unsqueeze(1), top_k=1).squeeze(1).squeeze(1)
            ego_action_step = sample_with_top_k_top_p(plan_logits_step[ego_index].unsqueeze(1), top_k=self.rollout_top_k).squeeze(1).squeeze(1)
            
            pred_dist_step = Categorical(logits=pred_logits_step[ego_index])
            plan_dist_step = Categorical(logits=plan_logits_step[ego_index])
            if step == 0:
                pred_logps = pred_dist_step.log_prob(ego_action_step).unsqueeze(1)
                plan_logps = plan_dist_step.log_prob(ego_action_step).unsqueeze(1)
            else:
                pred_logps = torch.cat([pred_logps, pred_dist_step.log_prob(ego_action_step).unsqueeze(1)], dim=1)
                plan_logps = torch.cat([plan_logps, plan_dist_step.log_prob(ego_action_step).unsqueeze(1)], dim=1)

            action_step[ego_index] = ego_action_step
            data = self.transition(data, action_step)

        rewards, _, valid_mask = self.reward_fn(data)
        advantages = self.compute_ae_outcome_supervision(rewards)

        return advantages, pred_logps, plan_logps, rewards, valid_mask
    
    def compute_ae_outcome_supervision(self, rewards):
        # group computation
        rewards_reshape = rewards.view(self.num_samples, -1)
        rewards_mean = rewards_reshape.mean(dim=0)
        rewards_std = rewards_reshape.std(dim=0)

        advantages = (rewards_reshape - rewards_mean) / (rewards_std + 1e-4)
        advantages = advantages.view(-1).unsqueeze(-1).expand(-1, self.num_future_intervals)

        return advantages

    def reward_fn(self, data):
        agent_collision_done, agent_collision_reward = self.agent_collision_reward(data)
        obstacle_collision_done, obstacle_collision_reward = self.obstacle_collision_reward(data)
        on_road_done, on_road_reward = self.on_road_reward(data)
        done = agent_collision_done | obstacle_collision_done | on_road_done

        valid_mask = (~done).float().cumprod(dim=1).bool()
        valid_mask = torch.cat([torch.ones(done.size(0), 1, device=done.device, dtype=torch.bool), valid_mask[:, :-1]], dim=1)

        comfort_reward = self.comfort_reward(data)
        ttc_reward = self.ttc_reward(data)
        progress_reward = self.progress_reward(data)
        speed_limit_reward = self.speed_limit_reward(data)

        reward = on_road_reward * obstacle_collision_reward * agent_collision_reward * (
                 self.comfort_reward_weight * comfort_reward + 
                 self.ttc_reward_weight * ttc_reward + 
                 self.speed_limit_reward_weight * speed_limit_reward + 
                 self.progress_reward_weight * progress_reward) / (self.comfort_reward_weight + self.ttc_reward_weight + self.speed_limit_reward_weight + self.progress_reward_weight)
        
        return reward, done, valid_mask
    
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]