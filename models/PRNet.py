import torch
import torch.nn as nn

class LatentChannelRecalibration(nn.Module):

    def __init__(self, channel_num, configs):
        super(LatentChannelRecalibration, self).__init__()
        
        mid_channel = max(1, channel_num // configs.reduction)
        
        if configs.gate_activation == 'relu':
            activation = nn.ReLU()
        elif configs.gate_activation == 'gelu':
            activation = nn.GELU()
        else:
            activation = nn.ReLU()
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(channel_num * 2, mid_channel),
            activation,
            nn.Linear(mid_channel, channel_num)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        std_pool = torch.std(x, dim=-1) # [B, C]

        energy_pool = torch.mean(torch.abs(x), dim=-1) # [B, C]
        
        channel_stat = torch.cat([std_pool, energy_pool], dim=-1) # [B, 2C]
        
        scores = self.gate_mlp(channel_stat) # [B, C]
        scores = scores.unsqueeze(-1) # [B, C, 1]
        
        scale = 1.0 + self.tanh(scores)
        
        out = x * scale
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        print(">>>> Using Model: PRNet (Full Version) <<<<")

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.dropout_rate = configs.dropout
        self.use_revin = getattr(configs, 'use_revin', True)

        self.input_proj_x = nn.Linear(self.seq_len, self.d_model)

        self.learnable_cycle = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
        self.input_proj_q = nn.Linear(self.seq_len, self.d_model)

        self.channel_recalibration = LatentChannelRecalibration(
            channel_num=self.enc_in,
            configs=configs
        )

        self.fusion_gating = nn.Linear(2 * self.d_model, self.d_model)
        self.sigmoid = nn.Sigmoid()

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1) 
        
        x_emb = self.input_proj_x(x_input) # [B, C, D]
        
        x_emb = self.channel_recalibration(x_emb)

        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=x.device).view(1, -1)) % self.cycle_len
        q_raw = self.learnable_cycle[gather_index].permute(0, 2, 1) 
        q_emb = self.input_proj_q(q_raw) 

        cat_features = torch.cat([x_emb, q_emb], dim=-1)
        gate_weights = self.sigmoid(self.fusion_gating(cat_features))
        fused_emb = gate_weights * x_emb + (1 - gate_weights) * q_emb 

        output = self.output_proj(fused_emb).permute(0, 2, 1) 

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
