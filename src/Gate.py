import torch
import torch.nn as nn

class Gate(nn.Module):
    def __init__(self, user_dim, recency_dim, content_dim, dnn_hidden_dims=[64,32]):
        super(Gate, self).__init__()
        self.input_dim = user_dim + recency_dim + content_dim

        mlp_list = []
        last_dim = self.input_dim
        for dim in dnn_hidden_dims:
            mlp_list.append(nn.Linear(last_dim, dim))
            mlp_list.append(nn.ReLU())
            mlp_list.append(nn.Dropout(0.1))
            last_dim = dim

        self.mlp = nn.Sequential(*mlp_list, nn.Linear(dnn_hidden_dims[-1], 1), nn.Sigmoid())

    def forward(self, user_emb, recency_emb, content_emb):
        assert user_emb.dim() == recency_emb.dim() == content_emb.dim(), "user_dim, recency_dim, content_dim维度不一致"

        concat_tensor = torch.cat([user_emb, recency_emb, content_emb], dim=-1)
        
        return self.mlp(concat_tensor)

        