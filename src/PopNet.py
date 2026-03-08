import torch
import torch.nn as nn

class CrossLayer(nn.Module):
    def __init__(self, layer_num, embedding_dim):
        """
        :param layer_num: 交叉层的层数 (通常设为 2 到 3 层)
        :param input_dim: 输入特征的总维度 (比如时间和内容拼接后是 128 维)
        """
        super(CrossLayer, self).__init__()
        self.layer_num = layer_num
        self.cross_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(layer_num)
            ])

    def forward(self, x0):
        """
        :param x0: 原始输入 [Batch, input_dim]
        """
        x_l = x0
        for i in range(self.layer_num):
            x_l = self.cross_layers[i](x_l) * x0 + x_l

        return x_l


class PopNet(nn.Module):
    """
    DCN架构
    """
    def __init__(self, recency_dim, content_dim, cross_layer_num=2, dnn_hidden_dims=[512,256,128]):
        """
        经典并行 DCN 架构
        :param recency_dim: 时间特征维度
        :param content_dim: 内容特征维度
        :param cross_layer_num: 交叉网络的层数
        :param dnn_hidden_dims: 深度网络的隐藏层维度列表
        """
        super(PopNet, self).__init__()
        self.input_dim = recency_dim + content_dim
        # 交叉网络
        self.cross_net = CrossLayer(layer_num=cross_layer_num, embedding_dim=self.input_dim)

        # 全连接网络
        layers = []
        prev_dim = self.input_dim
        for dim in dnn_hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = dim

        self.mlp = nn.Sequential(*layers)

        # 整合网络
        concat_dim = self.input_dim + dnn_hidden_dims[-1]
        self.final_linear = nn.Linear(concat_dim, 1)

    def forward(self, recency_emb, content_emb):
        assert recency_emb.dim() == content_emb.dim(), "recency_emb和content_emb维度不同"
        x0 = torch.cat([recency_emb, content_emb], dim=-1)
        # 分别计算交叉网络和全连接网络的输出结果
        cross_out = self.cross_net(x0)
        mlp_out = self.mlp(x0)

        # 将两个网络的输出结果concat后输入到整合网络
        final_input = torch.cat([cross_out, mlp_out], dim=-1)
        
        return self.final_linear(final_input)

        