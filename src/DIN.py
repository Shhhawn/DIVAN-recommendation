import torch
import torch.nn as nn
import torch.nn.functional as F

class DINAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=[128, 64]):
        """
        :param embedding_dim: query和key的embedding维度
        :param hidden_dim: 注意力网络的隐藏层维度
        """
        super(DINAttention, self).__init__()
        list = []
        last_dim = embedding_dim * 4

        for dim in hidden_dim:
            list.append(nn.Linear(last_dim, dim))
            list.append(nn.ReLU())
            list.append(nn.Dropout(0.2))
            last_dim = dim

        self.attention_mlp = nn.Sequential(
            *list,
            nn.Linear(hidden_dim[-1], 1)
        )

    def forward(self, querys, keys, mask=None):
        """
        :param querys: [batch_size, 1, embedding_dim] 候选文章的embedding
        :param keys: [batch_size, seq_len, embedding_dim] 历史文章的embedding
        :param mask: [batch_size, seq_len] 用于屏蔽padding位置
        """
        # [!note] 使用.expand()将query的维度扩充成与keys相同以便于拼接，.expand(a,b,c)表示把三个维度分别扩充成a、b、c，-1表示不变
        input = torch.cat([
            querys.expand(-1, keys.shape[1], -1), 
            keys, 
            querys - keys, 
            querys * keys
        ], dim=-1)

        # attention_score形状(batch, seq_len, 1)
        attention_score = self.attention_mlp(input)

        # 不能写成if mask:
        if mask is not None:
            assert mask.dim() == 2, f"mask应该是2维，现在是{mask.dim()}维"
            # mask形状是(batch, seq_len)，先将mask扩充一维使其与attention_score的形状对齐
            attention_score = attention_score.masked_fill(mask.unsqueeze(-1)==0, -1e4)

        # attention_weights: (batch, seq_len)
        attention_weights = F.softmax(attention_score, dim=1)

        output = (attention_weights * keys).sum(dim=1)
        return output


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embedding_dim = 64

    dummy_query = torch.randn(batch_size, 1, embedding_dim)
    dummy_keys = torch.randn(batch_size, seq_len, embedding_dim)

    dummy_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0]
    ])

    attention_layer = DINAttention(embedding_dim=embedding_dim)
    output = attention_layer(dummy_query, dummy_query, dummy_mask)

    print(f"注意力层输出张量：{output}")
    print(f"输出shape：{output.shape}")