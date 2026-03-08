import torch
import torch.nn as nn
from DIN import DINAttention
from PopNet import PopNet
from Gate import Gate


class DIVAN(nn.Module):
    def __init__(self, 
                 feature_cache,
                 user_num, 
                 article_num,
                 age_num,
                 device_num,
                 gender_num,
                 article_type_num,
                 article_topic_num,
                 category_num,
                 subcat_num,
                 sentiment_num,
                 pretrain_content_emb_matrix,
                 id_emb_dim=64,
                 age_emb_dim=64,
                 device_emb_dim=64,
                 gender_emb_dim=64,
                 article_emb_dim=64,
                 content_dim=64,
                 article_topic_emb_dim=64,
                 category_embed_dim=64,
                 subcat_emb_dim=64,
                 sentiment_emb_dim=64,
                 recency_dim=64,
                 model='DIVAN'):
        super(DIVAN, self).__init__()
        self.padding_sign = 0
        self.feature_cache = feature_cache
        # ==========================================
        # 1. 用户侧 Embedding 柜子
        # ==========================================
        self.user_embedding = nn.Embedding(user_num, id_emb_dim, padding_idx=self.padding_sign)
        self.age_embedding = nn.Embedding(age_num, age_emb_dim, padding_idx=self.padding_sign)
        self.device_embedding = nn.Embedding(device_num, device_emb_dim, padding_idx=self.padding_sign)
        self.gender_embedding = nn.Embedding(gender_num, gender_emb_dim, padding_idx=self.padding_sign)
        raw_content_dim = pretrain_content_emb_matrix.shape[1]
        self.content_proj = nn.Linear(raw_content_dim, content_dim, bias=False)

        self.user_emb_dim = id_emb_dim + age_emb_dim + device_emb_dim + gender_emb_dim
        self.model = model
        

        # ==========================================
        # 2. 物品侧 Embedding 柜子
        # ==========================================
        # 因为在处理history的时候用0来padding，因此第0条新闻只是占位符，使用padding_idx=0告诉embedding网络不要更新第0条新闻
        self.article_embedding = nn.Embedding(article_num, id_emb_dim, padding_idx=self.padding_sign)
        self.article_type_embedding = nn.Embedding(article_type_num, article_emb_dim, padding_idx=self.padding_sign)
        self.article_topic_embedding = nn.Embedding(article_topic_num, article_topic_emb_dim, padding_idx=self.padding_sign)
        self.article_category_embedding = nn.Embedding(category_num, category_embed_dim, padding_idx=self.padding_sign)
        self.article_subcat_embedding = nn.Embedding(subcat_num, subcat_emb_dim, padding_idx=self.padding_sign)
        self.article_sentiment_embedding = nn.Embedding(sentiment_num, sentiment_emb_dim, padding_idx=self.padding_sign)

        self.article_emb_dim = id_emb_dim + article_emb_dim + article_topic_emb_dim + category_embed_dim + subcat_emb_dim + sentiment_emb_dim

        # # ------------------------------------------
        # print("🧱 正在向 GPU 显存写入全局特征查找表...")
        
        # # 1. 文章属性表 (Article Tables)
        # pub_ts_map = torch.zeros(article_num, dtype=torch.float32)
        # type_map = torch.zeros(article_num, dtype=torch.long)
        # topic_map = torch.zeros((article_num, 3), dtype=torch.long) # 注意：Topic 是二维的
        # cat_map = torch.zeros(article_num, dtype=torch.long)
        # subcat_map = torch.zeros(article_num, dtype=torch.long)
        # senti_map = torch.zeros(article_num, dtype=torch.long)

        # for art_id, meta in news_feat_dict.items():
        #     pub_ts_map[art_id] = meta.get('published_ts', 0.0)
        #     type_map[art_id] = meta.get('article_type', 0)
        #     cat_map[art_id] = meta.get('category', 0)
        #     subcat_map[art_id] = meta.get('subcat', 0)
        #     senti_map[art_id] = meta.get('sentiment_label', 0)
            
        #     # Topic 的 Padding 逻辑在这里一次性做完！
        #     topics = meta.get('topics', [])
        #     if topics is None: topics = []
        #     topics = topics[-max_topic_len:] + [0] * (max_topic_len - len(topics))
        #     topic_map[art_id] = torch.tensor(topics, dtype=torch.long)

        # self.register_buffer('article_to_pub_ts', pub_ts_map)
        # self.register_buffer('article_to_type', type_map)
        # self.register_buffer('article_to_topics', topic_map)
        # self.register_buffer('article_to_cat', cat_map)
        # self.register_buffer('article_to_subcat', subcat_map)
        # self.register_buffer('article_to_senti', senti_map)

        # # 2. 用户历史表 (History Table)
        # # 尺寸：[用户总数, 50]，占用显存极小！
        # history_matrix = torch.zeros((user_num, max_history_len), dtype=torch.long)
        # for uid, hist in user_history_dict.items():
        #     # 这里拦截 OOV 用户 (uid == 0)，确保 0 号用户的历史全是 0
        #     if uid == 0 or hist is None:
        #         continue
        #     hist = hist[-max_history_len:] + [0] * (max_history_len - len(hist))
        #     history_matrix[uid] = torch.tensor(hist, dtype=torch.long)
            
        # self.register_buffer('user_to_history', history_matrix)

        # # ------------------------------------------

        # ==========================================
        # 3. 多模态入库
        # ==========================================
        
        self.multimodal_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrain_content_emb_matrix),
            freeze=True,
            padding_idx=self.padding_sign
        )

        # ==========================================
        # 4. 核心子网络
        # ==========================================
        self.time_linear = nn.Linear(1, recency_dim)
        self.din = DINAttention(embedding_dim=id_emb_dim+category_embed_dim+content_dim)
        self.pop_net = PopNet(recency_dim=recency_dim, content_dim=content_dim)
        self.gate = Gate(user_dim=self.user_emb_dim, recency_dim=recency_dim, content_dim=content_dim)

        # din_mlp输入维度：用户emb_dim + 文章emb_dim + din网络输出的加权emb_dim
        din_mlp_input_dim = self.user_emb_dim + self.article_emb_dim + (id_emb_dim + category_embed_dim + content_dim)

        self.din_mlp = nn.Sequential(
            nn.Linear(din_mlp_input_dim, 512), # 输入用户emb、文章idemb、历史emb
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            # nn.ReLU(),    # 不能在sigmoid前面添加relu，否则sigmoid无法输出<0.5的值
            nn.Sigmoid(),
        )

        # cat_map = torch.zeros(article_num, dtype=torch.long)
        # for art_id, meta in news_feat_dict.items():  # 需要在实例化模型时传入 news_feat_dict
        #     cat_map[art_id] = meta.get('category', 0)
        
        # # 注册为 buffer，它会自动跟随 model.to(device)
        # self.register_buffer('article_to_cat', cat_map)


    def forward(self, batch_dict):
        # output = α * din_mlp() + (1-α) * pop_score
        # ===== embedding =====
        # user_id形状为(batch,1)，torch的embedding层会在原形状最后添加一个embedding_dim，变成(batch,1,embedding_dim)，因此需要squeeze成(batch,embedding_dim)
        user_id = batch_dict['user_id'].squeeze(-1)
        age = batch_dict['age'].squeeze(-1)
        gender = batch_dict['gender'].squeeze(-1)
        device = batch_dict['device'].squeeze(-1)
        imp_time = batch_dict['imp_time'].squeeze(-1)
        target_id = batch_dict['target_id'].squeeze(-1)

        features = self.feature_cache(user_id, target_id, imp_time)

        history_ids = features['history_ids']
        target_type = features['target_type']
        target_topics = features['target_topics']
        target_category = features['target_category']
        target_subcat = features['target_subcat']
        target_sentiment_label = features['target_sentiment_label']
        target_published_ts_emb_input = features['target_published_ts_emb_input']
        hist_cat_ids = features['history_category'] # 在 GPU 上瞬间完成映射！

        # history_ids = batch_dict['history_article_ids'] # (B,50)
        # target_published_ts = batch_dict['target_published_ts']
        # target_type = batch_dict['target_article_type']
        # target_topics = batch_dict['target_article_topics']
        # target_category = batch_dict['target_article_category']
        # target_subcat = batch_dict['target_article_subcat']
        # target_sentiment_label = batch_dict['target_article_sentiment_label']

        # ==================================
        # embedding 
        # ==================================
        # user embedding
        user_id_emb = self.user_embedding(user_id).squeeze(1)   # (B,64)
        age_emb = self.age_embedding(age).squeeze(1)       # (B,16)
        gender_emb = self.gender_embedding(gender).squeeze(1)   # (B,16)
        device_emb = self.device_embedding(device).squeeze(1)   # (B,16)
        user_concated_emb = torch.cat([user_id_emb, age_emb, gender_emb, device_emb], dim=-1)   # (B,112)

        # article embedding
        target_id_emb = self.article_embedding(target_id).squeeze(1)    # (B,64)
        history_ids_emb = self.article_embedding(history_ids)   # (B,50,64)
        hist_category_emb = self.article_category_embedding(hist_cat_ids)

        target_type_emb = self.article_type_embedding(target_type).squeeze(1)   # (B,16)
        target_category_emb = self.article_category_embedding(target_category).squeeze(1)   # (B,16)
        target_subcat_emb = self.article_subcat_embedding(target_subcat).squeeze(1) # (B,16)
        target_sentiment_label_emb = self.article_sentiment_embedding(target_sentiment_label).squeeze(1)    # (B,16)

        target_topics_emb = self.article_topic_embedding(target_topics)
        target_topics_emb = torch.sum(target_topics_emb, dim=1)

    
        target_concated_emb = torch.cat([
            target_id_emb, target_type_emb, target_category_emb, target_subcat_emb, target_sentiment_label_emb, target_topics_emb], dim=-1)

        # content embedding
        # content_emb = self.multimodal_embedding(target_id).squeeze(1)   # (B,128)
        # history_content_emb = self.multimodal_embedding(history_ids)
        target_content_emb = self.content_proj(self.multimodal_embedding(target_id).squeeze(1))
        history_content_emb = self.content_proj(self.multimodal_embedding(history_ids))

        target_din_input = torch.cat([target_id_emb, target_category_emb, target_content_emb], dim=-1)
        history_din_input = torch.cat([history_ids_emb, hist_category_emb, history_content_emb], dim=-1)
        

        if target_published_ts_emb_input.dim() == 1:
            target_published_ts_emb_input = target_published_ts_emb_input.unsqueeze(-1)
        target_published_ts_emb_input = self.time_linear(target_published_ts_emb_input) # (B,16)

        mask = history_ids != self.padding_sign
        history_att_emb = self.din(querys=target_din_input.unsqueeze(1), keys=history_din_input, mask=mask)   # (B,64)
        din_mlp_input = torch.cat([user_concated_emb, target_concated_emb, history_att_emb], dim=-1)
        din_proba = self.din_mlp(din_mlp_input)
        
        if self.model == 'DIN':
            return {
                "y_pred": din_proba,      # 强制把最终预测指向 din
                "din_proba": din_proba,
                "vir_proba": torch.zeros_like(din_proba), # 用 0 占位防报错
                "alpha": torch.ones_like(alpha)           # 权重全给 DIN
            }

        pop_score = self.pop_net(recency_emb=target_published_ts_emb_input, content_emb=target_content_emb)
        pop_proba = torch.sigmoid(pop_score)

        alpha = self.gate(user_emb=user_concated_emb, recency_emb=target_published_ts_emb_input, content_emb=target_content_emb)
        
        y_pred = alpha * din_proba + (1 - alpha) * pop_proba
        self.current_alpha_mean = alpha.detach().mean().item()
        # return y_pred
        return {
            "y_pred": y_pred,
            "din_proba": din_proba,
            "vir_proba": pop_proba,
            "alpha": alpha
        }        
