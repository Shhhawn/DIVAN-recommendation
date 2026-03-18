import torch
import torch.nn as nn

class GPUFeatureCache(nn.Module):
    def __init__(self, 
                 news_feat_dict, 
                 train_history_dict, 
                 val_history_dict, 
                 article_num, 
                 user_num, 
                 max_history_len=50, 
                 max_topic_len=3):
        """
        :param news_feat_dict: 离线新闻属性字典
        :param train_history_dict: 训练集用户历史表
        :param user_history_dict: 离线用户历史字典
        :param article_num: 物品最大 ID + 1
        :param user_num: 用户最大 ID + 1
        :param max_history_len: 最大保留历史条数
        :param max_topic_len: 最多保留topic数
        """
        super(GPUFeatureCache, self).__init__()
        print("正在向 GPU 显存构建全局特征查找表 (GPUFeatureCache)...")
        
        # ==========================================
        # 1. 文章属性表 (Article Tables)
        # ==========================================
        pub_ts_map = torch.zeros(article_num, dtype=torch.float32)
        type_map = torch.zeros(article_num, dtype=torch.long)
        topic_map = torch.zeros((article_num, max_topic_len), dtype=torch.long)
        cat_map = torch.zeros(article_num, dtype=torch.long)
        subcat_map = torch.zeros(article_num, dtype=torch.long)
        senti_map = torch.zeros(article_num, dtype=torch.long)

        for art_id, meta in news_feat_dict.items():
            pub_ts_map[art_id] = int(meta.get('published_ts', 0.0))
            type_map[art_id] = meta.get('article_type', 0)
            cat_map[art_id] = meta.get('category', 0)
            subcat_map[art_id] = meta.get('subcat', 0)
            senti_map[art_id] = meta.get('sentiment_label', 0)
            
            # Topic 的 Padding 逻辑
            topics = meta.get('topics', [])
            if topics is None: topics = []
            topics = topics[-max_topic_len:] + [0] * (max_topic_len - len(topics))
            topic_map[art_id] = torch.tensor(topics, dtype=torch.long)

        # 注册为 buffer，让它们长驻显存并跟随模型保存
        self.register_buffer('article_to_pub_ts', pub_ts_map)
        self.register_buffer('article_to_type', type_map)
        self.register_buffer('article_to_topics', topic_map)
        self.register_buffer('article_to_cat', cat_map)
        self.register_buffer('article_to_subcat', subcat_map)
        self.register_buffer('article_to_senti', senti_map)

        # ==========================================
        # 2. 用户历史表 (History Table)
        # ==========================================
        
        train_hist_matrix = torch.zeros((user_num, max_history_len), dtype=torch.long)
        for uid, hist in train_history_dict.items():
            if uid == 0 or hist is None: continue
            hist = hist[-max_history_len:] + [0] * (max_history_len - len(hist))
            train_hist_matrix[uid] = torch.tensor(hist, dtype=torch.long)
        self.register_buffer('train_history', train_hist_matrix)

        val_hist_matrix = torch.zeros((user_num, max_history_len), dtype=torch.long)
        for uid, hist in val_history_dict.items():
            if uid == 0 or hist is None: continue
            hist = hist[-max_history_len:] + [0] * (max_history_len - len(hist))
            val_hist_matrix[uid] = torch.tensor(hist, dtype=torch.long)
        self.register_buffer('val_history', val_hist_matrix)

    def forward(self, user_id, target_id, imp_time):
        """
        前向查询引擎：接收一个 Batch 的光秃秃 ID，瞬间返回所有的富文本特征和计算结果
        """
        # 1. 极速查表
        if self.training:
            history_ids = self.train_history[user_id]   # (B, 50)
        else:
            history_ids = self.val_history[user_id]     # (B, 50)
        # history_ids = self.user_to_history[user_id]             
        history_category = self.article_to_cat[history_ids]
        target_type = self.article_to_type[target_id]           # (B,)
        target_topics = self.article_to_topics[target_id]       # (B, 3)
        target_category = self.article_to_cat[target_id]        # (B,)
        target_subcat = self.article_to_subcat[target_id]       # (B,)
        target_sentiment_label = self.article_to_senti[target_id] # (B,)
        target_pub_ts = self.article_to_pub_ts[target_id]       # (B,)

        # 2. 极速时序计算：代替 numpy.log1p
        target_pub_ts = torch.where(target_pub_ts == 0.0, imp_time, target_pub_ts)
        delta_hours = torch.clamp(imp_time - target_pub_ts, min=0) / 3600000.0
        target_published_ts_emb_input = torch.log1p(delta_hours.to(torch.float32)).unsqueeze(-1) # 变回 (B, 1)

        # 打包返回
        return {
            'history_ids': history_ids,
            'history_category': history_category,
            'target_type': target_type,
            'target_topics': target_topics,
            'target_category': target_category,
            'target_subcat': target_subcat,
            'target_sentiment_label': target_sentiment_label,
            'target_published_ts_emb_input': target_published_ts_emb_input
        }