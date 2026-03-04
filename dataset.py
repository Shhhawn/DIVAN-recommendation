import torch
from torch.utils.data import Dataset
import polars as pl


def build_dataset_dict(df_history_path, df_articles_path):
    df_history = pl.read_parquet(df_history_path)
    df_articles = pl.read_parquet(df_articles_path).with_columns(pl.col('published_time').dt.timestamp("ms").alias('published_ts'))

    # 构建users-history字典
    # [!note] .to_list()将history的类型从polars的series转换成python的list
    user_history_dict = dict(zip(df_history['user_id'], df_history['article_id_fixed'].to_list()))
    
    # 构建articles-info字典
    article_dict = dict(zip(df_articles['article_id'], df_articles['published_ts'].to_list()))

    return user_history_dict, article_dict

def explode_user_behaviors(behavor_path):
    df_behaviors = pl.scan_parquet(behavor_path)

    df_behaviors = (
        df_behaviors
        .explode('article_ids_inview')
        .with_columns(pl.col('article_ids_inview').is_in(pl.col('article_ids_clicked')).cast(pl.Int8).alias('labels'))
        .select(['user_id', 'article_ids_inview', 'labels'])
        .collect()
    )

    return df_behaviors.to_dicts()

# def padding_history()

class EbnerdDataset(Dataset):
    def __init__(self, history_dict, article_dict, behaviors_dict, MAX_HISTORY_LEN=50):
        self.history_dict = history_dict
        self.article_dict = article_dict
        self.behaviors_dict = behaviors_dict
        self.MAX_HISTORY_LEN = MAX_HISTORY_LEN

    def __len__(self):
        return len(self.behaviors_dict)
    
    def __getitem__(self, idx):
        # user_id, target_article_id, history_article_ids, target_article_published_ts, label
        user_behavior = self.behaviors_dict[idx]
        user_id = user_behavior['user_id']
        target_article_id = user_behavior['article_ids_inview']
        target_published_ts = self.article_dict.get(target_article_id, 0)
        label = user_behavior['labels']

        history = self.history_dict.get(user_id, [])[-self.MAX_HISTORY_LEN:]
        history_with_padding = history + [0] * (self.MAX_HISTORY_LEN - len(history))

        return {
            # [!note] 这里的user_id等单值需要加上[]，给特征单独一个维度，便于后续batch时的拼接
            'user_id': torch.tensor([user_id], dtype=torch.long),
            'target_article_id': torch.tensor([target_article_id],dtype=torch.long),
            'history_article_ids': torch.tensor(history_with_padding,dtype=torch.long),
            'target_published_ts': torch.tensor([target_published_ts],dtype=torch.long),
            'label': torch.tensor([label], dtype=torch.float)
        }



