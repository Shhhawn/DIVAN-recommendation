import torch
from torch.utils.data import Dataset
import polars as pl
import joblib
import gc
import numpy as np
from sklearn.decomposition import PCA
import random
import os

def create_id_mapping(raw_ids_series: pl.Series) -> dict:
    """
    将离散特征映射到[1, 特征数]，0作为padding
    :param raw_ids_series: Polars Series(包含所有唯一的原始ID)
    :return dict: {raw_id: mapped_id}(mapped_id从1开始)
    """
    if raw_ids_series.dtype == pl.List:
        unique_ids = raw_ids_series.explode().drop_nulls().unique().to_list()
    else:
        unique_ids = raw_ids_series.drop_nulls().unique().to_list()
    ids_map = {raw_id:mapped_id + 1 for mapped_id, raw_id in enumerate(unique_ids)}
    return ids_map


def build_offline_article_vault(
        article_path, 
        text_emb_path, 
        image_emb_path, 
        output_dir, 
        dataset_size='large',
        emb_dim=64) -> dict:
    """
    将新闻离散特征转化成id；
    保存{mapped_id: {published_time, topics, category, subcategory, sentiment_label, article_type}}字典为。pkl文件；
    保存text_emb,image_emb为npy文件

    :param arcicle_path: 文章数据集路径
    :param text_emb_path: 文本emb数据集路径
    :param image_emb_path: 图片emb数据集路径
    :param output_dir: 输出文件路径
    :param emb_dim: pca降维维度

    :return article_ids_mapping_dict: 文章id映射字典{raw_id:mapped_id}
    """

    # ==========================================
    # 1. 处理文章特征
    # ==========================================
    print("=" * 25 + " Processing article data " + "=" * 25)
    df_article = (pl.scan_parquet(article_path)
                  .drop_nulls("article_id")
                  .collect())

    article_ids_mapping_dict = create_id_mapping(df_article['article_id'])

    if os.path.exists(os.path.join(output_dir, f"{dataset_size}_news_feature_dict.pkl")):
        print("news features 已落盘")
    else:
        df_article = (
            df_article
            .with_columns(
                pl.col('article_id').replace(article_ids_mapping_dict, default=0),
                pl.col('published_time').dt.timestamp("ms").alias('published_ts'),
                pl.col('subcategory').list.first().fill_null(0).alias('subcat'),
            )
            .select(['article_id', 
                    'published_ts', 
                    'topics', 
                    'category', 
                    'subcat', 
                    'sentiment_label', 
                    'article_type'])
        )

        article_discrete_col = ['article_id','article_type','topics','sentiment_label','category','subcat']
        for col in article_discrete_col[1:]:
            col_map = create_id_mapping(df_article[col])
            if df_article[col].dtype == pl.List:
                df_article = df_article.with_columns(
                    pl.col(col).map_elements(lambda lst: [col_map.get(x,0) for x in lst] if lst is not None else [], return_dtype=pl.List(pl.Int64))
                )
            else:
                df_article = df_article.with_columns(pl.col(col).replace(col_map, default=0))
            del col_map

        news_features_dict = {row['article_id']:
                            {'published_ts':row['published_ts'],
                            'article_type':row['article_type'],
                            'topics':row['topics'],
                            'category':row['category'],
                            'subcat':row['subcat'],
                            'sentiment_label':row['sentiment_label']
                            }for row in df_article.iter_rows(named=True)}    # .iter_rows默认返回元组，named=True开启字符串取值
        
        joblib.dump(news_features_dict, f"{output_dir}/{dataset_size}_news_feature_dict.pkl")
        print(f'新闻特征输出至 {output_dir}/{dataset_size}_news_feature_dict.pkl')

        del news_features_dict
        gc.collect()

    # ==========================================
    # 2. 处理text embedding和image embedding
    # ==========================================
    if os.path.exists(os.path.join(output_dir, f"{dataset_size}_multimodal_matrix.npy")):
        print("multimodal matrix 已落盘")
    else:
        print("=" * 25 + " Processing embedding data " + "=" * 25)
        pca = PCA(n_components=emb_dim)

        df_text_emb = pl.read_parquet(text_emb_path).rename({'FacebookAI/xlm-roberta-base':'text_embedding'})
        df_image_emb = pl.read_parquet(image_emb_path)

        text_emb = pca.fit_transform(np.array(df_text_emb['text_embedding'].to_list()))
        image_emb = pca.fit_transform(np.array(df_image_emb['image_embedding'].to_list()))

        text_emb_dict = {id:emb for id, emb in zip(df_text_emb['article_id'], text_emb)}
        image_emb_dict = {id:emb for id, emb in zip(df_image_emb['article_id'], image_emb)}

        num_articles = len(article_ids_mapping_dict) + 1
        multimodal_matrix = np.zeros((num_articles, emb_dim*2), dtype=np.float32)


        for raw_id, mapped_id in article_ids_mapping_dict.items():
            t_emb = text_emb_dict.get(raw_id, np.zeros(emb_dim, dtype=np.float32))
            i_emb = image_emb_dict.get(raw_id, np.zeros(emb_dim, dtype=np.float32))
            multimodal_matrix[mapped_id] = np.concatenate([t_emb, i_emb])

        np.save(f'{output_dir}/{dataset_size}_multimodal_matrix.npy', multimodal_matrix)
        print(f'新闻内容embedding输出至 {output_dir}/{dataset_size}_multimodal_matrix.npy')

        del text_emb_dict, image_emb_dict, multimodal_matrix
        gc.collect()

    return article_ids_mapping_dict


'''
def process_history_dynamic(
        history_path, 
        behavior_path, 
        article_ids_mapping_dict,
        user_ids_mapping_dict=None, 
        age_mapping_dict=None, 
        gender_mapping_dict=None, 
        device_mapping_dict=None,
        neg_samples=14, 
        max_rows=None
    ):
    """
    处理曝光和用户点击行为，以及当此曝光时刻时用户的历史信息
    """
    
    print("=" * 25 + " Processing behavior data " + "=" * 25)
    # ==========================================
    # 1. 提取行为数据并构建画像映射字典
    # ==========================================
    df_behavior = pl.read_parquet(behavior_path).drop_nulls('user_id')
    if max_rows is not None:
        df_behavior = df_behavior.head(max_rows)
    # df_behavior = (
    #     df_behavior
    #     .drop_nulls('user_id')
    #     .select(['user_id','device_type','article_ids_inview','article_ids_clicked','gender','age'])
    #     .collect()
    # )

    # 构建id映射
    if user_ids_mapping_dict is None:
        df_temp = df_behavior.select(['user_id', 'age', 'gender', 'device_type'])# .collect()
        user_ids_mapping_dict = create_id_mapping(df_temp['user_id'])
        age_mapping_dict = create_id_mapping(df_temp['age'])
        gender_mapping_dict = create_id_mapping(df_temp['gender'])
        device_mapping_dict = create_id_mapping(df_temp['device_type'])
        del df_temp
        gc.collect()

    # ==========================================
    # 2. 曝光日志：负采样与 Point-wise 展开
    # ==========================================
    df_behavior = (
        df_behavior
        # .lazy()  # 在进行复杂运算前重新开启惰性引擎
        .with_columns(
            # 将离散特征替换成id
            # pl.col('user_id').replace(user_ids_mapping_dict, default=0),
            pl.col('age').replace(age_mapping_dict, default=0),
            pl.col('gender').replace(gender_mapping_dict, default=0),
            pl.col('device_type').replace(device_mapping_dict, default=0),
            clicked_times=pl.col('article_ids_clicked').list.len(),
        )
        .filter(pl.col('clicked_times') == 1)
        .with_columns(
            pl.col('article_ids_inview').list.set_difference(pl.col('article_ids_clicked'))
            .alias('neg_candidates')
        )
        .with_columns(
            pl.col('neg_candidates')
            .map_elements(
                lambda lst: random.sample(list(lst), min(len(list(lst)), neg_samples)) if lst is not None else [], return_dtype=pl.List(pl.Int64)
            )
            # .list.head(neg_samples)
            .alias('samples')
        )
        .with_columns(pl.concat_list(['samples', 'article_ids_clicked']))
        .drop(['neg_candidates','article_ids_inview','clicked_times'])
        .explode('samples')
        .with_columns(
            pl.col('samples').is_in(pl.col('article_ids_clicked')).cast(pl.Float32).alias('labels'),
            pl.col('samples').replace(article_ids_mapping_dict, default=0).alias('target_ids')
        )
        .drop(['samples', 'article_ids_clicked'])
    )
    gc.collect()

    # ==========================================
    # 3. 处理历史记录序列
    # ==========================================
    df_history = (
        pl.read_parquet(history_path)
        .drop_nulls('user_id')
        .with_columns(
            # pl.col('user_id').replace(user_ids_mapping_dict, default=0),
            pl.col('article_id_fixed').map_elements(
                lambda lst: [article_ids_mapping_dict.get(x, 0) for x in lst] if lst is not None else [], return_dtype=pl.List(pl.Int64)
            ).alias('history_ids')
        )
        .select(['user_id','history_ids'])
    )

    df_behavior = (
        df_behavior
        .join(df_history, on='user_id', how='left')
        # .collect()
        # .sample(fraction=1.0, shuffle=True)
    )
    
    df_final = df_final.with_columns(
        pl.col('user_id').replace(user_ids_mapping_dict, default=0)
    )
    # behavior_dict: user_id, age, gender, device_type, target_ids, labels, history_ids
    # behavior_dict = df_behavior.to_dicts()

    return df_behavior, user_ids_mapping_dict, age_mapping_dict, gender_mapping_dict, device_mapping_dict
'''
def process_history_dynamic(
        history_path, 
        behavior_path, 
        article_ids_mapping_dict,
        user_ids_mapping_dict=None, 
        age_mapping_dict=None, 
        gender_mapping_dict=None, 
        device_mapping_dict=None,
        neg_samples=14, 
        max_rows=None,
        is_train=True
    ):
    """
    处理曝光和用户点击行为，以及当此曝光时刻时用户的历史信息
    """
    print("=" * 25 + " Processing behavior data " + "=" * 25)
    
    # ==========================================
    # 1. 提取行为数据并构建画像映射字典
    # ==========================================
    # 彻底抛弃 lazy，直接 read_parquet 全内存执行，对于 20 万行数据这是最快的
    df_behavior = (
        pl.read_parquet(behavior_path)
        .drop_nulls('user_id')
        .select(['article_id','device_type','article_ids_inview','article_ids_clicked','user_id','gender','age','impression_time'])
    )

    if max_rows is not None:
        df_behavior = df_behavior.head(max_rows)

    df_behavior = df_behavior.with_row_index("impression_id")

    # 提前过滤出有点击行为的数据
    df_behavior = (
        df_behavior.with_columns(clicked_times=pl.col('article_ids_clicked').list.len())
        .filter(pl.col('clicked_times') == 1)
    )

    # 只有训练集才构建新字典
    if user_ids_mapping_dict is None:
        print("🔍 正在构建离散特征字典...")
        # 临时切片提取字典，用完即毁
        df_temp = df_behavior.select(['user_id', 'age', 'gender', 'device_type'])
        user_ids_mapping_dict = create_id_mapping(df_temp['user_id'])
        age_mapping_dict = create_id_mapping(df_temp['age'])
        gender_mapping_dict = create_id_mapping(df_temp['gender'])
        device_mapping_dict = create_id_mapping(df_temp['device_type'])
        del df_temp
        gc.collect()

    # 根据是否是训练集，决定是否截断负样本
    if is_train:
        # 训练集：打乱并截取 neg_samples 个负样本
        sample_expr = pl.col('neg_candidates').list.eval(pl.element().shuffle()).list.head(neg_samples)
    else:
        # 🌟 验证/测试集：保留同一次曝光下的所有候选新闻，原汁原味！
        sample_expr = pl.col('neg_candidates')

    # ==========================================
    # 2. 曝光日志：原生底层负采样与展开
    # ==========================================
    df_behavior = (
        df_behavior
        .with_columns(
            pl.col('impression_time').dt.timestamp("ms"),
            # user_id，保留原始字符串
            pl.col('age').replace(age_mapping_dict, default=0),
            pl.col('gender').replace(gender_mapping_dict, default=0),
            pl.col('device_type').replace(device_mapping_dict, default=0),
            clicked_times=pl.col('article_ids_clicked').list.len()
        )
        .with_columns(
            pl.col('article_ids_inview').list.set_difference(pl.col('article_ids_clicked')).alias('neg_candidates')
        )
        .with_columns(
            # 🚀 王者归来：使用底层的 list.eval 和 shuffle，速度比 python 快十倍！
            # pl.col('neg_candidates').list.eval(pl.element().shuffle()).list.head(neg_samples).alias('samples')
            sample_expr.alias('samples')
        )
        .with_columns(pl.concat_list(['samples', 'article_ids_clicked']))
        .drop(['neg_candidates','article_ids_inview','clicked_times'])
        .explode('samples')
        .with_columns(
            pl.col('samples').is_in(pl.col('article_ids_clicked')).cast(pl.Float32).alias('labels'),
            pl.col('samples').replace(article_ids_mapping_dict, default=0).alias('target_ids')
        )
        .drop(['samples', 'article_ids_clicked'])
    )

    df_behavior = df_behavior.with_columns(
        pl.col('user_id').replace(user_ids_mapping_dict, default=0)
    )

    # ==========================================
    # 3. 处理历史记录序列
    # ==========================================

    
    print("⏳ 正在构建轻量级历史记录字典...")
    df_history = (
        pl.read_parquet(history_path)
        .drop_nulls('user_id')
        .with_columns(
            pl.col('article_id_fixed').list.eval(
                pl.element().replace(article_ids_mapping_dict, default=0)
            ).alias('history_ids')
        )
        .with_columns( # 映射 user_id
            pl.col('user_id').replace(user_ids_mapping_dict, default=0)
        )
        .select(['user_id','history_ids'])
    )

    # 🌟 转换为极速查表的 Python 原生字典：{mapped_user_id: [mapped_history_ids]}
    user_history_dict = dict(zip(df_history['user_id'].to_list(), df_history['history_ids'].to_list()))
    
    # 清理一波内存
    del df_history
    gc.collect()

    # 🌟 返回值增加 user_history_dict
    return df_behavior, user_ids_mapping_dict, age_mapping_dict, gender_mapping_dict, device_mapping_dict, user_history_dict



    # df_history = (
    #     pl.read_parquet(history_path)
    #     .drop_nulls('user_id')
    #     .with_columns(
    #         # ⚠️ 注意：同样绝对不能碰 user_id！
    #         # 🚀 王者归来：用底层的 list.eval 替代 map_elements
    #         pl.col('article_id_fixed').list.eval(
    #             pl.element().replace(article_ids_mapping_dict, default=0)
    #         ).alias('history_ids')
    #     )
    #     .select(['user_id','history_ids'])
    # )

    # # ==========================================
    # # 4. 终极 Join 与 安全映射
    # # ==========================================
    # print("⏳ 正在执行原生态精准 Join 操作...")
    # df_behavior = df_behavior.join(df_history, on='user_id', how='left')
    
    # # user_id不能在join之前转换，否则碰到OOV的用户，在join时会出现笛卡尔积爆炸
    # df_behavior = df_behavior.with_columns(
    #     pl.col('user_id').replace(user_ids_mapping_dict, default=0)
    # )
    
    # # 清理一波内存，保证返回时干干净净
    # del df_history
    # gc.collect()

    # return df_behavior, user_ids_mapping_dict, age_mapping_dict, gender_mapping_dict, device_mapping_dict
    

class EbnerdDataset(Dataset):
    def __init__(self, 
                 parquet_path, 
                #  news_features_dict, 
                #  user_history_dict, 
                #  max_history_len=50, 
                #  max_topic_len=3
                 ):
        """
        :param behavior_dict_list: process_history_dynamic 输出的字典列表
        :param news_features_dict: 离线构建的文章属性字典 (包含 published_ts 等)
        """
        print(f"加载{parquet_path}...")
        df = pl.read_parquet(parquet_path)

        self.data = {
            'impression_id': torch.from_numpy(df['impression_id'].to_numpy().copy()),
            'user_id': torch.from_numpy(df['user_id'].to_numpy().copy()),
            'age': torch.from_numpy(df['age'].to_numpy().copy()),
            'gender': torch.from_numpy(df['gender'].to_numpy().copy()),
            'device': torch.from_numpy(df['device_type'].to_numpy().copy()),
            'target_id': torch.from_numpy(df['target_ids'].to_numpy().copy()),
            'imp_time': torch.from_numpy(df['impression_time'].to_numpy().copy()),
            'label': torch.from_numpy(df['labels'].cast(pl.Float32).to_numpy().copy())
        }
        # -------------------------------
        # self.impression_ids = df_behavior['impression_id'].to_numpy()  # 新增读取
        # self.user_ids = df_behavior['user_id'].to_numpy()
        # self.ages = df_behavior['age'].to_numpy()
        # self.genders = df_behavior['gender'].to_numpy()
        # self.devices = df_behavior['device_type'].to_numpy()
        # self.target_ids = df_behavior['target_ids'].to_numpy()
        # self.labels = df_behavior['labels'].to_numpy()
        # self.impression_time = df_behavior['impression_time'].to_numpy()
        # -------------------------------

        # 列表类型的列，转换为 Python 原生嵌套列表
        # self.histories_series = df_behavior['history_ids']
        # self.news_features_dict = news_features_dict
        # self.user_history_dict = user_history_dict
        # self.max_history_len = max_history_len
        # self.max_topic_len = max_topic_len
        self.length = len(df)

        del df
        gc.collect()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # user_id = self.user_ids[idx]
        # age = self.ages[idx]
        # gender = self.genders[idx]
        # device = self.devices[idx]
        # target_id = self.target_ids[idx]
        # label = self.labels[idx]
        # imp_time = self.impression_time[idx]

        # if user_id == 0:
        #     history = []
        # else:
        #     history = self.user_history_dict.get(user_id, [])

        # if history is None:
        #     history = []
        # history = list(history)
        # history = history[-self.max_history_len:]
        # history = history + [0] * (self.max_history_len - len(history))


        # article_meta = self.news_features_dict.get(target_id, {})
        # pub_ts = article_meta.get('published_ts', imp_time)
        # delta_hours = max(0, imp_time - pub_ts) / (1000 * 3600)  # 转换为小时
        # # 取对数平滑，彻底激活 PopNet 的时间感知能力！
        # published_ts = np.log1p(delta_hours)


        # article_type = article_meta.get('article_type', 0)
        
        # topics = article_meta.get('topics', [])
        # if topics is None:
        #     topics = []

        # topics = topics[-self.max_topic_len:]
        # topics = topics + [0] * (self.max_topic_len - len(topics))
        # category = article_meta.get('category', 0)
        # subcat = article_meta.get('subcat', 0)
        # sentiment_label = article_meta.get('sentiment_label', 0)


        return {
            # 离散特征类型为torch.long
            # 'impression_id': self.impression_ids[idx],
            # 'user_id': torch.tensor([user_id], dtype=torch.long),
            # 'age': torch.tensor([age], dtype=torch.long),
            # 'gender': torch.tensor([gender], dtype=torch.long),
            # 'device': torch.tensor([device], dtype=torch.long),
            # 'target_id': torch.tensor([target_id], dtype=torch.long),
            # 'target_article_type': torch.tensor([article_type], dtype=torch.long),
            # 'target_article_topics': torch.tensor(topics, dtype=torch.long),
            # 'target_article_category': torch.tensor([category], dtype=torch.long),
            # 'target_article_subcat': torch.tensor([subcat], dtype=torch.long),
            # 'target_article_sentiment_label': torch.tensor([sentiment_label], dtype=torch.long),

            # 'history_article_ids': torch.tensor(history, dtype=torch.long),

            # 连续特征和标签使用torch.float32
            # 'target_published_ts': torch.tensor([published_ts], dtype=torch.float32),
            # 'label': torch.tensor([label], dtype=torch.float32)
            # -------------------------------
            # 'impression_id': self.impression_ids[idx],
            # 'user_id': torch.tensor([self.user_ids[idx]], dtype=torch.long),
            # 'age': torch.tensor([self.ages[idx]], dtype=torch.long),
            # 'gender': torch.tensor([self.genders[idx]], dtype=torch.long),
            # 'device': torch.tensor([self.devices[idx]], dtype=torch.long),
            # 'target_id': torch.tensor([self.target_ids[idx]], dtype=torch.long),
            
            # # 🌟 把时间戳当做特征直接扔给 GPU
            # 'imp_time': torch.tensor([self.impression_time[idx]], dtype=torch.float32), 
            # 'label': torch.tensor([self.labels[idx]], dtype=torch.float32)

            'impression_id': self.data['impression_id'][idx],
            'user_id': self.data['user_id'][idx],
            'age': self.data['age'][idx],
            'gender': self.data['gender'][idx],
            'device': self.data['device'][idx],
            'target_id': self.data['target_id'][idx],
            'imp_time': self.data['imp_time'][idx],
            'label': self.data['label'][idx]
        }

