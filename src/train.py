import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import joblib
from sklearn.metrics import roc_auc_score
from data_process import build_offline_article_vault, process_history_dynamic, EbnerdDataset
from FeatureCache import GPUFeatureCache
from DIVAN import DIVAN
from tqdm import tqdm
import gc
import random
import json
from datetime import datetime
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==========================================
# 参数与路径配置
# ==========================================
NOW_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_BATCH_SIZE = 12288
VAL_BATCH_SIZE = 40960
DATASET_SIZE = 'small'
EVAL_STEP_FREQ = 10
SEED = 42

train_dir = os.path.join("..", "data", str(DATASET_SIZE), "train")
val_dir = os.path.join("..", "data", str(DATASET_SIZE), "validation")
article_path = os.path.join("..", "data", str(DATASET_SIZE), "articles.parquet")
text_emb_path = os.path.join("..", "data", "roberta_vector.parquet")
image_emb_path = os.path.join("..", "data", "image_embeddings.parquet")

# 输出主目录
output_dir = os.path.join("..", "output", str(DATASET_SIZE))
processed_data_output_dir = os.path.join(output_dir, "processed_data")
result_dir = os.path.join(output_dir, "results")
os.makedirs(output_dir, exist_ok=True)

# 处理后的文件路径定义
news_feature_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_news_feature_dict.pkl')
train_behvior_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_train_behavior.parquet')
val_behvior_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_val_behavior.parquet')
user_maps_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_user_maps.pkl')
multimodal_matrix_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_multimodal_matrix.npy')

train_history_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_train_history.pkl')
val_history_path = os.path.join(processed_data_output_dir, f'{DATASET_SIZE}_val_history.pkl')




def seed_everything(seed=42):
    """
    设置随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果有多块 GPU
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"全局随机种子已固定为: {seed}")

def plot_training_metrics(history_dict, output_dir, dataset_size="small"):
    """
    绘制并保存工业级全景训练指标图 (Loss, AUC, Ranking Metrics)
    history_dict 格式示例:
    {
        'steps': [500, 1000, 1500, ...],
        'train_loss': [0.5, 0.4, ...],
        'val_loss': [0.55, 0.45, ...],
        'group_auc': [0.65, 0.68, ...],
        'global_auc': [0.64, 0.67, ...],
        'mrr': [0.3, 0.35, ...],
        'ndcg_5': [0.4, 0.45, ...],
        'ndcg_10': [0.42, 0.47, ...]
    }
    """
    print("正在绘制训练指标全景图...")
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    steps = history_dict['steps']

    # ==========================================
    # 子图 1: Loss 曲线
    # ==========================================
    ax = axes[0]
    # 找一下当前是否有 train_loss，为了防止维度不匹配，安全获取
    if 'train_loss' in history_dict and len(history_dict['train_loss']) == len(steps):
        ax.plot(steps, history_dict['train_loss'], label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    if 'val_loss' in history_dict:
        ax.plot(steps, history_dict['val_loss'], label='Val Loss', color='red', linewidth=2, marker='s', markersize=4)
    
    ax.set_title('Loss Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)

    # ==========================================
    # 子图 2: AUC 曲线
    # ==========================================
    ax = axes[1]
    if 'group_auc' in history_dict:
        ax.plot(steps, history_dict['group_auc'], label='Group AUC (Core)', color='darkorange', linewidth=2.5, marker='*')
    if 'global_auc' in history_dict:
        ax.plot(steps, history_dict['global_auc'], label='Global AUC', color='green', linewidth=1.5, linestyle='--')
    
    ax.set_title('AUC Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Steps', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.legend(fontsize=10)

    # ==========================================
    # 子图 3: 排序指标
    # ==========================================
    ax = axes[2]
    if 'ndcg_5' in history_dict:
        ax.plot(steps, history_dict['ndcg_5'], label='NDCG@5', color='purple', linewidth=2)
    if 'ndcg_10' in history_dict:
        ax.plot(steps, history_dict['ndcg_10'], label='NDCG@10', color='magenta', linewidth=2, linestyle='-.')
    if 'mrr' in history_dict:
        ax.plot(steps, history_dict['mrr'], label='MRR', color='teal', linewidth=2, linestyle=':')
    
    ax.set_title('Ranking Metrics (NDCG & MRR)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Steps', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=10)

    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{NOW_TS}_{DATASET_SIZE}_training_metrics_dashboard.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练指标图已保存至: {save_path}")


# ==========================================
# 辅助函数：计算单个 Impression 组的所有指标
# ==========================================
def calc_single_group_metrics(preds, labels):
    # 必须同时包含正负样本
    if len(np.unique(labels)) <= 1:
        return None
    
    # 1. 算 AUC
    auc = roc_auc_score(labels, preds)
    
    # 2. 算 MRR
    preds_with_noise = preds + np.random.uniform(0, 1e-9, size=len(preds))
    order = np.argsort(preds_with_noise)[::-1]
    ranks = np.where(labels[order] == 1)[0]
    mrr = 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0
    
    # 3. 算 NDCG
    ndcg_5 = calculate_ndcg_at_k(labels, preds_with_noise, k=5)
    ndcg_10 = calculate_ndcg_at_k(labels, preds_with_noise, k=10)
    
    return auc, mrr, ndcg_5, ndcg_10


def print_model_stats(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print("-" * 30)
    print(f"[Model Stats]")
    print(f"   Trainable Params: {trainable_params:,}")
    print(f"   Total Params: {all_params:,}")
    print("-" * 30)


def calculate_ndcg_at_k(labels, preds, k):
    """
    labels: 真实点击标签 (0或1)
    preds: 模型预测概率
    k: 取前几名计算
    """
    # 1. 按照预测概率从大到小排序，并截取前 K 个
    order = np.argsort(-preds)
    labels_top_k = labels[order][:k]
    
    # 2. 计算当前推荐列表的 DCG@K
    # 折损公式: 1 / log2(rank + 1)。其中 rank 从 1 开始，所以分母是 log2(i + 2)
    discount = 1.0 / np.log2(np.arange(2, len(labels_top_k) + 2))
    dcg = np.sum(labels_top_k * discount)
    
    # 3. 计算理想情况下的 IDCG@K (即真实正样本全排在最前面)
    ideal_labels_top_k = np.sort(labels)[::-1][:k]
    ideal_discount = 1.0 / np.log2(np.arange(2, len(ideal_labels_top_k) + 2))
    idcg = np.sum(ideal_labels_top_k * ideal_discount)
    
    # 4. 归一化 (如果 idcg 为 0，说明这 K 个里根本没有正样本，得分为 0)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate(model, val_loader, device, criterion, mode='DIVAN'):
    model.eval()
    val_loss = 0.0
    all_preds_list, all_labels_list, all_impressions_list = [], [], []

    with torch.no_grad():
        for batch_dict in tqdm(val_loader, desc="Validation Steps"):
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            # 接收字典
            return_dict = model(batch_dict) 
                
            # # 从中提取真正的预测概率
            # y_pred = return_dict["y_pred"].view(-1).float()
            # din_proba = return_dict["din_proba"].view(-1).float()
            # vir_proba = return_dict["vir_proba"].view(-1).float()
            # alpha = return_dict["alpha"].float()
            # label = batch_dict['label'].view(-1).float()
            
            # # 验证集通常只看最终输出的 Loss 就足够了
            # loss_final = criterion(y_pred, label)
            # loss_din = criterion(din_proba, label)
            # loss_pop = criterion(vir_proba, label)
            # loss_alpha_reg = 1e-4 * torch.norm(alpha)
            # loss = loss_final + loss_din + loss_pop + loss_alpha_reg
            # val_loss += loss.item()

            # 1. 提取所有模式共有的主预测值和真实标签
            y_pred = return_dict["y_pred"].view(-1).float()
            label = batch_dict['label'].view(-1).float()
            loss_final = criterion(y_pred, label)
            
            # 2. 动态判断：如果是完整 DIVAN 模式，则提取辅助网络并加上辅助 Loss
            if return_dict.get("vir_proba") is not None and return_dict.get("alpha") is not None:
                din_proba = return_dict["din_proba"].view(-1).float()
                vir_proba = return_dict["vir_proba"].view(-1).float()
                alpha = return_dict["alpha"].float()
                
                loss_din = criterion(din_proba, label)
                loss_pop = criterion(vir_proba, label)
                loss_alpha_reg = 1e-4 * torch.norm(alpha)
                loss = loss_final + loss_din + loss_pop + loss_alpha_reg
            else:
                # 如果是 DIN Baseline 模式，只有主 Loss
                loss = loss_final
                
            val_loss += loss.item()


            # 提取预测值并存入列表
            all_preds_list.append(y_pred.cpu().numpy().flatten())
            all_labels_list.append(batch_dict['label'].cpu().numpy().flatten())
            all_impressions_list.append(batch_dict['impression_id'].cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    all_impressions = np.concatenate(all_impressions_list)

    # 计算 Global AUC
    global_auc = roc_auc_score(all_labels, all_preds)

    # 计算 Group AUC
    # 对 impression_id 进行排序，并拿到排序后的索引
    sort_idx = np.argsort(all_impressions)
    sorted_impressions = all_impressions[sort_idx]
    sorted_preds = all_preds[sort_idx]
    sorted_labels = all_labels[sort_idx]

    # 找到 impression_id 发生变化的“边界线”
    _, split_indices = np.unique(sorted_impressions, return_index=True)
    
    # 沿着边界线切开，直接得到所有分组的 List[ndarray]
    grouped_preds = np.split(sorted_preds, split_indices[1:])
    grouped_labels = np.split(sorted_labels, split_indices[1:])

    del all_preds_list, all_labels_list, all_impressions_list
    del all_preds, all_labels, all_impressions
    del sort_idx, sorted_impressions, sorted_preds, sorted_labels
    gc.collect()

    group_aucs = []
    mrrs = []
    ndcg_5s = []
    ndcg_10s = []

    with Parallel(n_jobs=-1, batch_size='auto', backend='loky') as parallel:
        results = parallel(
            delayed(calc_single_group_metrics)(preds, labels)
            for preds, labels in tqdm(zip(grouped_preds, grouped_labels), total=len(grouped_preds), desc="Calculating Metrics")
        )

    for res in results:
        if res is not None:
            group_aucs.append(res[0])
            mrrs.append(res[1])
            ndcg_5s.append(res[2])
            ndcg_10s.append(res[3])

    group_auc = np.mean(group_aucs) if group_aucs else 0.0
    avg_mrr = np.mean(mrrs) if mrrs else 0.0
    avg_ndcg_5 = np.mean(ndcg_5s) if ndcg_5s else 0.0
    avg_ndcg_10 = np.mean(ndcg_10s) if ndcg_10s else 0.0

    return val_loss / len(val_loader), global_auc, group_auc, avg_mrr, avg_ndcg_5, avg_ndcg_10

def main(model="DIVAN"):
    seed_everything(seed=42)

    # 自动选择设备：Apple M系列芯片(MPS) > NVIDIA GPU(CUDA) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"当前训练设备: {device}")
    print(f"使用数据集: {DATASET_SIZE}")
    # ==========================================
    # 离线特征库构建 (仅需跑一次，跑完可落盘)
    # ==========================================
    print("\n" + "="*50)
    print("阶段一: 构建离线特征库")
    print("="*50)
    
    article_ids_mapping = build_offline_article_vault(
        article_path=article_path,
        text_emb_path=text_emb_path,
        image_emb_path=image_emb_path,
        output_dir=processed_data_output_dir,
        dataset_size=DATASET_SIZE,
        emb_dim=64
    )

    # ==========================================
    # 在线曝光日志清洗 (划分训练集与验证集)
    # ==========================================
    print("\n" + "="*50)
    print("阶段二: 动态清洗曝光日志 (Train & Val)")
    print("="*50)

    # 测试集
    if os.path.exists(train_behvior_path) and os.path.exists(user_maps_path) and os.path.exists(train_history_path):
        print("训练集已处理，读取user maps")
        u_map, age_map, g_map, d_map = joblib.load(user_maps_path)
        train_history_dict = joblib.load(train_history_path)
    else:
        print("测试集处理中...")
        train_df, u_map, age_map, g_map, d_map, train_history_dict = process_history_dynamic(
            history_path=f"{train_dir}/history.parquet",
            behavior_path=f"{train_dir}/behaviors.parquet",
            article_ids_mapping_dict=article_ids_mapping,
            neg_samples=4,  # 训练时 1 个正样本配 4 个负样本
        )
        print("训练集处理完毕，落盘中")
        train_df.write_parquet(train_behvior_path)
        del train_df
        print("train_df落盘完毕，已释放train_df")
        joblib.dump((u_map, age_map, g_map, d_map), user_maps_path)
        joblib.dump(train_history_dict, train_history_path)
        print("maps落盘完毕")
        gc.collect()
        print("训练集落盘及内存清除完毕")

    # 验证集
    if os.path.exists(val_behvior_path) and os.path.exists(val_history_path):
        print('验证集已处理')
        val_history_dict = joblib.load(val_history_path)
    else:
        print("验证集处理中...")
        val_df, _, _, _, _, val_history_dict = process_history_dynamic(
            history_path=f"{val_dir}/history.parquet",
            behavior_path=f"{val_dir}/behaviors.parquet",
            article_ids_mapping_dict=article_ids_mapping,
            user_ids_mapping_dict=u_map, 
            age_mapping_dict=age_map, 
            gender_mapping_dict=g_map, 
            device_mapping_dict=d_map,
            max_rows=100000,
            is_train=False
        )
        print("测试集处理完毕，落盘中")
        val_df.write_parquet(val_behvior_path)
        joblib.dump(val_history_dict, val_history_path)
        del val_df
        gc.collect()
        print("测试集落盘及内存清除完毕")


    # ==========================================
    # 动态计算模型需要的 Vocab Size (词表大小)
    # ==========================================
    # 嵌入层的矩阵大小 = 最大的映射ID + 1 (留给 0 做 padding)
    print("\n正在动态扫描计算各特征的 Vocab Size...")
    user_num = len(u_map) + 1
    article_num = len(article_ids_mapping) + 1
    age_num = len(age_map) + 1
    gender_num = len(g_map) + 1
    device_num = len(d_map) + 1

    # 扫描离线新闻字典，获取目标文章各属性的 Vocab Size
    news_feat_dict = joblib.load(news_feature_path)
    max_type, max_topic, max_cat, max_subcat, max_senti = 0, 0, 0, 0, 0
    for meta in news_feat_dict.values():
        max_type = max(max_type, meta.get('article_type', 0))
        max_cat = max(max_cat, meta.get('category', 0))
        max_subcat = max(max_subcat, meta.get('subcat', 0))
        max_senti = max(max_senti, meta.get('sentiment_label', 0))
        # topics 是列表，找出里面最大的 ID
        topics = meta.get('topics', [])
        if topics:
            max_topic = max(max_topic, max(topics))
            
    # 加 1 作为 Padding 位
    type_num, topic_num = max_type + 1, max_topic + 1
    cat_num, subcat_num, senti_num = max_cat + 1, max_subcat + 1, max_senti + 1

    # 读取之前保存的多模态矩阵
    multimodal_matrix = np.load(multimodal_matrix_path)

    # ==========================================
    # 组装 Dataset 和 DataLoader
    # ==========================================
    print("\n正在打包 DataLoader...")
    train_dataset = EbnerdDataset(train_behvior_path)
    print("train Dataset 初始化完成")
    val_dataset = EbnerdDataset(val_behvior_path)
    print('validation Dataset 初始化完成')
    # 训练集需要打乱 (shuffle=True)，验证集不需要
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

    # ==========================================
    # 实例化 DIVAN 超级引擎
    # ==========================================
    print("合并train和val历史数据")
    global_history_dict = {**train_history_dict, **val_history_dict}

    feature_cache = GPUFeatureCache(
        news_feat_dict=news_feat_dict,
        train_history_dict=train_history_dict,
        val_history_dict=val_history_dict,
        article_num=article_num,
        user_num=user_num
    )

    print("正在清空 CPU 内存垃圾...")
    del global_history_dict
    del train_history_dict
    del val_history_dict
    del news_feat_dict
    gc.collect()

    print(f"\n正在实例化 DIVAN 主模型，送往 {device}...")
    model = DIVAN(
        feature_cache=feature_cache,
        user_num=user_num, article_num=article_num,
        age_num=age_num, device_num=device_num, gender_num=gender_num,
        article_type_num=type_num, article_topic_num=topic_num,
        category_num=cat_num, subcat_num=subcat_num, sentiment_num=senti_num,
        pretrain_content_emb_matrix=multimodal_matrix,
        model=model
    ).to(device)

    print_model_stats(model)

    # 定义损失函数：二元交叉熵 (点击/未点击 预估)
    criterion = nn.BCELoss()
    # 定义优化器：Adam 算法接管模型所有可训练参数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ==========================================
    # Train Loop
    # ==========================================
    print("\n训练开始...")
    num_epochs = 50
    best_auc = 0.0
    patience = 10
    no_improve_cnt = 0
    print_freq = 50
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    val_loss, global_auc, group_auc, avg_mrr, ndcg_5, ndcg_10 = evaluate(model, val_loader, device, criterion, mode="DIVAN")
    print(f"[!metrics] 初始化排序指标 -> Group AUC: {group_auc:.4f} | MRR: {avg_mrr:.4f} | NDCG@5: {ndcg_5:.4f} | NDCG@10: {ndcg_10:.4f}")
    global_step = 0
    metrics_history = {
        'steps': [], 'val_loss': [], 'group_auc': [], 
        'global_auc': [], 'mrr': [], 'ndcg_5': [], 'ndcg_10': []
    }

    for epoch in range(num_epochs):
        model.train()  # 开启训练模式
        total_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
        for batch_idx, batch_dict in enumerate(train_pbar):
            global_step += 1
            # 将字典里所有的 Tensor 转移到计算设备 (GPU/MPS) 上
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            return_dict = model(batch_dict)
            y_true = batch_dict['label'].view(-1)
            
            # 计算“三重辅助误差”
            loss_final = criterion(return_dict["y_pred"].view(-1), y_true)
            if return_dict["vir_proba"] is not None and return_dict["alpha"] is not None:
                loss_din = criterion(return_dict["din_proba"].view(-1), y_true)
                loss_pop = criterion(return_dict["vir_proba"].view(-1), y_true)
            
                # alpha 的 L2 正则化惩罚
                loss_alpha_reg = 1e-4 * torch.norm(return_dict["alpha"])
            
                # 总 Loss
                loss = loss_final + loss_din + loss_pop + loss_alpha_reg
            else:
                loss = loss_final
            
            # 反向传播
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx > 0 and batch_idx % EVAL_STEP_FREQ == 0:
                tqdm.write(f"\n[Mid-Epoch Eval] 触发步级验证 (Step {batch_idx})...")
                
                # 跑验证集
                val_loss, global_auc, group_auc, avg_mrr, ndcg_5, ndcg_10 = evaluate(
                    model, val_loader, device, criterion, mode="DIVAN"
                )

                metrics_history['steps'].append(global_step)
                metrics_history['val_loss'].append(val_loss)
                metrics_history['group_auc'].append(group_auc)
                metrics_history['global_auc'].append(global_auc)
                metrics_history['mrr'].append(avg_mrr)
                metrics_history['ndcg_5'].append(ndcg_5)
                metrics_history['ndcg_10'].append(ndcg_10)
                
                tqdm.write(f"--> Step [{batch_idx}] | Group AUC: {group_auc:.4f} | NDCG@5: {ndcg_5:.4f}")
                
                # 如果是历史最高分，立刻保存
                if group_auc > best_auc:
                    best_auc = group_auc
                    no_improve_cnt = 0
                    tqdm.write(f"捕获隐藏最高分: {best_auc:.4f}！正在保存快照...")
                    torch.save(model.state_dict(), f"{result_dir}/{DATASET_SIZE}_best_divan_model.pth")
                else:
                    no_improve_cnt += 1
                    tqdm.write(f"Group AUC 未提升 ({no_improve_cnt}/{patience})， 目前最高AUC: {best_auc:.4f}")
                
                # 切回 train 模式
                model.train()

                if no_improve_cnt >= patience:
                    print(f"[finish] 连续多轮未提升，触发 Early Stopping！最高AUC：{best_auc:.4f}")
                    plot_training_metrics(metrics_history, result_dir, DATASET_SIZE)

                    json_path = os.path.join(output_dir, "results", f"{NOW_TS}_{DATASET_SIZE}_training_metrics.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metrics_history, f, indent=4, ensure_ascii=False)
                    print(f"原始指标数据已安全落盘至: {json_path}")
                    return  # 直接结束 main 函数
        
        # --- 每个 Epoch 结束，跑一次验证集 ---
        if global_step % TRAIN_BATCH_SIZE != 0:
            torch.cuda.empty_cache()
            val_loss, global_auc, group_auc, avg_mrr, ndcg_5, ndcg_10 = evaluate(model, val_loader, device, criterion, mode="DIVAN")
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Val Loss: {val_loss:.4f} | Global auc: {global_auc:.4f}")
            print(f"[!metrics] 核心排序指标 | Group AUC: {group_auc:.4f} | MRR: {avg_mrr:.4f} | NDCG@5: {ndcg_5:.4f} | NDCG@10: {ndcg_10:.4f}")
            
            metrics_history['steps'].append(global_step)  
            metrics_history['val_loss'].append(val_loss)
            metrics_history['group_auc'].append(group_auc)
            metrics_history['global_auc'].append(global_auc)
            metrics_history['mrr'].append(avg_mrr)
            metrics_history['ndcg_5'].append(ndcg_5)
            metrics_history['ndcg_10'].append(ndcg_10)

            if group_auc > best_auc:
                best_auc = group_auc
                no_improve_cnt = 0
                print(f"[!metrics] 发现更高 Group AUC: {best_auc:.4f}！正在保存模型权重...")
                torch.save(model.state_dict(), f"{result_dir}/{DATASET_SIZE}_best_divan_model.pth")
            else:
                no_improve_cnt += 1
                print(f"Group AUC 未提升 ({no_improve_cnt}/{patience})， 目前最高AUC: {best_auc:.4f}")
                if no_improve_cnt >= patience:
                    print("[finish] 连续多轮未提升，触发 Early Stopping！最高AUC：{best_auc:.4f}")
                    plot_training_metrics(metrics_history, result_dir, DATASET_SIZE)

                    json_path = os.path.join(output_dir, "results", f"{NOW_TS}_{DATASET_SIZE}_training_metrics.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metrics_history, f, indent=4, ensure_ascii=False)
                    print(f"原始指标数据已安全落盘至: {json_path}")
                    break

        print('---------------------------------------------\n')
        scheduler.step()


if __name__ == "__main__":
    main(model="DIN")