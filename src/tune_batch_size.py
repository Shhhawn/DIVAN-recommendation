import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
import numpy as np

from data_process import EbnerdDataset
from DIVAN import DIVAN
from FeatureCache import GPUFeatureCache

# ==========================================
# 压测候选参数
# ==========================================
# 训练由于需要保存梯度，较吃显存，上限较低
TRAIN_CANDIDATES = [4096, 6144, 8192, 10240, 12288, 16384]

# 验证集不保存梯度，纯前向传播，可以调高上限
VAL_CANDIDATES = [16384, 24576, 32768, 40960, 49152, 65536]

# 压测步数配置
WARMUP_STEPS = 3      # 预热步数（不计入时间）
MEASURE_STEPS = 10    # 计时的稳定步数

def measure_throughput(model, data_loader, device, batch_size, is_train=True):
    """
    核心压测函数：计算每秒处理的样本数 (Samples / Second)
    """
    if is_train:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        model.eval()

    data_iter = iter(data_loader)
    
    try:
        # =================================
        # Warm-up
        # =================================
        for _ in range(WARMUP_STEPS):
            batch_dict = next(data_iter)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            if is_train:
                optimizer.zero_grad()
                out = model(batch_dict)
                loss = out['y_pred'].sum() # 随便弄个标量模拟 loss
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    model(batch_dict)
                    
        # 同步，等待预热的 GPU 任务彻底跑完
        torch.cuda.synchronize(device)
        
        # =================================
        # Measure
        # =================================
        start_time = time.time()
        
        for _ in range(MEASURE_STEPS):
            batch_dict = next(data_iter)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            if is_train:
                optimizer.zero_grad()
                out = model(batch_dict)
                loss = out['y_pred'].sum()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    model(batch_dict)
                    
        # 同步
        torch.cuda.synchronize(device)
        end_time = time.time()
        
        # 计算吞吐量
        total_time = end_time - start_time
        total_samples = batch_size * MEASURE_STEPS
        throughput = total_samples / total_time
        
        return True, throughput
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # 捕获 OOM 错误
            return False, 0.0
        else:
            raise e


def main():
    dataset_size = 'small'
    output_dir = os.path.join(".", "processed_data")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始压测，设备: {device}")

    # ==========================================
    # 模拟环境加载
    # ==========================================
    print("正在加载必要环境字典...")
    news_feat_dict = joblib.load(os.path.join(output_dir, f'{dataset_size}_news_feature_dict.pkl'))
    train_history_dict = joblib.load(os.path.join(output_dir, f'{dataset_size}_train_history.pkl'))
    val_history_dict = joblib.load(os.path.join(output_dir, f'{dataset_size}_val_history.pkl'))
    u_map, age_map, g_map, d_map = joblib.load(os.path.join(output_dir, f'{dataset_size}_user_maps.pkl'))
    
    user_num = len(u_map) + 1
    article_num = 200000 
    
    global_history_dict = {**train_history_dict, **val_history_dict}
    
    feature_cache = GPUFeatureCache(
        news_feat_dict=news_feat_dict,
        user_history_dict=global_history_dict,
        article_num=article_num,
        user_num=user_num
    )
    
    del global_history_dict, train_history_dict, val_history_dict, news_feat_dict
    import gc; gc.collect()
    
    # 假数据矩阵，模拟 pretrain_content_emb_matrix 占用的显存
    multimodal_matrix = np.zeros((article_num, 128), dtype=np.float32)

    model = DIVAN(
        feature_cache=feature_cache,
        user_num=user_num, article_num=article_num,
        age_num=len(age_map)+1, device_num=len(d_map)+1, gender_num=len(g_map)+1,
        article_type_num=20, article_topic_num=300, category_num=100, subcat_num=200, sentiment_num=5,
        pretrain_content_emb_matrix=multimodal_matrix,
        model='DIVAN'
    ).to(device)

    # 加载数据集
    train_parquet = os.path.join(output_dir, f'{dataset_size}_train_behavior.parquet')
    val_parquet = os.path.join(output_dir, f'{dataset_size}_val_behavior.parquet')
    
    train_dataset = EbnerdDataset(train_parquet)
    val_dataset = EbnerdDataset(val_parquet)

    # ==========================================
    # 训练集极限压测
    # ==========================================
    print("\n" + "="*50)
    print("阶段一: 寻找【训练集】最优 Batch Size (Train Mode)")
    print("="*50)
    
    best_train_bs = 0
    best_train_tp = 0
    
    for bs in TRAIN_CANDIDATES:
        loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        print(f"Testing Batch Size: {bs:<6} ... ", end="")
        
        success, throughput = measure_throughput(model, loader, device, bs, is_train=True)
        
        if success:
            print(f"Throughput: {throughput:,.0f} samples/sec")
            if throughput > best_train_tp:
                best_train_tp = throughput
                best_train_bs = bs
        else:
            print(f"OOM (Out of Memory)")
            torch.cuda.empty_cache() # 爆显存后清理现场
            break # 后面的数字更大，不用测了
            
        torch.cuda.empty_cache() # 测完一个清理一个

    # ==========================================
    # 验证集极限压测
    # ==========================================
    print("\n" + "="*50)
    print("阶段二: 寻找【验证集】最优 Batch Size (Eval Mode)")
    print("="*50)
    
    best_val_bs = 0
    best_val_tp = 0
    
    for bs in VAL_CANDIDATES:
        loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=True)
        print(f"Testing Batch Size: {bs:<6} ... ", end="")
        
        success, throughput = measure_throughput(model, loader, device, bs, is_train=False)
        
        if success:
            print(f"Throughput: {throughput:,.0f} samples/sec")
            if throughput > best_val_tp:
                best_val_tp = throughput
                best_val_bs = bs
        else:
            print(f"OOM (Out of Memory)")
            torch.cuda.empty_cache()
            break
            
        torch.cuda.empty_cache()

    print("\n" + "-"*20)
    print("压测结论报告：")
    print(f"最优训练 Batch Size: {best_train_bs} (吞吐量: {best_train_tp:,.0f} 样本/秒)")
    print(f"最优验证 Batch Size: {best_val_bs} (吞吐量: {best_val_tp:,.0f} 样本/秒)")
    print("请将这两个数字分别填入你的 train.py 中")

if __name__ == "__main__":
    main()