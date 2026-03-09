# DIVAN-plus: 工业级多模态新闻推荐系统

## 1. 项目概述

本项目基于 Ebnerd 数据集（支持 Small/Large 规模），实现并工程化了 DIVAN (Deep Interactive and Visual Attention Network) 新闻推荐模型。
项目不仅复现了以 DIN (深度兴趣网络) 和 PopNet (流行度网络) 为双塔核心的算法架构，更在工程实现上进行了极限压榨：重构了 GPU 显存级特征查表、引入了 Polars 极速数据处理引擎，并搭建了无死锁的 CPU 全核并发评估管线，实现了一套极度稳定、支持上万 Batch Size 的大厂级训练框架。

---

## 2. 数据处理与极速流转管线 (Data Pipeline)

为了打破传统的 `DataLoader` CPU-GPU 搬运瓶颈，本项目采用了“离线构建 + 显存常驻 + 极简打包”的三步走策略：

### 2.1 离线特征库 (Offline Vault)

* **多模态降维**：
对 RoBERTa 文本向量和 Image 视觉向量进行 PCA 降维（默认 64 维），拼接后持久化为 `.npy` 矩阵。
* **Polars 原生负采样**：摒弃低效的 Python `for` 循环，使用 Polars 底层 `list.eval(pl.element().shuffle()).list.head()` 在 C++ 层面完成同一次曝光下的新闻负采样。

### 2.2 显存级查表引擎 (GPUFeatureCache) 机制：
将几十万篇文章的属性字典、几十万用户的历史点击序列字典，直接转化为 `torch.Tensor` 并通过 `register_buffer` 注册到 GPU 显存中。

* **收益**：`EbnerdDataset` 每次只需吐出最基础的 ID（如 User ID, Target ID）和时间戳，模型内部的 `FeatureCache` 瞬间在显存内切片拉取所有高维稀疏特征（如 History IDs, Category, Topics），彻底消灭了 CPU 打包大字典的 IO 瓶颈。

---

## 3. 核心算法架构与物理意义

DIVAN 采用动态双路路由架构，每个子网络各司其职。

### 3.1 深度兴趣网络 (DIN Module)

* **输入特征**：候选文章的融合向量（ID + Category + Content），以及用户历史点击文章的融合向量序列。
* **物理意义**：“候选文章与用户的过去口味有多大程度的重合？”DIN 的核心注意力机制会扫描用户的历史记录，遇到与当前候选文章相似的记录，就赋予极高的权重（激活得分），从而精准捕获用户的个性化兴趣。

### 3.2 时效与流行度网络 (PopNet Module)

* **输入特征**：内容多模态向量 (`content_emb`)，以及候选新闻的新鲜度 (`recency_emb`，由当前曝光时间减去发布时间并通过 $\log(1+x)$ 平滑得出)。
* **物理意义**：“这篇文章在全网此时此刻的热度趋势如何？”PopNet 采用了基于 DCN (Deep & Cross Network) 的架构。它通过 Cross Layer 显式地将“内容特征”与“时间衰减特征”进行高维交叉，模型借此学到“哪些类别的新闻（如突发时政）具有极强的短时效性，而哪些（如深度科普）可以长尾发酵”。

### 3.3 动态门控网络 (Gate Module)

* **输入特征**：用户画像向量 (`user_emb`)、内容多模态向量 (`content_emb`)、新闻新鲜度 (`recency_emb`)。
* **输出**：一个 0 到 1 之间的融合权重系数 $\alpha$。
* **物理意义**：扮演一个极其聪明的“裁判”。它会根据当前的用户群体属性和新闻状态做出判断。例如：对于刚发布 5 分钟的重大突发新闻，Gate 会输出极小的 $\alpha$，让系统抛弃个性化，无脑推荐此时的高热度新闻（相信 PopNet）；而对于发布了 3 天的文章，则赋予极大的 $\alpha$，严格依据用户的个性化口味来分发（相信 DIN）。

---

## 4. 工业级工程体系与组件

* **步级雷达 (Intra-epoch Evaluation)**：针对推荐系统极易在单轮内过拟合的痛点，设置 `EVAL_STEP_FREQ`。在 Epoch 中途高频切出计算验证集，精准捕获半山腰的黄金泛化极值点并保存模型快照。
* **全景数据仪表盘 (Metrics Dashboard)**：每次评估后，自动将 Loss、Group AUC、NDCG、MRR 收集进字典，不仅一键渲染为高清三联子图（Loss 曲线、AUC 曲线、排序指标曲线），还将底层数据无损持久化为 `.json` 文件，实现实验的绝对可追溯。 
* **全核并发评估管线**：在 `evaluate` 函数中，对验证集结果切片分组，利用 `joblib.Parallel` 调用物理机全核心并发计算 AUC 与 NDCG，将数十分钟的指标评测压缩至秒级。

---

## 5. 遇到的问题及解决方案 (Troubleshooting)

在项目迭代过程中，我们遭遇并攻克了以下问题：

### 问题 1：数据穿越导致泛化能力暴跌

* **现象**：早期将训练集和验证集的历史记录合并成一个大字典送入特征缓存。导致模型在训练阶段“预知”了用户在验证集（未来）的点击行为，AUC 在训练时极高，验证时暴跌。
* **解决方案**：在 `GPUFeatureCache` 中严格分离 `train_history` 和 `val_history`。重写 `forward`，根据 `self.training` 标志位在底层动态切换查表对象，彻底切断了时间线污染。

### 问题 2：CPU/GPU 假死与 WSL 内存交换陷阱

* **现象**：第一个 Epoch 顺利跑完，进入第二个 Epoch 后 CPU 和 GPU 利用率双双跌至 10% 以下，进度条死锁。
* **原因与解决**：
1. **Joblib 僵尸进程**：`Parallel` 多进程计算完毕后未被销毁，霸占内存导致系统进入 Swap 假死状态。**对策**：改用 `with Parallel(backend='loky') as parallel:` 上下文管理器，强制绞杀完工的子进程。
2. **锁页内存枯竭**：极大的 `TRAIN_BATCH_SIZE (12288)` 配合 `pin_memory=True`，导致操作系统底层无法凑出连续的物理内存。**对策**：果断将 `DataLoader` 的 `pin_memory` 设为 `False`，放弃微小的 IO 加速换取系统绝对稳定。
3. **显存碎片化风暴**：验证集前未清理前向传播产生的零碎显存。**对策**：在 `evaluate` 前强插 `del` 中间变量与 `gc.collect()` 并在 GPU 端执行 `torch.cuda.empty_cache()`。



### 问题 3：心电图式震荡 (学习率与大 Batch Size 适配)

* **现象**：AUC 曲线呈现深 V 震荡，出现悬崖式暴跌（如 0.69 瞬间掉至 0.66）。
* **原因**：为了适配巨大的 Batch Size 调大了学习率，导致优化器步子过大跨过极值山谷（Overshooting）。
* **解决方案**：将 Adam 的学习率压低至 `1e-3`，并引入极其严厉的梯度截断 `clip_grad_norm_(max_norm=1.0)`。此后模型呈现出教科书般平滑上升的收敛曲线。

### 问题 4：精度截断导致的特征失效与报错

* **现象 1**：时间差感知失效。
**对策**：停止将原生 13 位 Unix 时间戳转为 `Float32`（会导致两分钟的漂移误差），全程使用 `torch.long (Int64)` 计算，得出时间差小时数后再转回浮点数。
* **现象 2**：早期开启 AMP（半精度）时出现 `c10::Half without overflow`。
**对策**：将 DIN 网络中 Masking 惩罚的极小值从超出了 Float16 物理极限的 `-1e9` 安全修改为 `-1e4`。

### 问题 5：多线程与硬件底层的“随机性幽灵”

* **现象**：固定了全局随机种子，但同一份代码跑两次，MRR 和 AUC 依然在小数点后三位有浮动。
* **解决方案**：加盖三道“终极封印”以实现 100% 可复现：
1. **环境层**：`OMP_NUM_THREADS=1` 等环境变量锁死 Numpy/PyTorch 底层 C++ 并发数学库。
2. **调度层**：在 `joblib` 的多进程目标函数内实例化独立的局域 `np.random.RandomState`，防止操作系统调度顺序污染随机序列生成。
3. **硬件层**：开启 `torch.backends.cudnn.deterministic = True` 及 `torch.use_deterministic_algorithms(True)`。