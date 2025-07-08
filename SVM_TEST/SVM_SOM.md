# 基于SMO的SVM分类器实验报告

## 1.实验概述

### 1.1 实验目标

本次实验旨在从零开始，使用 Python 和 NumPy 实现一个支持向量机 (SVM) 分类器。核心目标是深入理解并实现**序列最小最优化 (Sequential Minimal Optimization, SMO)** 算法，用于求解 SVM 的对偶问题。最终，利用训练好的模型对线性可分数据集进行分类，并可视化其决策边界和支持向量。

### 1.2 实验环境

- **编程语言**: Python 3.10
- **核心库**: NumPy (用于数值计算), Matplotlib (用于数据可视化), tqdm (用于显示进度条)。
- **数据集**: `linear.csv`，一个二维线性可分数据集，包含两个特征 (x1, x2) 和一个二元类别标签 (-1, 1)。

### 1.3 实验流程

1.数据加载与可视化：加载数据集并且直观展示其分布

2.数学原理回顾：SVM的基本原理，对偶问题和SMO算法的核心思想

3.代码实现与解析：详细分析SMO算法的Python代码实现，将代码与数学公式对应

4.模型训练与评估：使用实现的SMO算法模型，得到权重w和偏置b

5.结果可视化：绘制最终的决策边界，标记出支持向量

6.结论总结

## 2.SVM与SMO的算法数学原理

### 2.1 支持向量机（SVM）

SVM 的核心思想是在特征空间中寻找一个**最优超平面**，使得不同类别的样本点到该平面的**间隔 (Margin)** 最大化。对于线性可分数据，该超平面可以表示为： $$ \mathbf{w}^T\mathbf{x} + b = 0 $$ 其中 $\mathbf{w}$ 是法向量，决定了超平面的方向；$b$ 是偏置项，决定了超平面与原点的距离。

为了找到这个最优超平面，我们需要求解以下带约束的优化问题（软间隔SVM的**原始问题**）： $$ \min_{\mathbf{w}, b, \xi} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m}\xi_i $$ 约束条件为： $$ y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad i=1, \dots, m $$

- $C$ 是**惩罚参数**，用于权衡“最大化间隔”和“最小化分类错误”。$C$ 越大，模型对误分类的惩罚越重。
- $\xi_i$ 是**松弛变量**，允许某些点不满足间隔大于等于1的条件。

### 2.2 对偶问题

直接求解原始问题比较困难，通常我们通过拉格朗日乘子法将其转换为**对偶问题**： $$ \max_{\alpha} \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}*j) $$ 约束条件为： $$ \sum*{i=1}^{m}\alpha_i y_i = 0, \quad 0 \le \alpha_i \le C, \quad i=1, \dots, m $$

- $\alpha_i$ 是每个样本对应的拉格朗日乘子。
- $K(\mathbf{x}_i, \mathbf{x}_j)$ 是**核函数**，对于线性SVM，它就是内积 $\mathbf{x}_i^T \mathbf{x}_j$。
- **关键点**: 求解 SVM 的过程就变成了在约束条件下，求解能让目标函数最大化的 $\alpha$ 向量。

### 2.3 KKT条件

最优解必须满足 **KKT (Karush-Kuhn-Tucker) 条件**。对于任意样本 $(\mathbf{x}_i, y_i)$，其对应的 $\alpha_i$ 必须满足：

1. **$\alpha_i = 0 \implies y_i f(\mathbf{x}_i) \ge 1$**: 样本在间隔边界外，是“非支持向量”。
2. **$0 < \alpha_i < C \implies y_i f(\mathbf{x}_i) = 1$**: 样本在间隔边界上，是“支持向量”。
3. **$\alpha_i = C \implies y_i f(\mathbf{x}_i) \le 1$**: 样本在间隔边界内部，可能是被误分类的点。

其中 $f(\mathbf{x}_i) = \mathbf{w}^T\mathbf{x}_i + b$。SMO 算法正是利用 KKT 条件作为**优化是否终止的判断依据**。

### 2.4序列最小最优化（SMO）

SMO 算法是一种高效的求解 SVM 对偶问题的迭代算法。它的核心思想是：

1. 每次迭代时，从所有 $\alpha$ 中选择**两个**变量（例如 $\alpha_i$ 和 $\alpha_j$）。
2. **固定**其他所有 $\alpha$ 变量，将目标函数看作仅关于这两个变量的二次函数。
3. 对这个二次函数求极值，得到更新后的 $\alpha_i$ 和 $\alpha_j$。
4. 重复此过程，直到所有 $\alpha$ 都满足 KKT 条件为止。

## 3.代码实现与核心逻辑分析

### 3.1 数据导入以及数据预处理

```PytHon
import numpy as np
import matplotlib.pyplot as plt      
from matplotlib.colors import ListedColormap
from tqdm import tqdm, trange

data=np.loadtxt('linear.csv', delimiter=',')
print("数据集为：",data.shape)
x=data[:,0:2]
y=data[:,2]
#print(x.shape, y.shape)

#数据集可视化
plt.figure(figsize=(8, 6))
plt.scatter(x[y==-1,0], x[y==-1,1], color='red', label='Class of -1')
plt.scatter(x[y==1,0], x[y==1,1], color='blue', label='Class of 1')
plt.title('Data Visualization')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
```

1.数据导入的选择 可以使用numpy导入处理，也可以使用pandas导入处理。若类型numpy处理不了，则统一采用pandas对应的dateframe处理。

2.引入tqdm进度条 ，便于及时可视化处理结果。

3.在绘制点的时候，X[Y==-1,0]选择了所有y值为-1的点的第一个feature 便于快速选择所需的序列

### 3.2 SMO模型的定义与处理

```PytHon
def SMO(x, y, ker, C, max_iter, tol=1e-3):
    # 参数含义
    # x: 输入特征
    # y: 标签
    # ker: 核函数
    # C: 惩罚参数
    # max_iter: 最大迭代次数
    # tol: 容忍度
    
    # 初始化参数
    m = x.shape[0]
    alpha = np.zeros(m)
    b = 0.0
    
    # 预计算核矩阵
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = ker(x[i], x[j])
            
    for _ in trange(max_iter, desc="SMO Iteration"):
        alpha_changed_count = 0
        for i in range(m):
            # 计算预测值 f(xi) 和误差 Ei
            # f(xi) = sum(alpha_k * y_k * K(xk, xi)) + b
            f_xi = np.sum(alpha * y * K[:, i]) + b
            Ei = f_xi - y[i]
            
            # 检查 alpha[i] 是否违反 KKT 条件
            if (y[i] * Ei < -tol and alpha[i] < C) or \
               (y[i] * Ei > tol and alpha[i] > 0):
                
                # 随机选择 j != i
                j = np.random.choice([l for l in range(m) if l != i])
                
                # 计算预测值 f(xj) 和误差 Ej
                f_xj = np.sum(alpha * y * K[:, j]) + b
                Ej = f_xj - y[j]
                
                # 保存旧的 alpha 值
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                # 计算 L 和 H (alpha_j 的裁剪边界)
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                    
                # 计算 eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                # 更新 alpha[j] 并进行裁剪
                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                
                # 如果 alpha[j] 的变化太小，则忽略
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                # 更新 alpha[i]
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                
                # 更新偏置项 b
                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - \
                     y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - \
                     y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                alpha_changed_count += 1
        
        # 如果一轮迭代下来 alpha 没有变化，可以提前终止
        if alpha_changed_count == 0:
            break
            
    return alpha, b
```

参数解析

- `x`: 训练样本特征矩阵，形状为 (m, n)
- `y`: 标签向量，取值为 + 1 或 - 1
- `ker`: 核函数，用于将输入特征映射到高维空间
- `C`: 惩罚参数，控制分类间隔和误分类样本的权衡
- `max_iter`: 最大迭代次数，防止算法无限循环
- `tol`: KKT 条件检查的容忍度

初始化过程

k：对核矩阵预计算所有的样本对之间的核函数值，避免后续重复计算。

alpha：拉格朗日乘子向量，初始化为0

b：超平面偏离原点的距离

主循环迭代

```PytHon
for _ in trange(max_iter, desc="SMO Iteration"):
    alpha_changed_count = 0
    for i in range(m):
        # 计算预测值 f(xi) 和误差 Ei
        # f(xi) = sum(alpha_k * y_k * K(xk, xi)) + b
        f_xi = np.sum(alpha * y * K[:, i]) + b
        Ei = f_xi - y[i]
        
        # 检查 alpha[i] 是否违反 KKT 条件
        if (y[i] * Ei < -tol and alpha[i] < C) or \
           (y[i] * Ei > tol and alpha[i] > 0):
```

外层循环：控制最大的迭代次数 防止模型训练时发生过拟合

内层循环：遍历所有的样本，寻找违反KKT条件的样本

计算预测值与误差

```
f_xi = np.sum(alpha * y * K[:, i]) + b
        Ei = f_xi - y[i]
```

E_i：误差 用于后续的优化决策

选择第二个变量并且固定其余变量进行优化

```PytHon
 # 随机选择 j != i
                j = np.random.choice([l for l in range(m) if l != i])
                
                # 计算预测值 f(xj) 和误差 Ej
                f_xj = np.sum(alpha * y * K[:, j]) + b
                Ej = f_xj - y[j]
                
                # 保存旧的 alpha 值
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                # 计算 L 和 H (alpha_j 的裁剪边界)
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
```

变量选择：从所有的特征中，选择只要满足二者不用即可

- 边界计算
  - `L`和`H`是 alpha [j] 更新后的取值边界，确保满足约束条件 0 ≤ alpha ≤ C 和 sum (alpha_i * y_i) = 0
  - 根据标签 y [i] 和 y [j] 是否相同，采用不同的公式计算边界

更新拉格朗日乘法算子

```PytHon
 # 计算 eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                # 更新 alpha[j] 并进行裁剪
                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                
                # 如果 alpha[j] 的变化太小，则忽略
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                # 更新 alpha[i]
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
```

- **eta 计算**：二次规划问题的二阶导数，决定更新步长
- **alpha [j] 更新**：根据误差和 eta 值更新 alpha [j]，并裁剪到 [L, H] 范围内
- **alpha [i] 更新**：保证约束条件 sum (alpha_i * y_i) = 0 的满足

更新偏置项b

```PytHon
# 更新偏置项 b
                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - \
                     y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - \
                     y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                alpha_changed_count += 1
        
        # 如果一轮迭代下来 alpha 没有变化，可以提前终止
        if alpha_changed_count == 0:
            break
            
    return alpha, b
```

- 偏置项更新

     ：根据更新前后的 alpha 值和误差，计算新的偏置项 b

  - 如果 alpha [i] 在边界内 (0 < alpha [i] < C)，则使用 b1
  - 如果 alpha [j] 在边界内 (0 < alpha [j] < C)，则使用 b2
  - 否则取两者的平均值

- **终止条件**：如果一轮迭代中没有更新任何 alpha，则算法收敛并提前终止

### 3.3 设置超参数

```PytHon
#设置超参数
C=1.0 # 使用一个更常规的C值
max_iter=1000
np.random.seed(0)
```

### 3.4 训练并且分析结果

```PytHon
alpha, b = SMO(x, y, ker=np.inner, C=C, max_iter=max_iter)

sup_idx=alpha > 1e-5
print("支持向量的索引：", np.where(sup_idx)[0])
print("支持向量的个数：", np.sum(sup_idx))

w=np.sum((alpha[sup_idx]*y[sup_idx]).reshape(-1, 1) * x[sup_idx], axis=0)
print("w:", w)
print("b:", b)
print("支持向量机的决策函数：w*x + b = 0")

# 创建一个 x1 坐标范围
X = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 100)
# 根据决策边界方程 w0*x1 + w1*x2 + b = 0 计算对应的 x2 (即 Y)
Y = (-w[0] * X - b) / (w[1] + 1e-5)  # 避免除以0

plt.figure(figsize=(8, 6))
plt.scatter(x[y==-1,0],x[y==-1,1],color='red',label='Class of -1' )
plt.scatter(x[y==1,0],x[y==1,1],color='blue',label='Class of 1')
plt.plot(X,Y,color='black',label='Decision Boundary')

#标记出支持向量
plt.scatter(x[sup_idx,0],x[sup_idx,1],marker='o',color='none',edgecolors='purple',s=150,label='Support Vectors')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
```

sup_idx是一个布尔数组 起到标记哪些向量时支持向量

阈值选择很重要：过小可能漏掉真正的支持向量，过大可能错误包含非支持向量

计算权重向量w

```PytHon
w = np.sum((alpha[sup_idx]*y[sup_idx]).reshape(-1, 1) * x[sup_idx], axis=0)
print("w:", w)
print("b:", b)
print("支持向量机的决策函数：w*x + b = 0")
```

- 权重向量计算

  - 只使用支持向量计算权重向量`w`
  - `(alpha[sup_idx]*y[sup_idx])`是每个支持向量的系数
  - `.reshape(-1, 1)`将系数转换为列向量，以便与特征矩阵相乘
  - `x[sup_idx]`是支持向量的特征矩阵
  - 沿 axis=0 求和得到最终的权重向量`w`

  计算决策边界

  ```python
  # 创建一个 x1 坐标范围
  X = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 100)
  # 根据决策边界方程 w0*x1 + w1*x2 + b = 0 计算对应的 x2 (即 Y)
  Y = (-w[0] * X - b) / (w[1] + 1e-5)  # 避免除以0
  ```


- 决策边界计算

  ：

  - 在特征 x1 的取值范围内生成 100 个均匀分布的点
  - 根据决策边界方程`w0*x1 + w1*x2 + b = 0`解出 x2：`x2 = (-w0*x1 - b)/w1`
  - 为避免除以 0，在分母上加一个很小的数`1e-5`

可视化结果：

```PytHon
plt.figure(figsize=(8, 6))
plt.scatter(x[y==-1,0], x[y==-1,1], color='red', label='Class of -1')
plt.scatter(x[y==1,0], x[y==1,1], color='blue', label='Class of 1')
plt.plot(X, Y, color='black', label='Decision Boundary')

# 标记出支持向量
plt.scatter(x[sup_idx,0], x[sup_idx,1], marker='o', color='none', edgecolors='purple', s=150, label='Support Vectors')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
```

- 可视化步骤
  - 创建 8×6 英寸的图形
  - 绘制负类样本 (红色) 和正类样本 (蓝色)
  - 绘制决策边界 (黑色直线)
  - 用紫色圆圈标记支持向量 (空心圆圈，突出显示)
  - 添加坐标轴标签、图例和网格



## 4.实验结果分析

实验开始前的原始图像

![75154051618](C:\Users\王旭帆\AppData\Local\Temp\1751540516188.png)



实验开始后的结果以及图像为：

![75154056481](C:\Users\王旭帆\AppData\Local\Temp\1751540564811.png)



