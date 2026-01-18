# Pi/LeRobot 下基于 Segment Episode 的 Sweep 自动切分方案（方案一）

> 本文档整理并固化讨论得到的 **方案一：不改采样/不改隐式滑窗** 的切分策略。  
> 目标是：在 **Pi/LeRobot 的“一帧一个样本、步长=1、action horizon=H”** 的训练范式下，通过选择每个 segment episode 的边界 \((T_{t,0}, T_{t,1})\)，使得段内尽可能多的 window start（即 idx）产生“干净的单次 sweep 样本”，同时不跨到相邻 sweep。

---

## 0. 背景与关键约束

- 数据读取：Pi 系列使用 LeRobot 数据集，并通过 `delta_timestamps` 隐式实现 action chunk（长度 \(H\)）的滑动窗口。**每一帧 idx 都是一个训练样本起点**，步长恒为 1。
- episode 边界：LeRobot 在 `__getitem__` 中对超出 episode 的 query index 做 clamp，并提供 `*_is_pad` 标记；**不会跨 episode 取未来帧**。
- 我们的目标任务：对“扫字母（sweep）”动作进行原子化切分。一个 sweep 在时间上可分为四个阶段：  
  1. Approach（接近/下压前的进入阶段）  
  2. Engage（建立接触）  
  3. Stroke（有效扫动）  
  4. Retreat/Release（脱离/抬起退出）  
- 已知经验：在 fps 固定条件下，**Engage+Stroke 的时间长度约为 21 帧**；但我们认为 \(H=30\) 足以覆盖四阶段（Approach+Engage+Stroke+Retreat）。

---

## 1. Notation（符号约定）

- \(H\)：action horizon（action chunk 长度），本项目固定 \(H=30\)。
- \(t\)：第 \(t\) 次 sweep 的索引（按时间顺序）。
- \(P_{t,0}\)：第 \(t\) 次 sweep 的 **Engage 开始**（阶段 2 起点）的帧索引。
- \(P_{t,1}\)：第 \(t\) 次 sweep 的 **Stroke 结束**（阶段 3 终点）的帧索引。
- \(s\)：window start（训练样本起点），即 LeRobot 的 idx（帧索引）。
- 长度为 \(H\) 的动作窗口：  

\[
W(s) = [s,\ s+1,\ \dots,\ s+H-1]
\]

- \(A_{\min}\)：窗口中至少包含的 Approach 帧数下界（保证窗口左侧有足够进入相位）。
- \(R_{\min}\)：窗口中至少包含的 Retreat 帧数下界（保证窗口右侧有足够退出相位）。
- \(\mathcal{S}_t\)：第 \(t\) 次 sweep 的 **合格 window start 集合**（希望尽量大），其整数区间形式为  

\[
\mathcal{S}_t = \{s\in \mathbb{Z} \mid s_{\min}\le s\le s_{\max}\}
\]

- segment episode 的边界（我们最终要切出来的段）：  
  - \(T_{t,0}\)：第 \(t\) 个 segment 的起点帧索引  
  - \(T_{t,1}\)：第 \(t\) 个 segment 的终点帧索引  

---

## 2. 原理：为什么必须从 window start 约束反推 segment 边界

### 2.1 Pi/LeRobot 的训练样本语义

在 LeRobot 中，每个 idx（帧）都会返回：  
- 当前观测：\(o_s\)（包括图像、关节状态等）  
- 动作 chunk：\((a_s, a_{s+1}, \dots, a_{s+H-1})\)（通过 `delta_timestamps` 查询未来动作）

因此，**多样性不是 segment 起点的多样性**，而是：

> 在同一个 segment 内，有多少个不同的 idx 能作为 window start 产生合格的动作窗口 \(W(s)\)。

即我们希望尽可能多的 \(s\) 落在 \(\mathcal{S}_t\) 中。

### 2.2 方案一的核心含义（确定方案）

我们确定使用方案一：  
- 不修改 LeRobot 的隐式滑窗（步长=1）。  
- 不修改 Pi 的训练逻辑（每帧一个样本）。  
- 只通过选择每个 segment 的 \((T_{t,0},T_{t,1})\) 来影响哪些 idx 会被采样到。  

因此，我们需要一个严谨的边界选择，使得：  
- segment 内合格窗口起点尽量多（\(|\mathcal{S}_t|\) 尽量大）  
- 同时不跨到相邻 sweep（避免一个样本窗口内混入两次 sweep 的 2+3）

---

## 3. 约束建模与公式推导

### 3.1 合格样本的四个约束（对 window start \(s\)）

一个训练样本由 \(s\) 决定，其动作窗口为 \(W(s)=[s,s+H-1]\)。我们希望该窗口：

1) 包含 Engage 起点 \(P_{t,0}\)，并且窗口左侧至少保留 \(A_{\min}\) 帧用于 Approach：  

\[
P_{t,0} - s \ge A_{\min}
\Rightarrow
s \le P_{t,0} - A_{\min}
\]

2) 包含 Stroke 终点 \(P_{t,1}\)，并且窗口右侧至少保留 \(R_{\min}\) 帧用于 Retreat：  

\[
(s+H-1) - P_{t,1} \ge R_{\min}
\Rightarrow
s \ge P_{t,1}-H+1+R_{\min}
\]

3) 不混入上一段 sweep 的 2+3（保守约束）：  

\[
s \ge P_{t-1,1}+1
\]

4) 不混入下一段 sweep 的 2+3（关键：带 \(H\) 的约束）：  

\[
s+H-1 < P_{t+1,0}
\Rightarrow
s \le P_{t+1,0}-H
\]

### 3.2 合并得到 \(\mathcal{S}_t\) 的区间形式

将所有下界与上界合并：  

- 下界：  

\[
s_{\min}=\max\Big(P_{t-1,1}+1,\ \ P_{t,1}-H+1+R_{\min}\Big)
\]

- 上界：  

\[
s_{\max}=\min\Big(P_{t,0}-A_{\min},\ \ P_{t+1,0}-H\Big)
\]

于是：  

\[
\mathcal{S}_t = \{s\in\mathbb{Z}\mid s_{\min}\le s\le s_{\max}\}
\]

若 \(s_{\min}>s_{\max}\)，则不存在合格窗口起点，该 sweep 在当前 \(A_{\min},R_{\min}\) 设定下不可用。

### 3.3 多样性指标的严格定义

因为 idx 步长为 1，\(\mathcal{S}_t\) 内每个整数都对应一个不同训练样本起点，因此合格起点数量为：  

\[
|\mathcal{S}_t| = s_{\max}-s_{\min}+1
\]

---

## 4. 从 \(\mathcal{S}_t\) 反推唯一的 segment 边界 \((T_{t,0},T_{t,1})\)

### 4.1 最小覆盖原则（Minimal Cover）

对任意 \(s\in\mathcal{S}_t\)，窗口是 \([s,s+H-1]\)。为了让所有合格起点 \(s\in[s_{\min},s_{\max}]\) 的窗口都能在同一个 segment 内被取到且不依赖 clamp，我们必须让 segment 至少覆盖区间：  

\[
[s_{\min},\ s_{\max}+H-1]
\]

因此采用最小覆盖的唯一自然选择：  

\[
\boxed{T_{t,0}=s_{\min}},\qquad
\boxed{T_{t,1}=s_{\max}+H-1}
\]

### 4.2 方案一的必然现象：segment 尾部 padding

segment 的末尾 \(H-1\) 个 idx 的窗口右端会超出 segment 边界，LeRobot 将产生 clamp 和 `*_is_pad=True`。只要训练 loss 正确使用 pad mask，该问题主要表现为效率损失。

---

## 5. 框图：如何保证多样性最大化且不跨 sweep

```mermaid
flowchart TD
    A[原始长 episode\n连续帧、步长=1] --> B[运动学特征\n由关节+URDF做FK\n得到 z(t), v_xy(t)]
    B --> C[关键点检测\n得到 (P_{t,0}, P_{t,1})]
    C --> D[区间推导\n计算 s_min, s_max]
    D --> E[合格集合\nS_t={s|s_min<=s<=s_max}\n多样性=|S_t|]
    E --> F[最小覆盖\nT_{t,0}=s_min\nT_{t,1}=s_max+H-1]
    F --> G[切成 segment episodes\n不同 sweep 不同 episode]
    G --> H[Pi/LeRobot训练\n隐式滑窗 delta_timestamps\n不跨episode]
    H --> I[结果\nsegment内尽量多idx属于S_t\n且不跨相邻sweep]
```

---

## 6. 工程实现 Recipe（步骤与注意事项）

> 输入：连续 episode（图像、关节状态、动作），URDF（用于 FK）。  
> 输出：segment episodes 边界 \((T_{t,0},T_{t,1})\) 列表，并导出为新的 LeRobot episodes。

### Step 1：预处理与对齐

1. 对齐图像/关节/动作到同一帧索引。
2. 固定 \(H=30\)，记录 fps。
3. 准备 FK：从关节状态得到末端 \(x(t),y(t),z(t)\)。
4. 对 \(z(t)\)、\(v_{xy}(t)\) 做轻度平滑（5–9 帧滑动均值）。

**注意**：平滑过强会抹平短 stroke 的边界，过弱会造成抖动误检。

### Step 2：检测 sweep 关键点 \((P_{t,0},P_{t,1})\)
推荐“低位区 + 平面高速”代理：
1. 用 \(z(t)\) 的滞回阈值 \(z_{on}<z_{off}\) 找低位区间（近似接触阶段）。
2. 在低位区间内，用 \(v_{xy}(t)\) 超阈值的最长连续段作为 stroke 主体。
3. 定义：  
   - \(P_{t,0}\)：该连续段起点（Engage 开始代理）  
   - \(P_{t,1}\)：该连续段终点（Stroke 结束代理）
4. 质量过滤：\(L_{23}=P_{t,1}-P_{t,0}+1\) 落在合理范围（如 \([15,28]\)），否则标记异常。
5. 相机质检（可选）：ROI 运动能量在 \([P_{t,0},P_{t,1}]\) 内显著高于背景，过滤假阳性。

### Step 3：设定 \(A_{\min}, R_{\min}\)

- \(A_{\min}\)：窗口左侧 Approach 最少帧数  
- \(R_{\min}\)：窗口右侧 Retreat 最少帧数  

建议起步：\(A_{\min}=2\)，\(R_{\min}=2\)。若四阶段不稳定再增到 3。

### Step 4：计算 \([s_{\min}, s_{\max}]\)、\(|\mathcal{S}_t|\)
对每个 sweep \(t\)：
\[
s_{\min}=\max\big(P_{t-1,1}+1,\ P_{t,1}-H+1+R_{\min}\big)
\]
\[
s_{\max}=\min\big(P_{t,0}-A_{\min},\ P_{t+1,0}-H\big)
\]
若 \(s_{\min}>s_{\max}\)：该 sweep 丢弃或调整参数。  
多样性：\(|\mathcal{S}_t|=s_{\max}-s_{\min}+1\)。

### Step 5：确定 segment 边界并导出
最小覆盖：
\[
T_{t,0}=s_{\min},\quad T_{t,1}=s_{\max}+H-1
\]
导出时：
1. segment 之间不重叠（不复制帧）。
2. 重建新的 episode 索引与边界元信息。
3. 不改 `delta_timestamps`。

---

## 7. 推荐默认参数

- \(H=30\)
- \(A_{\min}=2,\ R_{\min}=2\)
- 低位区进入/退出持续帧数 \(m=2\sim 3\)
- \(L_{23}\) 过滤范围 \([15,28]\)

---

## 8. 论文方法表述建议

可将本方案表述为：  
- 在不修改 Pi/LeRobot 训练范式的前提下，通过 episode 分割实现 sweep 级原子段；  
- 将“窗口仅含一次 sweep 的关键相位且包含完整相位结构”的需求形式化为对 window start 的不等式约束；  
- 推导合格起点区间 \([s_{\min},s_{\max}]\)，并用 \(|\mathcal{S}_t|\) 量化可学习多样性；  
- 采用最小覆盖原则设定 segment 边界 \((T_{t,0},T_{t,1})=(s_{\min},s_{\max}+H-1)\)，保证不跨相邻 sweep 且最大化合格样本数。

