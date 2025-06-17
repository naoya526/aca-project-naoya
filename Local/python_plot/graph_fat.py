import matplotlib.pyplot as plt
import seaborn as sns

# データ定義
processes = [1, 2, 4, 8]

# Strong scalability
strong_time = [44.02, 11.10, 3.04, 0.88]
strong_speedup = [strong_time[0] / t for t in strong_time]
strong_mem = [1034.58, 1107.22, 1253.28, 1541.04]

# Weak scalability
weak_time = [5.51, 2.78, 1.24, 0.87]
weak_speedup = [weak_time[0] / t for t in weak_time]
weak_mem = [194.39, 387.45, 775.20, 1542.19]

# 理想的なスピードアップ
p = 0.9
ideal_speedup =  [1 / ((1 - p) + (p / n)) for n in processes]

sns.set(style="whitegrid")

# 1. Strong scalability plot with Speedup
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('# Processes')
ax1.set_ylabel('Execution Time (sec)', color=color)
ax1.plot(processes, strong_time, marker='o', label='Execution Time', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
ax2.set_ylabel('Speedup', color='tab:red')
ax2.plot(processes, strong_speedup, marker='s', label='Speedup', color='tab:red')
ax2.plot(processes, ideal_speedup, 'k--', label='Ideal Speedup')  # 黒の点線
ax2.tick_params(axis='y', labelcolor='tab:red')

# 凡例を統合
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper left')

plt.title('Fat Cluster - Strong Scalability (Time & Speedup)')
fig.tight_layout()
plt.show()

# 2. Weak scalability plot with Speedup
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:orange'
ax1.set_xlabel('# Processes')
ax1.set_ylabel('Execution Time (sec)', color=color)
ax1.plot(processes, weak_time, marker='o', label='Execution Time', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
ax2.set_ylabel('Speedup', color='tab:green')
ax2.plot(processes, weak_speedup, marker='s', label='Speedup', color='tab:green')
ax2.plot(processes, ideal_speedup, 'k--', label='Ideal Speedup(p=1)')  # 黒の点線
ax2.tick_params(axis='y', labelcolor='tab:green')

# 凡例を統合
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper left')

plt.title('Fat Cluster - Weak Scalability (Time & Speedup)')
fig.tight_layout()
plt.show()

