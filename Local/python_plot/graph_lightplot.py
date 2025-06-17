import matplotlib.pyplot as plt
import seaborn as sns

# データ定義
processes = [1, 2, 4, 8]
strong_time = [40.01, 13.70, 10.52, 5.65]
weak_time = [10.20, 7.90, 7.82, 25.73]

# スピードアップ計算
strong_speedup = [strong_time[0] / t for t in strong_time]
weak_speedup = [weak_time[0] / t for t in weak_time]

# 理想的なスピードアップ（Amdahl's law）
p = 0.9
#amdhal
ideal_speedup_a = [1 / ((1 - p) + (p / n)) for n in processes]
#gustafson
ideal_speedup_g = [n - (1 - p)*(n - 1) for n in processes]


# スタイル設定
sns.set_style("whitegrid")

def create_scalability_plot(processes, time_data, speedup_data, title, time_color, speedup_color, ideal_speedup):
    """スケーラビリティプロットを作成する関数"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # 実行時間プロット
    ax1.set_xlabel('# Processes')
    ax1.set_ylabel('Execution Time (sec)', color=time_color)
    line1 = ax1.plot(processes, time_data, marker='o', label='Execution Time', color=time_color)
    ax1.tick_params(axis='y', labelcolor=time_color)
    
    # スピードアッププロット
    ax2 = ax1.twinx()
    ax2.set_ylabel('Speedup', color=speedup_color)
    line2 = ax2.plot(processes, speedup_data, marker='s', label='Speedup', color=speedup_color)
    line3 = ax2.plot(processes, ideal_speedup, 'k--', label='Ideal Speedup')
    ax2.tick_params(axis='y', labelcolor=speedup_color)
    
    # Y軸のスケールをスピードアップに合わせて軸を揃える
    speedup_max = max(time_data)
    
    # 両軸をスピードアップのスケールに統一
    scale_factor = 1.1
    ax1.set_ylim(0, speedup_max * scale_factor)
    ax2.set_ylim(0, speedup_max * scale_factor)
    
    # 統合された凡例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.title(title)
    fig.tight_layout()
    plt.show()

# Strong scalabilityプロット
create_scalability_plot(
    processes, strong_time, strong_speedup,
    'Light Cluster - Strong Scalability (Time & Speedup)',
    'tab:blue', 'tab:red',ideal_speedup_a)

# Weak scalabilityプロット
create_scalability_plot(
    processes, weak_time, weak_speedup,
    'Light Cluster - Weak Scalability (Time & Speedup)',
    'tab:orange', 'tab:green', ideal_speedup_g
)