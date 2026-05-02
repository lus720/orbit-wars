#!/usr/bin/env python3
"""
绘制舰队速度与兵力关系图。
公式: speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
"""

import math
import matplotlib.pyplot as plt
import numpy as np

# 常数定义（从 submission.py 中提取）
MAX_SPEED = 6.0

def fleet_speed(ships):
    """根据舰队船数估算速度"""
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


# 生成数据（0-200 艘船的范围）
ships_values = np.linspace(0, 200, 500)  # 从 0 到 200 的线性均匀分布
# 处理 ships=0 的情况
ships_values = np.where(ships_values == 0, 1, ships_values)
speed_values = [fleet_speed(s) for s in ships_values]

# 创建图像
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制曲线
ax.plot(ships_values, speed_values, 'b-', linewidth=2.5, label='Fleet Speed')
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Min Speed (1.0)')

# 设置 x 轴范围和网格
ax.set_xlim(0, 200)
ax.set_xlabel('Ships Count', fontsize=13, fontweight='bold')
ax.set_ylabel('Speed (units/turn)', fontsize=13, fontweight='bold')
ax.set_title('Fleet Speed vs Ship Count (0-200 Ships)', fontsize=14, fontweight='bold')

# 更细致的 y 轴网格：主网格线每 0.2，次网格线每 0.1
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.grid(True, which='major', alpha=0.4, linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linewidth=0.4, linestyle=':')
ax.legend(fontsize=11, loc='lower right')

# 标注重要的点
key_ships = [1, 5, 10, 20, 50, 100, 150, 200]
for ships in key_ships:
    speed = fleet_speed(ships)
    ax.plot(ships, speed, 'ro', markersize=7)
    ax.annotate(f'{ships}\n{speed:.3f}', xy=(ships, speed), 
                xytext=(8, 8), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))


plt.tight_layout()

# 保存图像
output_path = '/home/y7000p/orbit-wars/fleet_speed_chart.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ 图像已保存到: {output_path}")

# 生成数据表（0-200 范围）
print("\n舰队速度对照表（0-200 艘船）：")
print("=" * 60)
print(f"{'Ships':<15} {'Speed':<15} {'Ratio %':<15} {'vs 1船速度比':<15}")
print("=" * 60)
base_speed = fleet_speed(1)
for ships in [1, 5, 10, 15, 20, 30, 50, 75, 100, 125, 150, 175, 200]:
    speed = fleet_speed(ships)
    ratio_pct = (speed - 1.0) / (MAX_SPEED - 1.0) * 100
    speed_multiplier = speed / base_speed
    print(f"{ships:<15} {speed:<15.4f} {ratio_pct:<15.1f}% {speed_multiplier:<15.2f}x")
print("=" * 60)

# 显示图像
plt.show()
