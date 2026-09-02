import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

line_style1 = dict(marker=".", 
                  markerfacecolor="black", 
                  markersize=10,
                  markeredgecolor="black",
                  linestyle="dashed",
                  linewidth=2
                  )

line_style2 = dict(marker="o", 
                  markerfacecolor="#1cd3fc", 
                  markersize=10,
                  markeredgecolor="black",
                  linestyle="solid",
                  linewidth=3
                  )


x = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026])
y1 = np.array([15, 20, 34, 67, 57, 88, 45])
y2 = np.array([0, 23, 42, 52, 67, 98, 10])

# Graphs
# plt.plot(x, y1, **line_style1)
# plt.plot(x, y2, **line_style2)

# Bar Chart
# plt.bar(x, y1)

# Pie Chart
plt.pie(y1, labels=x, autopct="%1.1f%%")

plt.xlabel("Year", fontweight="bold", fontsize=12)
plt.ylabel("Sales", fontweight="bold", fontsize=12)
plt.title("Sales Graph", fontweight="bold", fontsize=20)
# plt.grid(linestyle="dotted")

plt.show()