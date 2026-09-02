import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

df = pd.read_csv("../data-bases/pokemon.csv", index_col = "Name")

generationalGroup = df.groupby("Generation")
average = generationalGroup["Total"].mean()


x = ["Gen1", "Gen2", "Gen3", "Gen4", "Gen5", "Gen6"]
y = np.array([average.iloc[0], average.iloc[1], average.iloc[2], average.iloc[3], average.iloc[4], average.iloc[5]])

plt.bar(x, y)
plt.xlabel("Total")
plt.ylabel("Stats")
plt.title("Strongest Gen",
    fontweight= "bold",
    fontsize=20
)

plt.show()