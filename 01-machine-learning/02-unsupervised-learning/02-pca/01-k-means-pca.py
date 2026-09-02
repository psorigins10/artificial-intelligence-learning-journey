import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("../../04-data-science/data-bases/pokemon.csv")

X = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

model = KMeans(
    n_clusters = 5,
    random_state = 42,
    n_init = "auto"
)

labels = model.fit_predict(X)

print(labels)


pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters after PCA")

plt.show()