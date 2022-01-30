import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('data.csv')
x = df.iloc[:,[0,1]].values
kmenas = KMeans(n_clusters=3,init="k-means++", random_state=42)
kmenas.fit(x)

plt.figure(figsize=(15,7))
plt.xlabel("Patel Size")
plt.ylabel('Sapel size')
plt.title("Clusters of Flower")
plt.grid(False)

y_kmeans = kmenas.fit_predict(x)

sns.scatterplot(x[y_kmeans == 0,0] ,x[y_kmeans == 0,1], color= "yellow" , label="Cluster One")
sns.scatterplot(x[y_kmeans == 1,0] ,x[y_kmeans == 1,1], color= "red" , label="Cluster Two")
sns.scatterplot(x[y_kmeans == 2,0] ,x[y_kmeans == 2,1], color= "blue" , label="Cluster Three")
sns.scatterplot(kmenas.cluster_centers_[:,0] , kmenas.cluster_centers_[:,1] , color="green" , label="Centroid" , s=100, marker=",")

plt.show()


print(x)
# fig = px.scatter(df,x="petal_size",y="sepal_size")
# fig.show()
