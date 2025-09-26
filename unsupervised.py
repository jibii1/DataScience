import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

customer_data = pd.read_csv("Mall_Customers.csv")

print(customer_data.head())


X=customer_data.iloc[:,[3,4]].values
print(X)

wcss=[]


for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

  
sns.set()
plt.plot(range(1,11),wcss)
plt.title("elbow graph")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()

clusters = 5

kmeans = KMeans(n_clusters = 5,init = "k-means++",random_state = 0)

Y = kmeans.fit_predict(X)

print(Y)

clusters=0,1,2,3,4
plt.scatter(x[Y==0,0],x[Y==0,1],s=50,c='blue',label='Cluster 1')
plt.scatter(x[Y==1,0],x[Y==1,1],s=50,c='green',label='Cluster 2')
plt.scatter(x[Y==2,0],x[Y==2,1],s=50,c='pink',label='Cluster 3')
plt.scatter(x[Y==3,0],x[Y==3,1],s=50,c='black',label='Cluster 4')
plt.scatter(x[Y==4,0],x[Y==4,1],s=50,c='orange',label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red',label='centroids')
plt.title('Customer Group')
plt.xlabel('Annual Income')
plt.show()