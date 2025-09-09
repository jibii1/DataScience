import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x=np.linspace(0,10,100)
# y=np.sin(x)
# plt.plot(x,y, color='red',linestyle='--',marker='o')
# plt.title("line-plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()

# x= np.random.rand(50)
# y=np.random.rand(50)
# plt.scatter(x,y,color="green",marker='X')
# plt.title("scatter plot")
# plt.show()


# items=["RICE","SUGAR","OIL"]
# selled_items=[100,200,300]
# plt.bar(items,selled_items,color=["aqua","yellow","orange"])
# plt.title("bar plot")
# plt.show()


# items=["RICE","SUGAR","OIL"]
# selled_items=[100,200,300]
# plt.barh(items,selled_items,color=["aqua","olive","orange"])
# plt.title("bar plot")
# plt.show()

# data = np.random.randn(1000)
# plt.hist(data,bins=30,color="olive",edgecolor="skyblue")
# plt.title("histogram")
# plt.show()

# sizes=[10,20,30,40]
# items=["R","S","O","F"]
# plt.pie(sizes,items=items ,autopct='%1.1f%%',startangle=90)
# plt.title("pie")
# plt.show()

# days=[1,2,3,4,5]
# sleeping=[7,8,9,11,10]
# eating=[4,5,6,7,8]
# working=[7,6,2,3,4]
# playing=[10,5,6,7,8]
# plt.stackplot(days,sleeping,eating,working,playing,labels=["sleep","eat","work","play"])
# plt.legend(loc="upper left")
# plt.title("stacked area plot")
# plt.show()


data=np.random.normal(100,20,600)
plt.boxplot(data)
plt.title("box plot")
plt.show()