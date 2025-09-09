

import pandas as pd
# df = sns.load_dataset("tips")
# sns.histplot(df["total_bill"],bins=20,kde=True,color="orange")
# plt.title("histogarm+kde")
# plt.show()


# data=pd.DataFrame({
#     "category":["A","B","C","D"],
#     "values":[4,5,7,8]
# })

# sns.barplot(x="category",y="values",color="blue", data=data)
# plt.title("normal bar chart")
# plt.show()

# import seaborn as sns
# df = sns.load_dataset("tips")
# import matplotlib.pyplot as plt
# sns.countplot(x="day",data=df,palette="Set2")
# plt.title("count plot")
# plt.show()



import seaborn as sns
df = sns.load_dataset("tips")
import matplotlib.pyplot as plt
# sns.boxplot(x="day",y="total_bill",data=df,palette="pastel")
# plt.title("box plot")
# plt.show()
sns.scatterplot(x="total_bill",y="tip",data=df,hue="sex",style="time")
plt.title("scatter plot")
plt.show()
    