import numpy as np
# a=np.arange(9)

# c=a.reshape((3,3))
# print(c)

# arr=np.array([2,5,10,100])
# print(np.mean(arr))
# print(np.median(arr))
# print(np.std(arr))

# a=np.random.rand()
# print(a)

# b=np.random.rand(100)
# print(b)

# c=np.random.rand(5,5)
# print(c)


# a = np.random.randint(1,10)
# print(a)

# y=np.random.randint(0,100,size=5)
# print(y)

# z=np.random.randint(0,100,size=(5,5))
# print(z)

# otp=np.random.randint(1000,9999)
# print(otp)

import pandas as pd
# s=pd.Series(["james","bob","alex","alice"])
# print(s)

students={"name":["alex","bob","james",None,"babu","shashi"],
          "age":[20,30,40,50,60,70],
          "course":["bca","cs","math","economics","english",np.nan]}
table=pd.DataFrame(students)
# print(table.head())
# print(table.head(3))
# print(table.tail())
# print(table.tail(3))

# print(table.info())
# print(table.describe())
# print(table.columns)
# print(table.shape)
# table=pd.DataFrame(students,index=["a","b","c","d","e","f"])
# print(table)
# print(table.loc['b'])
# print(table.loc['b','name'])
# print(table.loc[:,["name","city"]])
# print(table.iloc[0])
# print(table.iloc[1,0])
# print(table.iloc[:, 0:2])
print(table.dropna())