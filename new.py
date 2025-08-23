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

students={"name":["alex","bob","james","alice","babu","shashi"],
          "age":[20,30,40,50,60,70],
          "course":["bca","cs","math","economics","english","malayalam"]}
table=pd.DataFrame(students)
print(table.head())
print(table.head(3))
