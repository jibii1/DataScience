# cart=["sugar", "flour", "eggs", "milk", "butter"]
# print(cart[0])

# print(cart[1:3])

# cart.append("rice")
# print(cart)

# if "flour" in cart:
# print("item present cart")


# if "oil" in cart:
# print("item present cart")

# else:
# print("item not present cart")


# cart.remove("eggs")
# print(cart)

# nonveg=["chicken","beef","mutton"]

# a = cart + nonveg
# print(a)


# cart=[{}]

# while True:
# print("1.ADD 2.VIEW 3.EXIT ")
# choice=input("Enter your choice : ")
    
# if choice=="1":
# ADD = input("Enter item to add:").strip()
# if ADD:
# cart.append(ADD)
# print("added sucessfully")
# else :
# print("no item sepcified")

# elif choice=="2":
# for i,cart in enumerate(cart,start=1):
# print(i,cart)

# elif choice=="3":
# print("Exicited from Cart")
# break        
   

# cart=[]
# n=int(input("Enter the No.of items you want to add:"))

# for i in range(n):

#     item=str(input("Enter the name of item :"))
#     price=float(input("enter the price : "))
#     quantity=str(input("enter the quantity:"))
#     if price>5:
#         discount=price*0.2
#         price=price-discount
#     n=cart.append({"item":item,"price":price,"quantity":quantity})
#     print(cart)


# d1_attendees = {"alice", "bob", "charlie"}
# d2_attendees = {"bob", "david", "edward"}

# both=d1_attendees.intersection(d2_attendees)
# print(both)

# print(d1_attendees)
# print(d2_attendees)
# unique=d1_attendees.union(d2_attendees)
# print(unique)

python={"ajay","akshay","arjun"}
datascience={"alice","bob","arjun","charlie"}
ML={"bob","david","edward","arjun"}

all=python.intersection(datascience,ML)
print("Student with all certificate:",all)

print("Student with Python certificate:",python)

py_ds=python.symmetric_difference(datascience)
print("Student with either Python or Data Science certificate:",py_ds)

any=python.union(datascience,ML)
print("Student with any certificate:",any)

python.add("frank")
print("new Python students:",python)


for s in [python, datascience, ML]:
    s.discard("bob")
