# -*- coding:utf-8 -*- 
import os
import random

result = open("data/input2.txt", "w")
mergeFileDir = os.getcwd() + "/data"
fns = os.listdir(mergeFileDir)
lines = []

lineItem = open("data/LINEITEM.txt", "r")
orders = open("data/ORDERS.txt", "r")
customer = open("data/CUSTOMER.txt", "r")

content = lineItem.readline()
while content:
    lines.append("+" + "\t" + content)
    content = lineItem.readline()
content = orders.readline()
while content:
    lines.append("+" + "\t" + content)
    content = orders.readline()
content = customer.readline()
while content:
    lines.append("+" + "\t" + content)
    content = customer.readline()

random.shuffle(lines)

zone = 1000
region = 50
total = len(lines)
times = (int)(total/zone)

# [1000, 2050, 3100...]
for i in range(1, times+1):
    row_number = zone * i + region * (i-1)
    print("execute add deleted tuples, start_row_number=", row_number, "random index=", row_number-zone+random.randint(0, zone))
    for i in range(region):
            lines.insert(row_number + i, "-" + "\t" + lines[row_number-zone+random.randint(0, zone)][2:])

        

total = len(lines)
final_lines = []
times = (int)(total/zone)
# [1000, 2051, 3102...]
for i in range(1, times+1):
    row_number = zone * i + region * (i-1) + (i-1)
    lines.insert(row_number, "Bye\n")
    print("row_number=", row_number, "insert Bye")

result.writelines(lines)


result.close()
