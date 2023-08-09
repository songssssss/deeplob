# -*- coding:utf-8 -*- 
import os
import random

result = open("data/input.txt", "w")
mergeFileDir = os.getcwd() + "/data"
fns = os.listdir(mergeFileDir)
lines = []

orders = open("data/ORDERS.txt", "r")
customer = open("data/CUSTOMER.txt", "r")
lineItem = open("data/LINEITEM.txt", "r")


content = orders.readline()
while content:
    lines.append("+" + "\t" + content)
    content = orders.readline()
content = customer.readline()
while content:
    lines.append("+" + "\t" + content)
    content = customer.readline()
content = lineItem.readline()
while content:
    lines.append("+" + "\t" + content)
    content = lineItem.readline()

random.shuffle(lines)

sample_lines = lines[:6000]


# for i in range(5):
#     sample_lines.append("-" + "\t" + lines[random.randint(0, prevLine)][2:])

result.writelines(lines)


result.close()
