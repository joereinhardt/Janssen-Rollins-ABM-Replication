# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:05:58 2016

@author: Joe
"""
import math
import matplotlib.pyplot as plt

eta_values = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
#Generated CPR for a given sum of investments:
def generated_resource(sum_of_contributions):
    if sum_of_contributions >= 0 and sum_of_contributions <= 1:
       produced_resource = 0
    elif sum_of_contributions >1 and sum_of_contributions <= 1.5:
       produced_resource = 0.5
    elif sum_of_contributions >1.5 and sum_of_contributions <= 2:
       produced_resource = 2
    elif sum_of_contributions >2 and sum_of_contributions <= 2.5:
       produced_resource = 4
    elif sum_of_contributions >2.5 and sum_of_contributions <= 3:
       produced_resource = 6
    elif sum_of_contributions >3 and sum_of_contributions <= 3.5:
       produced_resource = 7.5
    elif sum_of_contributions >3.5 and sum_of_contributions <= 4:
       produced_resource = 8.5
    elif sum_of_contributions >4 and sum_of_contributions <= 4.5:
       produced_resource = 9.5
    elif sum_of_contributions >4.5 and sum_of_contributions <= 5:
       produced_resource = 10
    return produced_resource
    
def investment_p(alpha, beta, eta, xj, yj):
    u_list = []
    for x in range(0, 11, 1):
       wi = 1 - x/10 + yj * generated_resource(x/10 + xj)
       wavg = 1 - 0.2 * x/10 - 0.2 * xj + 0.2 * generated_resource(x/10
           + xj)
       u_of_x = wi - alpha * max(wi - wavg, 0) + beta * max(wavg - wi,
           0)
       u_list.append(u_of_x)
    p_list = []
    denominator = 0
    for u in u_list:
       denominator += math.exp(eta * u)
    for u in u_list:
       numerator = math.exp(eta * u)
       p_list.append(numerator / denominator)
    return p_list
#Values of possible investments:
xs = []
for j in range(0, 11, 1):
    xs.append(j/10)
#stores expected investment values for given xj, yj, and eta values:
exp_list1 = []
exp_list2 = []
#looping through the defined values for eta:
for b in eta_values:
#looping through all possible values for xj: between 0 and 4 in 0.1
#increments:
    for z in range(0, 41, 1):
# looping through possible yj values from 0 to 1 in 0.01 increments:
       for g in range(0, 101, 1):
 # get probabilities for all 10 investment levels:
           prob_list1 = investment_p(-1, -1, b, z/10, g/100)
           prob_list2 = investment_p(1, 1, b, z/10, g/100)
 # calculating the expected value:
           exp_list1.append(sum([a*b for a,b in zip(prob_list1, xs)]))
           exp_list2.append(sum([a*b for a,b in zip(prob_list2, xs)]))
#Code Below: Visualization
yj_values = []
for s in eta_values:
    sublist = []
    for t in range(0, 41, 1):
       subsublist = []
       for q in range(0, 101, 1):
           subsublist.append(s)
       sublist.append(subsublist)
    yj_values.append(sublist)
plt.scatter(yj_values, exp_list1, color = "blue")
plt.title("Alpha: " + str(-1) + " Beta: " + str(-1))
plt.ylim(0, 1)
plt.yticks(xs)
plt.xscale("log", basex = 2)
plt.xticks(eta_values)
plt.xlabel("eta")
plt.ylabel("Range of possible exptected values for x")
plt.show()

plt.scatter(yj_values, exp_list2, color = "blue")
plt.title("Alpha: " + str(1) + " Beta: " + str(1))
plt.ylim(0, 1)
plt.yticks(xs)
plt.xscale("log", basex = 2)
plt.xticks(eta_values)
plt.xlabel("eta")
plt.ylabel("Range of possible exptected values for x")
plt.show()