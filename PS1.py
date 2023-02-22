#!/usr/bin/env python
# coding: utf-8

# ECON 280A
# 
# PS 1
# 
# By Yi-Fan, Lin

# In[1]:


import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, Eq, solve, nsolve
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel("/Users/ricky/Documents/椰林大學/Berkeley/International Econ/Data for PS 1.xls"
                   , sheet_name="trade flows mfg", header=0, nrows=20, usecols="B:T")

df = df[1:]
#exports from i to j


# In[3]:


df.head(5)


# In[4]:


#name of countries
print(df.columns)


# In[149]:


theta = 5 #the median theta as in slides
nc = 19 #number of countries
t_cost = np.ones([nc, nc]) #change in trade costs
prod = np.ones(nc) #change in productivities
labor = np.ones(nc) #change in labors

total_prod = df.sum(axis=1) #total production, Y_n
total_cons = df.sum(axis=0) #total consumption
deficit = [total_cons.iloc[i] - total_prod.iloc[i] for i in range(nc)]
df_share = df.divide(total_cons.iloc[0], axis=0) #pi_n
def_share = deficit/np.array(total_prod)


# In[190]:


def gen_denom(ncoun, w, theta, t_cost):
    #ncoun for the index of country
    
    denom = 0
    for k in range(nc):
        temp = df_share.iloc[ncoun, k]*(w[k]*t_cost[ncoun, k])**(-theta)
        denom = denom + temp
        
    return denom

def obj(w_vec, theta, t_cost):
    eq_vec = [0 for i in range(nc)]
    w_vec = w_vec
    
    for i in range(nc):
        rhs = 0
        for n in range(nc):
            denom = gen_denom(n, w_vec, theta, t_cost)
            rhs = rhs + df_share.iloc[n, i]*(w_vec[i]*t_cost[n, i])**(-theta)*(w_vec[n]*total_cons.iloc[n])/denom
        eq_vec[i] = w_vec[i]*total_cons.iloc[i] - rhs
        
    return eq_vec


# In[151]:


def welfare(ncoun, w, theta, t_cost):
    
    c_share = (w[ncoun]*t_cost[ncoun, ncoun])**(-theta)/gen_denom(ncoun, w, theta, t_cost)
    
    return c_share**(-1/theta)

def output(theta, t_cost):
    
    wage = fsolve(obj, np.ones(nc), (theta, t_cost))
    real_wage = np.array([welfare(n, wage, theta, t_cost) for n in range(nc)])
    price = wage/real_wage
    
    return [wage, real_wage, price]


# In[191]:


base = output(theta, t_cost)


# In[192]:


for i in range(nc):
    print(def_share[i]*100, (base[0])[i], (base[1])[i], (base[2])[i])


# In[154]:


t_cost_dec = t_cost*(1/1.3)
for i in range(nc):
    t_cost_dec[i, i] = 1 #except for own


# In[155]:


#tariff cut

tarcut = output(theta, t_cost_dec)


# In[173]:


for i in range(nc):
    print((df.columns)[i], def_share[i]*100, tarcut[0][i], tarcut[1][i], tarcut[2][i])


# In[159]:


#us-canada FTA
#canada: 3
#us: 18

t_cost_FTA = t_cost
t_cost_FTA[3, 18] = t_cost_FTA[3, 18]*(1/1.3)
t_cost_FTA[18, 3] = t_cost_FTA[18, 3]*(1/1.3)


# In[160]:


tarFTA = output(theta, t_cost_FTA)


# In[174]:


for i in range(nc):
    print((df.columns)[i], def_share[i]*100, tarFTA[0][i], tarFTA[1][i], tarFTA[2][i])


# In[162]:


plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100


# In[179]:


fig = plt.figure()
fig, ax = plt.subplots()

ax.scatter(def_share, base[0], c='blue', label='Relative')
ax.scatter(def_share, base[1], c='green', label='Real')

ax.legend()
plt.ylabel('Change in Wage')
plt.xlabel('Trade deficit')
plt.title('Baseline (graph 1.)')
plt.show()


# In[180]:


fig = plt.figure()
fig, ax = plt.subplots()

ax.scatter(def_share, tarcut[0], c='blue', label='Relative')
ax.scatter(def_share, tarcut[1], c='green', label='Real')

ax.legend()
plt.ylabel('Change in Wage')
plt.xlabel('Trade deficit')
plt.title('Overall Tariff Cut (graph 2.)')
plt.show()


# In[181]:


fig = plt.figure()
fig, ax = plt.subplots()

ax.scatter(def_share, tarFTA[0], c='blue', label='Relative')
ax.scatter(def_share, tarFTA[1], c='green', label='Real')

plt.text(def_share[3], tarFTA[0][3]+0.01, 'CAN')
plt.text(def_share[18], tarFTA[0][18]+0.01, 'USA')

ax.legend()
plt.ylabel('Change in Wage')
plt.xlabel('Trade deficit')
plt.title('US-Canada FTA (graph 3.)')
plt.show()


# In[182]:


fig = plt.figure()
fig, ax = plt.subplots()

ax.scatter(def_share, base[1], c='blue', label='Base')
ax.scatter(def_share, tarcut[1], c='green', label='Tariff cut')
ax.scatter(def_share, tarFTA[1], c='brown', label='FTA')

plt.text(def_share[3], tarFTA[1][3]+0.01, 'CAN')
plt.text(def_share[18], tarFTA[1][18]+0.01, 'USA')

ax.legend()
plt.ylabel('Change in Real Wage')
plt.xlabel('Trade deficit')
plt.title('Comparison (graph 4.)')
plt.show()


# In[183]:


table1 = pd.DataFrame({'Deficit': def_share*100, 'Baseline': base[1], 'Tariff Cut': tarcut[1], 'FTA': tarFTA[1]})
table1.index = df.columns
print("Change in Real wage (Table 1)")
print(table1)


# In[184]:


table2 = pd.DataFrame({'Deficit': def_share*100, 'Baseline': base[0], 'Tariff Cut': tarcut[0], 'FTA': tarFTA[0]})
table2.index = df.columns
print("Change in Relative wage (Table 2)")
print(table2)


# In[185]:


table3 = pd.DataFrame({'Deficit': def_share*100, 'Baseline': base[2], 'Tariff Cut': tarcut[2], 'FTA': tarFTA[2]})
table3.index = df.columns
print("Change in Price index (Table 3)")
print(table3)


# In[193]:


fig = plt.figure()
fig, ax = plt.subplots()

ax.scatter(def_share, base[2], c='blue', label='Base')
ax.scatter(def_share, tarcut[2], c='green', label='Tariff cut')
ax.scatter(def_share, tarFTA[2], c='brown', label='FTA')

plt.text(def_share[3], tarFTA[2][3]+0.01, 'CAN')
plt.text(def_share[18], tarFTA[2][18]+0.01, 'USA')

ax.legend()
plt.ylabel('Change in Price index')
plt.xlabel('Trade deficit')
plt.title('Comparison (graph 5.)')
plt.show()


# In[ ]:




