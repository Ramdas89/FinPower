#!/usr/bin/env python
# coding: utf-8

# In[16]:


#!pip install mlextend


# In[17]:


import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[18]:


# product_sets = pd.read_csv("C:/NextBestProd/ProductSetDetails.csv")
# product_sets = pd.read_csv("C:/NextBestProd/ProductKeyDetails.csv")
product_sets = pd.read_csv("C:/NextBestProd/ProductWOFC.csv")
product_sets.head(20)


# # Data Preparation 

# In[19]:


product_sets = pd.pivot_table(data=product_sets,index='INDNUM',columns='PRODUCT',values='Quantity',                         aggfunc='sum',fill_value=0)

product_sets.head(10)


# In[20]:


def convert_into_binary(x):
    if x > 0:
        return 1
    else:
        return 0


# In[21]:


product_sets = product_sets.applymap(convert_into_binary)


# # Apply Apriori Algorithm

# In[22]:


#call apriori function and pass minimum support here we are passing 0.1%. means 1/10 times in total number of transaction that product was present.
frequent_productsets = apriori(product_sets, min_support=0.01, use_colnames=True)
frequent_productsets.head(50)


# # Apply Association Rules

# In[23]:


# we have association rules which need to put on frequent productset. 
# here we are setting based on lift and has minimum lift as 1

rules_mlxtend = association_rules(frequent_productsets, metric="lift", min_threshold=0.5)
rules_mlxtend


# In[24]:


# rules_mlxtend.rename(columns={'antecedents':'lhs','consequents':'rhs'})

# as based business use case we can sort based on confidence and lift.

rules_mlxtend[ (rules_mlxtend['lift'] >= 1) & (rules_mlxtend['confidence'] >= 0.1) ]


# In[ ]:




