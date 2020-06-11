#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


bf = black_friday


# In[ ]:





# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[6]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return sum(bf[bf['Age'] == '26-35']['Gender'] == 'F')
    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return bf['User_ID'].unique().shape[0]
    


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return bf.dtypes.nunique()
    


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[9]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return black_friday.isnull().sum().max()/(black_friday.shape[0])
    


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[19]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return bf.isnull().sum().max()
    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[17]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(bf['Product_Category_3'].dropna().mode())
    


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[12]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return ((bf['Purchase']-bf['Purchase'].min())/(bf['Purchase'].max()-bf['Purchase'].min())).mean()
    


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[13]:


def q9():
    # Retorne aqui o resultado da questão 9.
    bf_std = (bf['Purchase']-bf['Purchase'].mean())/bf['Purchase'].std()
    return int(bf_std.between(-1,1).sum())
    


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[14]:


def q10():
    # Retorne aqui o resultado da questão 10.
    arr1 = bf['Product_Category_2'].isnull()
    arr2 = bf['Product_Category_3'].isnull()
    return arr1.equals(arr2)
    


# In[20]:


q9()


# In[ ]:




