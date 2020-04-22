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

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday


# In[5]:


print(f'O dataset Black Friday possui {black_friday.shape[0]} linhas e {black_friday.shape[1]} colunas.')


# In[6]:


black_friday.head()


# In[7]:


black_friday.info()


# In[8]:


black_friday.describe()


# In[9]:


black_friday.isnull().sum()


# In[10]:


black_friday.pivot_table(black_friday, index = ['Marital_Status'], aggfunc = ['mean'])


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    shape = black_friday.shape
    return shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    women = black_friday['Gender'] == 'F'
    age_between_26_35 = black_friday['Age'] == '26-35'
    result = black_friday[women & age_between_26_35]['User_ID'].count()
    return int(result)
  


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    unique_counts = black_friday['User_ID'].nunique()
    return unique_counts


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    types_count = black_friday.dtypes.nunique()
    return types_count


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    total_values = black_friday.shape[0]
    without_nan_values = black_friday.dropna().shape[0]
    result = (total_values - without_nan_values) / total_values
    return result


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 6.
    max_null_value = black_friday.isnull().sum().max()
    return int(max_null_value)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 7.
    prod_cat_3 = black_friday['Product_Category_3'].dropna()
    mode_freq = prod_cat_3.mode()
    return float(mode_freq)


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    # Retorne aqui o resultado da questão 8.
    purchase = black_friday['Purchase']
    purchase_min = purchase.min()
    purchase_max = purchase.max()
    purchase_normalization = ( purchase - purchase_min ) / (purchase_max - purchase_min) 
    purchase_normalization_mean = purchase_normalization.mean()
    return float(purchase_normalization_mean)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Retorne aqui o resultado da questão 9.
    purchase = black_friday['Purchase']
    purchase_mean = purchase.mean()
    purchase_std = purchase.std()
    purchase_standardization = (purchase - purchase_mean) / purchase_std
    result = ((purchase_standardization <= 1) & (purchase_standardization >= -1)).sum()
    return int(result)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    po_cat_2_null = black_friday['Product_Category_2'].isnull()
    po_cat_3_null = black_friday['Product_Category_3'].isnull()
    compare = po_cat_2_null == po_cat_3_null 
    return (True in compare)

