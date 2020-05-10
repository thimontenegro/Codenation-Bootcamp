#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[ ]:



from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[ ]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[ ]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# In[ ]:


dataframe.shape


# In[ ]:


dataframe.describe()


# In[ ]:


22.684324 - 23.000000


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[ ]:


q1_norm, q2_norm, q3_norm = dataframe['normal'].quantile((0.25,0.5,0.75))
print(f"Primeiro quantil ou 25% dos dados {q1_norm}")
print(f"Segundo quantil ou 50% dos dados {q2_norm}")
print(f"Terceiro quantil ou 75% dos dados {q3_norm}")


# In[ ]:


q1_binom, q2_binom, q3_binom = dataframe['binomial'].quantile((0.25,0.5,0.75))
print(f"Primeiro quantil ou 25% dos dados {q1_binom}")
print(f"Segundo quantil ou 50% dos dados {q2_binom}")
print(f"Terceiro quantil ou 75% dos dados {q3_binom}")


# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    q1_norm, q2_norm, q3_norm = dataframe['normal'].quantile((0.25,0.5,0.75))
    q1_binom, q2_binom, q3_binom = dataframe['binomial'].quantile((0.25,0.5, 0.75))
    #Calculando as diferenças dos quantis
    q1_diff = q1_norm - q1_binom
    q2_diff = q2_norm - q2_binom 
    q3_diff = q3_norm - q3_binom
    #Formatando para 3 casas decimais
    q1_diff_format = float("{:.3f}".format(q1_diff))
    q2_diff_format = float("{:.3f}".format(q2_diff))
    q3_diff_format = float('{:.3f}'.format(q3_diff))
    #Variavel de retorno
    result = (q1_diff_format, q2_diff_format, q3_diff_format)
    return result 


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    ecdf = ECDF(dataframe['normal'])
    mean = dataframe['normal'].mean()
    std = dataframe['normal'].std()
    interval = [mean - std, mean + std]
    interval_diff = ecdf(interval)[1] - ecdf(interval)[0]
    #Saida formatada
    interval_format = float('{:.3f}'.format(interval_diff))
    return interval_format



# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    #Media e variancia normal
    m_norm = dataframe['normal'].mean()
    v_norm = dataframe['normal'].var()
    #Media e variancia da binomial
    m_binom = dataframe['binomial'].mean()
    v_binom = dataframe['binomial'].var()
    # Calculo de diferenca
    mean_diff = m_binom - m_norm
    var_diff = v_binom - v_norm
    #Saida formatada
    mean_diff_format = float('{:.3f}'.format(mean_diff))
    var_diff_format = float('{:.3f}'.format(var_diff))
    # Saida
    result = (mean_diff_format, var_diff_format)
    return result


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[ ]:


stars = pd.read_csv("stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# In[ ]:


stars.head()


# ## Inicie sua análise da parte 2 a partir daqui

# In[ ]:


# Sua análise da parte 2 começa aqui.


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


def standardization(data):
  media = data.mean()
  desvio_padrao = data.std()
  z = (data - media) / desvio_padrao
  return z 

def q4():
    # Retorne aqui o resultado da questão 4.
    false_pulsar_mean_profile_filter = stars[stars['target'] == 0]['mean_profile']
    #Padronizacao
    false_pulsar_mean_profile_standardized = standardization(false_pulsar_mean_profile_filter)
    #Achando os quantis
    false_pulsar_standardized_80_quantil = sct.norm.ppf(0.80, loc = false_pulsar_mean_profile_standardized.mean(), scale = false_pulsar_mean_profile_standardized.std())
    false_pulsar_standardized_90_quantil = sct.norm.ppf(0.90, loc = false_pulsar_mean_profile_standardized.mean(), scale = false_pulsar_mean_profile_standardized.std())
    false_pulsar_standardized_95_quantil = sct.norm.ppf(0.95, loc = false_pulsar_mean_profile_standardized.mean(), scale = false_pulsar_mean_profile_standardized.std())
    
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    
    #Utilizando a CDF empírica
    fp_standardized_empirical_80_quantil = ecdf(false_pulsar_standardized_80_quantil)
    fp_standardized_empirical_90_quantil = ecdf(false_pulsar_standardized_90_quantil)
    fp_standardized_empirical_95_quantil = ecdf(false_pulsar_standardized_95_quantil)
    
    #Formatacao de saida
    false_pulsar_80_quantil_format = float("{:.3f}".format( fp_standardized_empirical_80_quantil))
    false_pulsar_90_quantil_format = float("{:.3f}".format( fp_standardized_empirical_90_quantil))
    false_pulsar_95_quantil_format = float("{:.3f}".format( fp_standardized_empirical_95_quantil))
     
    #Saida 
    result = (false_pulsar_80_quantil_format, false_pulsar_90_quantil_format, false_pulsar_95_quantil_format)
    return result

q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    false_pulsar_mean_profile_filter = stars[stars['target'] == 0]['mean_profile']
    false_pulsar_mean_profile_standardized = standardization(false_pulsar_mean_profile_filter)
    
    #Evitar numeros magicos
    mean = 0
    std = 1
    q1 = 0.25
    q2 = 0.50
    q3 = 0.75
    thousand = 100
    
    #quantis para versao padronizada
    fp_standardized_25_quantil = np.percentile(false_pulsar_mean_profile_standardized, q1 * thousand)
    fp_standardized_50_quantil = np.percentile(false_pulsar_mean_profile_standardized, q2 * thousand)
    fp_standardized_75_quantil = np.percentile(false_pulsar_mean_profile_standardized, q3 * thousand)

    #Distribuicao normal
    q1_normal = sct.norm.ppf(q1, loc=mean, scale=std)
    q2_normal = sct.norm.ppf(q2, loc=mean, scale=std)
    q3_normal = sct.norm.ppf(q3, loc=mean, scale=std)
    

    #Diferenca
    q1_diff = fp_standardized_25_quantil - q1_normal
    q2_diff = fp_standardized_50_quantil - q2_normal 
    q3_diff = fp_standardized_75_quantil - q3_normal

    #Formatacao
    q1_diff_format = float("{:.3f}".format(q1_diff))
    q2_diff_format = float("{:.3f}".format(q2_diff))
    q3_diff_format = float("{:.3f}".format(q3_diff))
    return (q1_diff_format, q2_diff_format, q3_diff_format)

q5()


# In[ ]:


0.027, 0.04, -0.004


# 
# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
