#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[52]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# In[53]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[54]:


countries = pd.read_csv("countries.csv",decimal=',')


# In[55]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[38]:


# Sua análise começa aqui.
countries.Country = countries.Country.str.strip()
countries.Region = countries.Region.str.strip()


# In[39]:


countries.describe()


# In[7]:


countries.isna().sum()


# In[8]:


countries.shape


# In[9]:


countries.dtypes


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[10]:


def q1():
    # Retorne aqui o resultado da questão 1.
    onehot = OneHotEncoder(sparse=False, dtype=np.int)
    region_encoded = onehot.fit_transform(countries[['Region']])
    return sorted(onehot.categories_[0])


# In[11]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    density_bins = discretizer.fit_transform(countries[['Pop_density']])
    return sum(countries['Pop_density'] > discretizer.bin_edges_[0][9])
    
    


# In[13]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[14]:


def q3():
    # Retorne aqui o resultado da questão 3.
    onehot = OneHotEncoder(sparse=False, dtype=np.int)
    countries['Climate'].fillna(0,inplace=True)
    region_encoded = onehot.fit_transform(countries[['Region','Climate']])
    return region_encoded.shape[1]


# In[15]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[138]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[139]:


def q4():
    # Retorne aqui o resultado da questão 4.
    df_test = pd.DataFrame([test_country],columns=countries.columns)
    pipeline = Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),('Scaler',StandardScaler())])
    pipeline.fit(countries._get_numeric_data())
    pipeline.transform(df_test._get_numeric_data())
    df_test.iloc[:,2:] = pipeline.transform(df_test._get_numeric_data())
    return float(round(df_test['Arable'],3))


# In[140]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[20]:


def q5():
    # Retorne aqui o resultado da questão 4.
    qt1,qt3 =  countries['Net_migration'].quantile([0.25,0.75])
    iqr = qt3-qt1
    outliers_abaixo = sum(countries['Net_migration'] < qt1-1.5*iqr)
    outliers_acima = sum(countries['Net_migration'] > qt3+1.5*iqr)
    pct = (outliers_abaixo + outliers_acima)/countries['Net_migration'].count()
    # Considerando um percentual de outliers acima de 50% para remover ou não
    return (outliers_abaixo,outliers_acima,bool(pct > 0.5))
    


# In[21]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[22]:


from sklearn.datasets import fetch_20newsgroups


# In[23]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[24]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)


# In[25]:


def q6():
    # Retorne aqui o resultado da questão 4.

    idx= count_vectorizer.vocabulary_.get('phone')
    return int(newsgroups_counts[:,idx].toarray().sum())


# In[26]:


q6()


# In[27]:


tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(newsgroups_counts)

newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[29]:


def q7():
    # Retorne aqui o resultado da questão 4.
    idx= count_vectorizer.vocabulary_.get('phone')
    return float(round(newsgroups_tfidf[:,idx].sum(),3))


# In[35]:





# In[ ]:




