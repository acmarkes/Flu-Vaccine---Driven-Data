#%%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats     #ferramentas estatísticas

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, MaxAbsScaler   #scaling
from sklearn.compose import ColumnTransformer   #transformador que aplica steps por colunas
from sklearn.model_selection import train_test_split  #criação de subsets de treino e teste
from sklearn.impute import SimpleImputer,KNNImputer  #preenchimento de dados faltantes de formas diferentes
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline  #criação de pipelines

from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate, GridSearchCV

from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from joblib import Memory #caching
memory = Memory('./cachedir', verbose=0)

np.random.seed(42) 
#%%
path_X_train =  '.\\data\\raw\\training_set_features.csv' 
path_y_train = '.\\data\\raw\\training_set_labels.csv'
X_raw = pd.read_csv(path_X_train)
y_raw = pd.read_csv(path_y_train)

#%%
df_og = pd.concat([X_raw,y_raw.iloc[:,1:]],axis=1)
df = df_og.copy()

#%%
relevant_industries = ['arjwrbjb', 'fcxhlnwr', 'haxffmxo', 'qnlwzans', 'wxleyezf']
df.loc[(~df['employment_industry'].isin(relevant_industries)), 'employment_industry'] = 'other'
df.drop('hhs_geo_region', axis=1)
df.drop('employment_occupation', axis=1)

#%%
df_cat = df.astype('category')
df_cat = df_cat.astype({'respondent_id':'int64','household_adults':'float64', 'household_children':'float64',
                        'h1n1_vaccine':'int64', 'seasonal_vaccine':'int64'})
df_cat.set_index('respondent_id', inplace=True)


#%%
X = df_cat.iloc[:,:-2]
y = df_cat.iloc[:,-2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, stratify=y)

#%%
#Cleaning nulls

nulls_per_col = X.isnull().sum()
null_cols = list(nulls_per_col.sort_values(ascending=False).index)

ordinal_cols = ['h1n1_concern', 'h1n1_knowledge',
'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group']

ohe_cols = ['education', 'race', 'sex', 'income_poverty', 
'marital_status', 'rent_or_own', 'employment_status', 'employment_industry', 'census_msa']


#%%
cv = KFold(shuffle=True)

iter = IterativeImputer(estimator=DecisionTreeClassifier(), 
                            initial_strategy='most_frequent',
                            max_iter=10, random_state=0)
simple = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

#Definindo os passos do ColumnTransformer
imputer = ColumnTransformer(
        transformers=[
            ('kimp', KNNImputer(missing_values=np.nan, n_neighbors=3), null_cols[:4]),
            ('simp', SimpleImputer(missing_values=np.nan, strategy='most_frequent') , null_cols[4:]),              
        ], remainder='passthrough')



encoder = ColumnTransformer(
                transformers=[
                    ('ordinal', OrdinalEncoder(), ordinal_cols),
                    ('ohe', OneHotEncoder(drop='first'), ohe_cols)
                ],remainder='passthrough')

scaler = MinMaxScaler()

kbest = SelectKBest(f_classif, k=10)
recursive_selection = RFECV(estimator=LogisticRegression(),cv=cv, scoring='accuracy')

#%%
#pipeline = make_pipeline(simple, encoder, scaler, kbest, OneVsRestClassifier(SVC()))

scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                        scoring=('accuracy', 'roc_auc'),
                        return_train_score=True)

#%%
pipeline1 = make_pipeline(simple, OrdinalEncoder(), OneHotEncoder(drop='first'), MaxAbsScaler(), kbest, MultiOutputClassifier(DecisionTreeClassifier()))
pipeline2 = make_pipeline(simple, OrdinalEncoder(), OneHotEncoder(drop='first'), MaxAbsScaler(), kbest, ClassifierChain(DecisionTreeClassifier()))

clf = pipeline.fit(X_train, y_train)

#%%
labels = ['h1n1_vaccine', 'seasonal_vaccine']
threshold = 2
selected_features = [] 
for label in labels:
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X_train, y_train[label])
    selected_features.append(list(selector.scores_))

# MeanC^2 
selected_features = np.mean(selected_features, axis=0) > threshold
# MaxC^2
#selected_features = np.max(selected_features, axis=0) > threshold
# %%
