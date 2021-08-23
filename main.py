#%%
import warnings

from sklearn import preprocessing
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats     #ferramentas estatísticas

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, MaxAbsScaler, FunctionTransformer   #scaling
from sklearn.compose import ColumnTransformer   #transformador que aplica steps por colunas
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV  #criação de subsets de treino e teste
from sklearn.impute import SimpleImputer,KNNImputer  #preenchimento de dados faltantes de formas diferentes
from sklearn.pipeline import Pipeline, make_pipeline  #criação de pipelines

from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFECV, SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV 

from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from joblib import Memory #caching
memory = Memory('./cachedir', verbose=0)

np.random.seed(42) 
#%%
path_X_train =  '.\\data\\raw\\training_set_features.csv' 
path_y_train = '.\\data\\raw\\training_set_labels.csv'
X_raw = pd.read_csv(path_X_train)
y_raw = pd.read_csv(path_y_train)

test_set = pd.read_csv('.\\data\\raw\\test_set_features.csv')

#%%
df_og = pd.concat([X_raw,y_raw.iloc[:,1:]],axis=1)



#%%
def feature_treatment(X):
  '''Função que 
    1. deleta as colunas 'hhs_geo_region', 'census_msa' e 'employment_occupation'
    2. diminui a cardinalidade de 'employment_industry' mantendo apenas as os 4 setores com mais empregados vacinados e agregando os outros em 'other'
    3. altera o tipo de dados internamente no dataframe e seu índice '''

  X_ = X.copy()

  #Removendo colunas com pouca informação e muitos nulos
  X_.drop('hhs_geo_region', axis=1, inplace=True, errors='ignore')
  X_.drop('census_msa', axis=1, inplace=True, errors='ignore')
  X_.drop('employment_occupation', axis=1, inplace=True, errors='ignore')
  X_.drop('employment_status', axis=1, inplace=True, errors='ignore')


  #diminuindo a cardinalidade de 'employment_industry', pegando as cinco industrias que mais se vacinaram e substituindo as outras por 'other'
  relevant_industries = ['haxffmxo', 'fcxhlnwr', 'wxleyezf', 'arjwrbjb', 'qnlwzans']
  X_.loc[(~X_['employment_industry'].isin(relevant_industries)) & (~X_['employment_industry'].isnull()), 'employment_industry'] = 'other'

  #alterando o dtype das colunas
  for col in X_.select_dtypes(include='number').columns :
    X_[col] = X_[col].astype('int64')
  X_ = X_.astype('category')
  X_ = X_.astype({'respondent_id':'int64'}, errors='ignore')
  X_.set_index('respondent_id', inplace=True)

  return X_


def proportion_null_treatment(X):
  ''' Função que substitui os nulos de cada coluna de um dataframe de forma aleatória mas mantendo a proporção de cada categoria
  '''
  X_ = X.copy()

  for col in X_.columns:
    if X_[col].dtype == 'numeric':
      pass
    else:
      proportions = X_[col].value_counts(normalize=True) 
      X_[col] = X_[col].fillna(pd.Series(np.random.choice(proportions.index, 
                                                          p=proportions.values, size=len(X_))))
      
      '''EXEMPLO: df['race'].value_counts(normalize=True) == White                0.794623
                                                             Black                0.079305
                                                             Hispanic             0.065713
                                                             Other or Multiple    0.060359
         onde .index são as categorias e .values são suas proporções. As categorias serão selecionadas aleatoriamente de acordo com as 
         probabilidades definidas o que mantém as proporções das categorias no resultado final'''
         
  return X_

FeatTreatment = FunctionTransformer(feature_treatment)
NullTreatment = FunctionTransformer(proportion_null_treatment)  


#%%

df = NullTreatment.fit_transform(df_og)
X = df.iloc[:,:-2]
y = df.iloc[:,-2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, stratify=y)


#%%


ordinal_cols = ['h1n1_concern', 'h1n1_knowledge',
'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group']

ohe_cols = ['education', 'race', 'sex', 'income_poverty', 
'marital_status', 'rent_or_own', 'employment_industry']

encoder = ColumnTransformer(transformers=[
                                  ('ordinal', OrdinalEncoder(), ordinal_cols),
                                  ('ohe', OneHotEncoder(drop='first'), ohe_cols)
                                  ],remainder='passthrough')

scaler = MinMaxScaler()


#%%
preprocessor = Pipeline([('feat_t',FeatTreatment),
                        ('encoder',encoder),
                        ('scaler',scaler)])


X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

#%%
model = ClassifierChain(XGBClassifier(), order=[1,0])


parameters = [
    {
        'base_estimator': [XGBClassifier()],
        'base_estimator__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'base_estimator__max_depth': [3, 5, 7, 10],
        'base_estimator__min_child_weight': [0.1, 0.3, 0.5, 0.8, 1],
        'base_estimator__gamma': [i/10.0 for i in range(0,5)],
        'base_estimator__reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }
]


#%%
grid_search = RandomizedSearchCV(model, parameters, scoring='roc_auc', n_jobs=-1)
grid_result = grid_search.fit(X_train_proc, y_train)

final_model = grid_result.best_estimator_


#%%

XGB_model = ClassifierChain(XGBClassifier(gamma= 0.0,
                                            learning_rate= 0.2,
                                            scale_pos_weight=3.7,
                                            max_depth= 3,
                                            min_child_weight= 1),
                                            order=[1,0])


#final_model = XGB_model

#%%

final_model.fit(X_train_proc, y_train)

#%%
predictions = final_model.predict_proba(X_test_proc)

predictions_se = predictions[:,1].reshape(-1,1)
predictions_h1 = predictions[:,0].reshape(-1,1)

from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(
        f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
    )

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 8))
plot_roc(
    y_test['h1n1_vaccine'], 
    predictions_h1, 
    'h1n1_vaccine',
    ax=ax1
)

plot_roc(
    y_test['seasonal_vaccine'], 
    predictions_se, 
    'seasonal_vaccine',
    ax=ax2
)
#%%
roc_auc_score(y_test, np.hstack((predictions_h1, predictions_se)))


# %%
X_proc = preprocessor.fit_transform(X)
model = final_model.fit(X_proc, y)

#%%
test_proc = NullTreatment.fit_transform(test_set)
test_proc = preprocessor.named_steps['feat_t'].transform(test_proc)
test_proc = preprocessor.named_steps['encoder'].transform(test_proc)
test_proc = preprocessor.named_steps['scaler'].transform(test_proc)
#%%
final_predictions = model.predict_proba(test_proc)


final_predictions_se = final_predictions[:,1].reshape(-1,1)
final_predictions_h1 = final_predictions[:,0].reshape(-1,1)
# %%
submission_df = pd.read_csv("./submission_format.csv", 
                            index_col="respondent_id")

# Make sure we have the rows in the same order
#np.testing.assert_array_equal(test_set.index.values, 
                              #submission_df.index.values)

# Save predictions to submission data frame
submission_df["h1n1_vaccine"] = final_predictions_h1
submission_df["seasonal_vaccine"] = final_predictions_se

submission_df.head()

#%%
submission_df.to_csv('my_submission1.csv', index=True)
# %%
