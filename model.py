import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

DataFinal = pd.read_csv("/Users/belmontclaire/Documents/Projet7/DataFinal.csv")

dataTrain = DataFinal[DataFinal['TARGET'].notnull()]

X = dataTrain.drop(["TARGET","index","SK_ID_CURR"],axis=1)
y = dataTrain["TARGET"]
# fit predictor and target variable
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
model_rfc = RandomForestClassifier(max_depth = 6,
                                   min_samples_split = 2,
                                   n_estimators =52
                                  ).fit(X_smote.values, y_smote.values)

filename = '/Users/belmontclaire/Documents/Projet7/finalized_model.sav'
pickle.dump(model_rfc, open(filename, 'wb'))
