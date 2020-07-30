import pandas as pd
import numpy as np
import os 
import pandas as pd 
from sklearn import metrics 
from sklearn.model_selection import StratifiedKFold
import joblib 
import warnings 
import random
warnings.filterwarnings ( "ignore" ) 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report 
from sklearn.metrics import make_scorer, matthews_corrcoef as mcc_scorer

def train_model(X,y,pipeline,parameters):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=[random.randint(0,200) for i in range(1)][0] ,stratify=y)
    print("Model Training Started")

    gscv= GridSearchCV(pipeline, parameters, cv=StratifiedKFold(n_splits=5,
     random_state =[random.randint(0,200) for i in range(1)][0] 
    , shuffle = True))

    gscv.fit(X_train,y_train)

    print("Completed Fit")

    preds=gscv.best_estimator_.predict(X_test)

    print(classification_report(y_test,preds))
    #print(gscv.best_estimator_)
    

    joblib.dump(gscv.best_estimator_,"Bestmodel.pkl",compress=1)
    print("Pickle file generated for the best model")
    return gscv



