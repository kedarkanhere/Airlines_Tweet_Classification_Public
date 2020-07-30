import pandas as pd
from sklearn.preprocessing import LabelEncoder



def load_data(file_name):
    dataset=pd.read_csv(file_name)

    return dataset


def dataset_transform(df,independent,target):
    Y = pd.get_dummies(df[target]).values #for multilabelled computation
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target])
    #print(y)
    X=df[independent].values

    return X,y,label_encoder
