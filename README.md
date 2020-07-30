# Airlines_Tweet_Classification
 This project is an effort towards classifying the user the customer feedback towards positive, negative and neutral sentiment.

The data is present in input/Usecase3_Dataset.csv which has 3 columns namely : 1) airline_sentiment 2) airline 3)text

Objective: The goal here is to build a classification model to decide whether it is a positive or negative or neutral category. Also, build a system to understand themes around the customer's feedback.   


Solution Approach : 

We will create a ML model after doing the following:
1) EDA
2) Text Cleansing
3) Comparing Models

Currently the structure has 3 folders: input, output and src

All the code that we require is in src folder

We have tried to create a framework which can take in various vectorizers & models, perform grid search, find best model and then give back the results

We also create a pickle file for the best classifier. 
This framework currently support traditional ML models like boosting, bagging etc. It does not support Neural Networks wrappers like keras, or other frameworks like
PyTorch.

To run the whole code you can run engine.py present in src folder.

Average accuracy achieved is 76%. 
To Do:
1) Create different models using BERT, Golve etc
2) Use undersampling or oversampling to make dataset balanced
3) Finding customer themes by using libraries like LIME and SHAP 

Thanks!
