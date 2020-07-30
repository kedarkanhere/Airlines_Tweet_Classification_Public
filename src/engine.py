from ML_Pipeline import dataset
from ML_Pipeline import utils
from ML_Pipeline import dispatcher
from ML_Pipeline import train_model
from ML_Pipeline import predict_model

print("Loading dataset")
data=dataset.load_data('D:\\technicaltestfromjet2traveltechnologies\\input\\Usecase3_Dataset.csv')

#Cleanse the text(tweet) column
print("Cleansing text, please wait...")
data["Clean_Text"]=data["text"].apply(utils.normalize_text,lowercase=True, remove_stopwords=True)

#Convert the dataset dependent and independent labels to correct form
print("Text cleansing completed. Working towards data conversion")
X,y,encoder=dataset.dataset_transform(data,"Clean_Text","airline_sentiment")

#Import all the parameters in the pipeline
print("Loading pipeline for models to pass GS")
pipeline,params=dispatcher.model()

#Create a model using CV and GS to find best estimator.

print("Parameters passed to model training. PLease wait, this may take some time")
model=train_model.train_model(X,y,pipeline,params)


#Predict sentiment of a tweet
tweet="Thank you for the awesome flight"


sentiment=predict_model.predict_tweet_sentiment(model,tweet,encoder)

print("The sentiment of the tweet is {0}",sentiment)
