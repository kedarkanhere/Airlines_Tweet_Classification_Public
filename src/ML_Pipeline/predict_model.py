from ML_Pipeline import utils
import clfswitcher


def predict_tweet_sentiment(model,tweet,encoder):
    clean_tweet=utils.normalize_text(tweet,lowercase=True, remove_stopwords=True)
    result=model.best_estimator_.predict([clean_tweet])
    
    return encoder.inverse_transform(result)[0]