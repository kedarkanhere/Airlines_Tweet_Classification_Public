from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from clfswitcher import ClfSwitcher
from sklearn.ensemble import GradientBoostingClassifier


def model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', ClfSwitcher()),
    ])

    parameters = [
        {
            'clf__estimator': [XGBClassifier()], # SVM if hinge loss / logreg if log loss
            'tfidf__max_df': [0.75, 1.0],
            'tfidf__ngram_range':[(4,4)],
            'tfidf__stop_words': ['english'],
            'clf__estimator__n_estimators': [250],
            'clf__estimator__learning_rate': [0.5,0.1],
            'clf__estimator__objective':['multi:softmax'],
            'clf__estimator__num_class':[3],
        },
        {
            'clf__estimator': [GradientBoostingClassifier()],
            'clf__estimator__n_estimators': [250],
            'clf__estimator__max_depth': [5,7],
        },
    ]

    return pipeline,parameters