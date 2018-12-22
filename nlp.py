import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from empath import Empath #for accuracy
lexicon = Empath()
x=lexicon.analyze("he hit hit , kill  dead   the other person. ",
categories=["violence",
            "aggresion",
            "crime",
            "horror",
            "kill",
            "neglect",
            'dominant_personality',
            'fight',
            'terrorism',
            'negative_emotion'], normalize=True)
print(x)
