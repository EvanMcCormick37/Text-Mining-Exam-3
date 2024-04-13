import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame(columns=['LABEL','REVIEW'])
for i in range(1,16):
    for sentiment in ['Positive','Negative']:
            f = open(f'data/{sentiment}/{i}.txt','r')
            content = f.read()
            df.loc[len(df.index)]=[sentiment,content]


wnl = WordNetLemmatizer()

def cleanLemmatize(t):
    text = re.sub("[^a-zA-Z'\\s+]","", t)
    text = re.split("\\s+",text)
    return ' '.join([wnl.lemmatize(word.lower()) for word in text])

v_CL = np.vectorize(cleanLemmatize)
df['REVIEW'] = v_CL(df['REVIEW'])

CV1 = CountVectorizer(stop_words='english',min_df=2)
CV2 = CountVectorizer(stop_words='english',min_df=2,max_features=100)

wdm1 = pd.concat([ 
    df['LABEL'],
    pd.DataFrame(CV1.fit_transform(df['REVIEW']).toarray(),columns=CV1.get_feature_names_out())
    ], axis=1)
wdm2 = pd.concat([
    df['LABEL'],
    pd.DataFrame(CV2.fit_transform(df['REVIEW']).toarray(),columns=CV2.get_feature_names_out())
    ], axis=1)

wdm1.to_csv('data/wdms/WDMAll.csv')
wdm2.to_csv('data/wdms/WDM100.csv')