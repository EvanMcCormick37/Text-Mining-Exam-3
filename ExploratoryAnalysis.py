#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df_100 = pd.read_csv('data/wdms/WDM100.csv',index_col=0)
df_all = pd.read_csv('data/wdms/WDMAll.csv',index_col=0)

wdm_100 = df_100.iloc[:,1:]
wdm_all = df_all.iloc[:,1:]

fig, ax = plt.subplots(2,1,constrained_layout=True)

for i, wdm in enumerate([wdm_100,wdm_all]):
    sums = wdm.sum()
    wordcloud = WordCloud().generate_from_frequencies(sums)
    ax[i].imshow(wordcloud)