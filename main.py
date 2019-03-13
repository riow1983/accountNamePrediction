"""
Join DPC with HOSP2017-2 using Japanese account name
"""
import pandas as pd
import numpy as np
from functools import reduce
import pickle
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


######## preparation -----------------------------------------------------------------

t = Tokenizer(wakati=True)
tfidf_vectorizer = TfidfVectorizer(stop_words=["病院", "クリニック"])


######## read data -----------------------------------------------------------------

ihep = pd.read_excel("./data/HOSP2017-2.xlsx")
dpc = pd.read_excel("./data/DPC_180609.xlsx")

## 病診療区分: {0:診療所 1:病院}
ihep = ihep[ihep["病診区分"] == 1]


######## cosine similarity -----------------------------------------------------------------

## make a list
list_temp = np.append(ihep["医療機関名"].values, dpc["施設名"].values)
    
## tokenize (monophological analysis)
res = []
for n in list_temp:
    tokens = t.tokenize(n)
    string = " ".join(tokens)
    res.append(string)
    
## tf-idf vectorization onto res
tfidf_matrix = tfidf_vectorizer.fit_transform(res)

## calculate cosine similarity
gres = {}
len1 = len(ihep)
len2 = len(dpc)
for i in range(len1):
    res = []
    for k in range(len2):
        # retrieve cosine similarity (0 < item < 1) 
        item = cosine_similarity(tfidf_matrix[i], tfidf_matrix[len1+k]).item()
        print("item", item)
        res.append(item)
    # get argmax of res which is the index for the closest site for i^th site
    nres = np.array(res)
    max_index = np.argmax(nres)
    chosen = dpc.loc[max_index, "施設名"]
    gres[i] = chosen
    
## save the outputs
pickle_out = open("./data/gres.pickle","wb")
pickle.dump(gres, pickle_out)
pickle_out.close()


######## recover gres from pickle file -----------------------------------------------------------------

pickle_in = open("./data/gres.pickle","rb")
gres = pickle.load(pickle_in)

## insert results
ihep.reset_index(inplace=True)
ihep.loc[:, "コサイン類似施設"] = ihep["index"].map(gres)

  
######## ihep + dpc (merge)  -----------------------------------------------------------------

comp = pd.merge(dpc, ihep, right_on="コサイン類似施設", left_on="施設名", how='left')


######## save output file -----------------------------------------------------------------

comp.to_excel("./data/HOSP2017-2_DPC.xlsx", index=False)

