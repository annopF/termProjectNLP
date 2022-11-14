import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
import re
import numpy as np
import spacy
from spacy import displacy
import time

stopWord = list(stopwords.words("english"))
text0 = open("F:/Work Folder/KMUTT/NLP/codingAssNLP/sampletext.txt",encoding="UTF-8").read()

pattern = "(?:\w+\.){2,}|\w+['’]\w+|\w+" 
start = time.time()
trimText = re.findall(pattern, text0)
end = time.time()
print("Time taken: ", end-start)
nlp = spacy.load("en_core_web_trf")
doc = nlp(text0)


freq = {}
for word in trimText:
    lower = word.lower()
    if lower not in freq:
        freq[lower]=0
    freq[lower]+=1
    #print(lower,freq[lower])

res = dict(sorted(freq.items(), key = lambda t: t[1], reverse=True)[:100]) #change[:xx] to get top_K words

for w in list(res):
    if w in stopWord:
        del res[w]

def printFreq ():
    for key in res:
        print(key, ' : ', res[key]) 

def printToken ():
    for i in range(len(trimText)):
        print(trimText[i])
        
def spaPrintEntity():
    
    for entity in doc.ents:
        print("Entity: ",entity, entity.label_)

def spaPrintToken ():
    for i in doc:
        print(i)

spaPrintToken()

vocab = res.keys()
print(vocab)
word_count = res.values()
y_pos = np.arange(len(vocab))

fig, plot = plt.subplots()


plot.barh(y_pos,word_count)
plot.set_yticks(y_pos)
plot.set_yticklabels(vocab)
plot.invert_yaxis()  # labels read top-to-bottom
plot.set_xlabel('occurrence')
plot.set_ylabel('vocab')

plot.set_title("Most repeated words in the document")


plt.show()