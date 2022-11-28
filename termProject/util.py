from tokenizer import selectTokenizer
from ngram import *
import numpy as np
import matplotlib.pyplot as plt
import spacy
#clean token by removing words listed here
#return a list of cleaned tokenized text 
def cleanToken(token):
    stopword = ["Is","Am","Are","Was","Were","I","You","The","A","An","Of","Then","to"
    ,"As","That","It","My","This","There","So","Me","They","Do","Does","Did","Be","These",
    "Not","At","Have","Has","Had","Her","Or",""]
    stopwordLow = [x.lower() for x in stopword]
    
    return [x for x in token if x not in stopword and x not in stopwordLow]

#join tokenized text back to string
#return string 
def cat(token):
    return (" ".join(token))

#convert tokenized text to Spacy Doc object (for further processing ex. NER)
#return Spacy Doc object
def convertToDocObject(input):
    nlp = spacy.load("en_core_web_trf")
    return (nlp(cat(input)))

def plotFreq(word_count,vocab):

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

""" def countFreq(max,input):
    freq = {}
    for word in input:
        lower = word.lower()
        if lower not in freq:
            freq[lower]=0
        freq[lower]+=1
        #print(lower,freq[lower])

    return dict(sorted(freq.items(), key = lambda t: t[1], reverse=True)[:max]) #change[:xx] to get top_K words
 """