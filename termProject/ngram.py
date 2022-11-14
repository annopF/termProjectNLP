import nltk
from nltk.collocations import *
from nltk.tokenize import *
from nltk import *
import spacy
from util import cleanToken
from collections import Counter
#global

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()


def showList(list):
    for i in range(len(list)):
        print(list[i])


def bigram(mode, num, tok):
    
    finder = BigramCollocationFinder.from_words(tok)

    finder.apply_freq_filter(2)
    bigram = bigrams(tok)
    score =  finder.score_ngrams(bigram_measures.raw_freq)
    nbest =  finder.nbest(bigram_measures.raw_freq,None)
    fdist = nltk.FreqDist(bigram)
    #bg = bigrams(tok)
    #count = [(item, list(bg).count(item)) for item in sorted(set(bg))]
    count = [(k,v) for k,v in fdist.items() if v > 5]

    
    if mode == "sc":
        showList(score)
    elif mode == "nb":
        showList(sorted(count))
        return(nbest)
    

def trigram(mode, num, tok):

    finder = TrigramCollocationFinder.from_words(tok)

    finder.apply_freq_filter(2)
    score =  finder.score_ngrams(trigram_measures.likelihood_ratio)
    nbest =  finder.nbest(trigram_measures.likelihood_ratio,num)
    count = [(item, score.count(item)) for item in sorted(set(score))]

    if mode == "sc":
        showList(score)
        return(score)
    elif mode == "nb": 
        showList(count)
        return(nbest)
        

def unigram(max,tok):
    clean = cleanToken(tok)
    
    count = Counter(clean)

    res = list(sorted(count.items(), key = lambda t: t[1], reverse=True)) #change[:xx] to get top_K words

    showList(res[0:max])
    