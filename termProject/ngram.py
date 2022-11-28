import nltk
from nltk.collocations import *
from nltk.tokenize import *
from nltk import *
from collections import Counter
#global

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()


def showList(list):
    for i in range(len(list)):
        print(list[i])


def bigram(num, tok):
    finder = BigramCollocationFinder.from_words(tok)
    finder.apply_freq_filter(3)
    bg_rf =  finder.score_ngrams(bigram_measures.raw_freq)
    fdist = nltk.FreqDist(bigrams(tok))
    bg_ct_toList= [(k,v) for k,v in fdist.items()]
    bg_ct = (sorted(bg_ct_toList, key=lambda x:x[1], reverse=True))[:num]
    return (bg_ct)
    

def trigram(num, tok):

    finderT = TrigramCollocationFinder.from_words(tok)
    finderT.apply_freq_filter(3)
    tg_rf =  finderT.score_ngrams(trigram_measures.raw_freq)
    fdist = nltk.FreqDist(trigrams(tok))
    tg_ct_toList= [(k,v) for k,v in fdist.items()]
    tg_ct = (sorted(tg_ct_toList, key=lambda x:x[1], reverse=True))[:num] 
    return(tg_ct)

def unigram(max,tok):
    clean = cleanToken(tok)
    
    count = Counter(clean)

    res = list(sorted(count.items(), key = lambda t: t[1], reverse=True))[:max] #change[:xx] to get top_K words

    return (res)   