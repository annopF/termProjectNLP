#test which assocMeasure to use 
from tokenizer import *
from ngram import *
from util import *
from nltk.collocations import *
from nltk.lm import NgramCounter
from nltk.tokenize import *
from nltk import *
import spacy
import pandas as pd

apply_frequency = 5 #set number of finder.apply_freq_filter
apply_frequency_T = 3

maxU = 50 #set maximum number of unigram to print
maxB = 20 #set maximum number of bigrams to print
maxT = 10 #set maximum number of trigrams to print

#change path to input text file here
raw = open("F:/Work Folder/KMUTT/NLP/codingAssNLP/pixeltext.txt",encoding="UTF-8").read() 
#raw = open("F:/Work Folder/KMUTT/SeniorProject/nlpSPX/dataset/ieltsText/ie4.txt",encoding="UTF-8").read() 

#tokenize text using the specified tokenizer (see tokenizer.py for all available options)
tok = selectTokenizer("regxUltra",raw)
print(tok)
#create bigram lists with differnt assocMeasure score for comparison purposes
finder = BigramCollocationFinder.from_words(tok)
finder.apply_freq_filter(apply_frequency)
bg_ll =  finder.score_ngrams(bigram_measures.likelihood_ratio)
bg_rf =  finder.score_ngrams(bigram_measures.raw_freq)
bg_pmi = finder.score_ngrams(bigram_measures.pmi)
bg_chi = finder.score_ngrams(bigram_measures.chi_sq) 
fdist = nltk.FreqDist(bigrams(tok))
bg_ct_toList= [(k,v) for k,v in fdist.items()]
bg_ct = (sorted(bg_ct_toList, key=lambda x:x[1], reverse=True))

#create trigram lists with differnt assocMeasure score for comparison purposes
finderT = TrigramCollocationFinder.from_words(tok)
finderT.apply_freq_filter(apply_frequency_T)
tg_ll =  finderT.score_ngrams(trigram_measures.likelihood_ratio)
tg_rf =  finderT.score_ngrams(trigram_measures.raw_freq)
tg_pmi = finderT.score_ngrams(trigram_measures.pmi)
tg_chi = finderT.score_ngrams(trigram_measures.chi_sq) 
fdistT = nltk.FreqDist(trigrams(tok))
tg_ct_toList= [(k,v) for k,v in fdistT.items()]
tg_ct = (sorted(tg_ct_toList, key=lambda x:x[1], reverse=True))

#count the occurrences of each bigram in the input text
res_bg_ll = [[item,fdist[item[0]]] for item in bg_ll][:maxB]              
res_bg_rf = [[item,fdist[item[0]]] for item in bg_rf][:maxB] 
res_bg_pmi = [[item,fdist[item[0]]] for item in bg_pmi][:maxB] 
res_bg_chi = [[item,fdist[item[0]]] for item in bg_chi][:maxB] 

#count the occurrences of each trigram in the input text
res_tg_ll = [[item,fdistT[item[0]]] for item in tg_ll][:maxT]           
res_tg_rf = [[item,fdistT[item[0]]] for item in tg_rf][:maxT] 
res_tg_pmi = [[item,fdistT[item[0]]] for item in tg_pmi][:maxT] 
res_tg_chi = [[item,fdistT[item[0]]] for item in tg_chi][:maxT] 


""" for item in bg_ll:
    res_bg_ll.append([item,fdist[item[0]]]) 
 """

#create dataframe of the following structure: [(("ngram1","ngram2"),score),occurrence]
data = {"Likelihood":res_bg_ll[:maxB],"Raw Freq":res_bg_rf[:maxB],"PMI":res_bg_pmi[:maxB],"Chi Square":res_bg_chi[:maxB], "Count":bg_ct[:maxB]}
data2 = {"Likelihood":res_tg_ll[:maxT],"Raw Freq":res_tg_rf[:maxT],"PMI":res_tg_pmi[:maxT],"Chi Square":res_tg_chi[:maxT], "Count":tg_ct[:maxT]}

df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)

print("UNIGRAM ------------------------------------------------------------------------------------------------------------------------------------------")
print (unigram(maxU,tok))
print("BIGRAM ------------------------------------------------------------------------------------------------------------------------------------------")
print (df) 
print("TRIGRAM ------------------------------------------------------------------------------------------------------------------------------------------")
print(df2)

#prepare data for chart ploting by
#creating a dict of ngram vs freq -> {(ngram1,ngram2):count} 
out = {}
for item in res_bg_ll:
    out[item[0][0]] = item[1]

out = dict(sorted(out.items(),key = lambda x: x[1], reverse = True))

#plot chart
plotFreq(out.values(),out.keys())