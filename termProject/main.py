from tokenizer import selectTokenizer
from ngram import *
from util import *

#read text file from directory
raw = open("F:/Work Folder/KMUTT/NLP/codingAssNLP/sampletext.txt",encoding="UTF-8").read() 

#tokenizer word using the selected tokenizer
#(see tokenizer.py module for all available tokenizers)
tok = selectTokenizer("regxUltra",raw)

#find ngram 
ug = dict(unigram(20,tok))
bg = dict(bigram(20,tok))
tg = dict (trigram(10,tok))

#plot bar chart
plotFreq(bg.values(),bg.keys())
plotFreq(tg.values(),tg.keys())
plotFreq(ug.values(),ug.keys())
