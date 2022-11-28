from tokenizer import selectTokenizer
from ngram import *
from util import *
raw = open("F:/Work Folder/KMUTT/NLP/codingAssNLP/sampletext.txt",encoding="UTF-8").read() 
tok = selectTokenizer("regxUltra",raw)


ug = dict(unigram(20,tok))
bg = dict(bigram(20,tok))
tg = dict (trigram(10,tok))

plotFreq(bg.values(),bg.keys())
plotFreq(tg.values(),tg.keys())
plotFreq(ug.values(),ug.keys())
