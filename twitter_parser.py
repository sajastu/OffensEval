import pandas as pd
import spacy
from tqdm import tqdm

def custom_pipeline(nlp):
    return (nlp.parser)


# 1.6M tweets

nlp = spacy.load("en_core_web_sm", create_pipeline=custom_pipeline)
outf = open('twitter-sent/4.txt', mode='a')

tweets = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header=0).iloc[:,5]

for doc in tqdm(nlp.pipe(tweets[320000*4:], batch_size=500, n_threads=32), total=320000):
    sents =[s.text for s in list(doc.sents)]
    for sent in sents:
        outf.write(sent.replace('\n', ''))
        outf.write('\n')
    outf.write('\n')
