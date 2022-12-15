# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Jun 29 15:24:11 2021
@author: Francis Lareau
This is Project Astrobiology.
Preprocessing
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import sys
import pandas as pd
import pickle
import datetime
import re
import bz2
from sklearn.feature_extraction.text import CountVectorizer
import treetaggerwrapper #  TreeTagger must be install and path specified

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("D:\projetjstor\Astrobiology\Final")
os.chdir(main_path)

treetagger_path = "C:\TreeTagger"

#==============================================================================
# ################################################################# Import data
#==============================================================================

DF = pd.read_pickle(os.path.join(main_path,
                                 "0. Data",
                                 "Private",
                                 "DF_Consolidation_Astrobiology.pkl"))

#==============================================================================
# ########################### Word tokenization, POS tagging, and lemmatization
#==============================================================================

tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR=treetagger_path)
DF['Lemma'] = ''

time_start = datetime.datetime.now()
for i in range(len(DF)):
    tokens = treetaggerwrapper.make_tags(tagger.tag_text(DF.Article[i]))
    list_of_lemma=[]
    for token in tokens:
        if (isinstance(token,treetaggerwrapper.Tag)
        and bool(re.findall('FW|MD|VVPRHASAL|VV.?|JJ.?|NN.?|NP.?|RB.?',token[1]))        
        and len(token[2])>=3):            
            token = token[2].lower()
            token = re.sub('^[\.\-\s\']+|[\.\-\s\']+$|^[a-z][\-\'][a-z]$','',token)
            list_of_lemma.append(token)
    DF.Lemma[i]=list_of_lemma
    sys.stdout.write("\rLemmatizing till article #"+str(i)+" took %s"%(str(datetime.datetime.now() - time_start)))
    sys.stdout.flush() # 3h

#==============================================================================
# ############################################################### Vectorization
#==============================================================================

def identity_tokenizer(text):
    ''' Method to use with Countvectorizer '''
    return text

#building stopwords set
stopwords = {line.strip() for line in open(
        os.path.join(main_path,
                     "0. Data",
                     "stopwords_en.txt"),
                     encoding='utf-8')}

stopwords = stopwords.union({'','etc.'})

#setup
vectorizer = CountVectorizer(lowercase = False,
                             min_df = 20,
                             analyzer = 'word', 
                             tokenizer = identity_tokenizer,
                             preprocessor = identity_tokenizer,
                             stop_words = stopwords, 
                             ngram_range = (1, 1))
#Create matrix and vocab
freq_term_matrix = vectorizer.fit_transform(DF.Lemma)
vocab = vectorizer.vocabulary_

#==============================================================================
# ################################################################### Save Data
#==============================================================================

pd.to_pickle(DF,os.path.join(main_path,
                             "0. Data",
                             "Private",
                             "BIG3_dataframe.pkl"))

writer = pd.ExcelWriter(os.path.join(main_path,
                                     "0. Data",
                                     "Private",
                                     "BIG3_dataframe.xlsx"),
                        options={'strings_to_urls': False})
DF.to_excel(writer,'General')
writer.save()
writer.close()

pd.to_pickle(
        DF[['HTML_url', 'PDF_url', 'DOI', 'Journal_ID', 'Journal', 'Volume', 
            'Issue', 'Date', 'Page_first', 'Page_last', 'Page_range', 'Section', 
            'Type', 'Language', 'Abstract', 'Keywords', 'Bibliography', 
            'Authors', 'Year', 'Title', 'Citation']],
            os.path.join(main_path,
                         "0. Data",
                         "BIG3_dataframe_metadata.pkl"))

with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_vocab_sklearn.pbz2"), "w") as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    
with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_dtm_sklearn.pbz2"), "w") as f:
    pickle.dump(freq_term_matrix, f, pickle.HIGHEST_PROTOCOL)

