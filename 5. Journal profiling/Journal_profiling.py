# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Mon Aug 16 11:05:32 2021
@author: Francis Lareau
This is Project Astrobiology.
Journal profiling
"""
#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import pandas as pd
import numpy as np
import pickle
import bz2

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("your_main_path")
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_lda_models_alpha.pbz2"), "r") as f:
    lda_models = pickle.load(f)
    
ldamodel = lda_models[0.2]

with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_vocab_sklearn.pbz2"), "r") as f:
    vocab = pickle.load(f)
    
with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_dtm_sklearn.pbz2"), "r") as f:
    dtm = pickle.load(f)

DF = pd.read_pickle(os.path.join(main_path,
                                 "0. Data",
                                 "BIG3_dataframe_metadata.pkl"))

DF['Period'] = DF.Year.apply(lambda x: #2 years period
    str(int(x)-(int(x)-1967)%3)+'-'+str((int(x)-(int(x)-1967)%3)+2))
    
#==============================================================================
# ##################### Data statistic, lda model score and lda hyperparameters
#==============================================================================

df_param=pd.DataFrame(index=['Value'])
df_param['Sparsity']=((dtm.todense() > 0).sum() / 
        dtm.todense().size*100) #sparsicity (% nonzero)
df_param['Log Likelyhood']=ldamodel.loglikelihood() #Log Likelyhood (higher better)
df_param['Perplexity']='' #Perplexity (lower better, exp(-1. * log-likelihood per word)
df_param['alpha']=ldamodel.alpha
df_param['eta']=ldamodel.eta
df_param['n_iter']=ldamodel.n_iter
df_param['n_components']=ldamodel.n_topics
df_param['random_state']=ldamodel.random_state
df_param['refresh']=ldamodel.refresh

#==============================================================================
# ########################################################### Topic by document
#==============================================================================

#Topic for each document
lda_output=ldamodel.doc_topic_
topicnames = ["Topic_" + str(i) for i in range(len(ldamodel.components_))]
docnames = [i for i in range(dtm.shape[0])]
df_document_topic = pd.DataFrame(lda_output, 
                                 columns=topicnames,
                                 index=docnames)
df_document_topic_sorted = df_document_topic.apply(lambda x: x.sort_values(ascending=False).values)
dominant_topic = np.argmax(df_document_topic.values, axis=1)
#add results to statistic general
DF['Dom_topic'] = dominant_topic
DF_topic=pd.concat([DF,df_document_topic],
                   axis=1,
                   join='inner')
    
#==============================================================================
# ############################################ Topic by period for each journal
#==============================================================================

Journals_TP = {}
for journal in set(DF.Journal_ID):
    membership = DF.Journal_ID==journal
    DF_PT1=pd.DataFrame(lda_output[membership],
                        columns=topicnames,
                        index=[i for i, x in enumerate(membership) if x])
    DF_PT1['Period']=DF_topic.Period[membership]
    DF_PT1 = DF_PT1.groupby(['Period']).sum()
    DF_TP1 = DF_PT1.transpose()
    DF_TP1 = DF_TP1/DF_TP1.sum()
    DF_TP_Overall1 = DF_PT1.transpose()
    DF_TP_Overall1['Raw'] = DF_PT1.sum()
    DF_TP_Overall1['Overall'] = DF_PT1.sum() / sum(DF_PT1.sum())
    Journals_TP[journal]=[DF_TP1,DF_TP_Overall1]
    
#==============================================================================
# ################################################################ Save results
#==============================================================================
        
# Save lda results to excel
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "0. Data",
                                     "BIG3_LDA_Results_T"+str(ldamodel.n_topics)+"_A"+str(ldamodel.alpha)+"_Journal.xlsx"))
df_param.T.to_excel(writer,'Para Score',encoding='utf8')        
Journals_TP['ASTRO'][0].to_excel(writer,'ASTRO Topics vs Periods',encoding='utf8')
Journals_TP['ASTRO'][1].to_excel(writer,'ASTRO Overall Topics vs Periods',encoding='utf8')
Journals_TP['IJA'][0].to_excel(writer,'IJA Topics vs Periods',encoding='utf8')
Journals_TP['IJA'][1].to_excel(writer,'IJA Overall Topics vs Periods',encoding='utf8')
Journals_TP['OLEB'][0].to_excel(writer,'OLEB Topics vs Periods',encoding='utf8')
Journals_TP['OLEB'][1].to_excel(writer,'OLEB Overall Topics vs Periods',encoding='utf8')
writer.save()
writer.close()
