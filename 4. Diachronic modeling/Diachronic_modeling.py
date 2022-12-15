# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Mon Aug 16 11:05:32 2021
@author: Francis Lareau
This is Project Astrobiology.
Diachronic modeling
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
    
#count document by topic
df_topic_distribution = DF['Dom_topic'].value_counts(
        ).reset_index(name="Num_Documents")
df_topic_distribution.columns = ['Topic_Num', 'Num_Doc']
# Topic - keyword Matrix
df_topic_keywords = pd.DataFrame(ldamodel.components_)#every row =1
df_topic_keywords.index = topicnames
#Transpose to topic - keyword matrix
df_keywords_topic = df_topic_keywords.transpose()
df_keywords_topic.index = vocab
# Topic - Top Keywords Dataframe
n_top_words = 50+1
DF_Topic_TKW = pd.DataFrame(columns=range(n_top_words-1),index=range(len(ldamodel.components_)))
topic_word = ldamodel.components_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    DF_Topic_TKW.loc[i]=topic_words

DF_Topic_TKW.columns = ['Word_'+str(i) for i in range(DF_Topic_TKW.shape[1])]
DF_Topic_TKW.index = ['Topic_'+str(i) for i in range(DF_Topic_TKW.shape[0])]
DF_Topic_TKW['Sum_Doc'] = np.array(DF['Dom_topic'].value_counts(
        ).sort_index())
DF_Topic_TKW['Top-10_Words'] = ''
for idx,row in DF_Topic_TKW.iterrows():
    DF_Topic_TKW['Top-10_Words'][idx]=(row['Word_0']+'; '+row['Word_1']+'; '+
                row['Word_2']+'; '+row['Word_3']+'; '+row['Word_4']+'; '+
                row['Word_5']+'; '+row['Word_6']+'; '+row['Word_7']+'; '+
                row['Word_8']+'; '+row['Word_9'])

#==============================================================================
# ############################################################# Topic by period
#==============================================================================

# Topic - Period Matrix
DF_PT=pd.DataFrame(lda_output,
                   columns=topicnames,
                   index=docnames)

DF_PT['Period']=DF_topic.Period
DF_PT = DF_PT.groupby(['Period']).sum()
DF_TP = DF_PT.transpose()
DF_TP = DF_TP/DF_TP.sum()
DF_TP_Overall = DF_PT.transpose()
DF_TP_Overall['Raw'] = DF_PT.sum()
DF_TP_Overall['Overall'] = DF_PT.sum() / sum(DF_PT.sum())


# Periods - Topics top_10 articles Matrix (sorted by year)
DF_PT_T10A=pd.DataFrame(data='', index=DF_TP.columns,columns=DF_TP.index)
for period in DF_TP.columns:
    for topic in DF_TP.index:
        for idx in DF_topic[DF_topic.Period==period].nlargest(
                10,topic).sort_values('Year',ascending=False).index:
            DF_PT_T10A[topic][period]=DF_PT_T10A[topic][period]+DF_topic.Citation[idx]+'\n'
            
# Topics top_20 articles Matrix by Periods - (sorted by weight)
DF_PT_T20A=pd.DataFrame(data='', index=DF_TP.columns,columns=DF_TP.index)
for period in DF_TP.columns:
    for topic in DF_TP.index:
        for idx in DF_topic.nlargest(20,topic).index:
            if DF_topic.Period[idx]==period:
                DF_PT_T20A[topic][period]=DF_PT_T20A[topic][period]+DF_topic.Citation[idx]+'\n'

#==============================================================================
# ########################################################### Topic correlation
#==============================================================================

# Topic Pearson Correlation
DF_TfromD = df_document_topic.corr(method='pearson')
DF_TfromD_Stack = pd.DataFrame(columns=['Topic_A','Topic_B','Correlation'])
for id1,topic1 in enumerate(topicnames):    
    for id2,topic2 in enumerate(topicnames):
        n_id = DF_TfromD_Stack.shape[0]
        DF_TfromD_Stack.loc[n_id] = [str(id1+1),str(id2+1),DF_TfromD[topic1][topic2]]
##
DF_TfromW=df_topic_keywords.T.corr(method='pearson')
DF_TfromW_Stack = pd.DataFrame(columns=['Topic_A','Topic_B','Correlation'])
for id1,topic1 in enumerate(topicnames):    
    for id2,topic2 in enumerate(topicnames):
        n_id = DF_TfromW_Stack.shape[0]
        DF_TfromW_Stack.loc[n_id] = [str(id1+1),str(id2+1),DF_TfromW[topic1][topic2]]
    
#==============================================================================
# ################################################################ Save results
#==============================================================================
        
# Save lda results to excel
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "0. Data",
                                     "BIG3_LDA_Results_T"+str(ldamodel.n_topics)+"_A"+str(ldamodel.alpha)+".xlsx"))
df_param.T.to_excel(writer,'Para Score',encoding='utf8')        
DF_topic.to_excel(writer,'Doc vs Topic',encoding='utf8')
DF_Topic_TKW.to_excel(writer,'Top 50 Topics Words',encoding='utf8')
df_keywords_topic.to_excel(writer,'Words vs Topics',encoding='utf8',
                           header=topicnames,
                           index=list(vocab))
DF_TP.to_excel(writer,'Topics vs Periods',encoding='utf8')
DF_TP_Overall.to_excel(writer,'Overall Topics vs Periods',encoding='utf8')
DF_PT_T10A.to_excel(writer,'Top 10 articles',encoding='utf8')
DF_PT_T20A.to_excel(writer,'Top 20 articles',encoding='utf8')
df_document_topic_sorted.to_excel(writer,'Doc vs Topic Decreas',encoding='utf8')
DF_TfromD.to_excel(writer,'Topic Cor. from Doc',encoding='utf8')
DF_TfromD_Stack.to_excel(writer,'Topic Cor. from Doc Stack',encoding='utf8')
DF_TfromW.to_excel(writer,'Topic Cor. from Word',encoding='utf8')
DF_TfromW_Stack.to_excel(writer,'Topic Cor. from Word Stack',encoding='utf8')
writer.save()
writer.close()
