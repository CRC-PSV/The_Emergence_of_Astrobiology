# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Mon Aug 16 11:05:32 2021
@author: Francis Lareau
This is Project Astrobiology.
Topic modeling
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import sys
import datetime
import pandas as pd
import numpy as np
import pickle
import bz2
import lda
import tmtoolkit

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("your_main_path")
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

DF = pd.read_pickle(os.path.join(main_path,
                                 "0. Data",
                                 "Private",
                                 "BIG3_dataframe.pkl"))

with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_vocab_sklearn.pbz2"), "r") as f:
    vocab = pickle.load(f)
    
with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_dtm_sklearn.pbz2"), "r") as f:
    dtm = pickle.load(f)

#==============================================================================
# ################################################################ Multiple LDA
#==============================================================================

#number of topics to consider
k_topics = 25
num_topics = list(range(k_topics+1)[1:]) # Considering 1-k topics, as the last is cut off
num_topics = num_topics + [(x+2)*25 for x in list(range(13))] # add increment of 25

lda_models = {}
time_start = datetime.datetime.now()
for i in num_topics[-2:]:
    ldamodel = lda.LDA(n_topics=i,
                       n_iter=1000, #default 2000                       
                       alpha=5/i, #default 0,1, 50/i in Griffiths TL, Steyvers M (2004). “Finding Scientific Topics.” Proceedings of the National Academy of Sciences of the United States of America, 101, 5228–5235.
                       eta=0.01, #default 0,01, 1/i in sklearn, some suggest 200/words in dict.
                       random_state=1234) 
    lda_models[i] = ldamodel.fit(dtm.toarray().astype(int))
    sys.stdout.write("\rModeling till "+str(i)+" topics took %s"%(str(datetime.datetime.now() - time_start)))
    sys.stdout.flush() 

#==============================================================================
# ################################################################### Coherence
#==============================================================================

roder_2015 = [tmtoolkit.topicmod.evaluate.metric_coherence_gensim(
        measure='c_v',
        topic_word_distrib=lda_models[i].components_,
        vocab=np.array(vocab),
        dtm=dtm, 
        top_n=20,
        texts=DF.Lemma,
        return_mean=True) for i in num_topics] #Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. WSDM 2015 - Proceedings of the 8th ACM International Conference on Web Search and Data Mining, 399–408. 

#==============================================================================
# ############################################################### Other metrics
#==============================================================================
   
alphas = [lda_models[i].alpha for i in num_topics]

betas = [lda_models[i].eta for i in num_topics]

loglikelihoods = [lda_models[i].loglikelihood() for i in num_topics]

perplexities = ['n/a' for i in num_topics]

griffiths_2004 = [tmtoolkit.topicmod.evaluate.metric_griffiths_2004(lda_models[i].loglikelihoods_) for i in num_topics] #Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of the National Academy of Sciences 101, suppl 1: 5228–5235.

cao_2009 = [tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(topic_word_distrib=lda_models[i].components_) for i in num_topics] #Cao Juan, Xia Tian, Li Jintao, Zhang Yongdong, and Tang Sheng. 2009. A density-based method for adaptive LDA model selection. Neurocomputing — 16th European Symposium on Artificial Neural Networks 2008 72, 7–9: 1775–1781.

arun_2010 = [tmtoolkit.topicmod.evaluate.metric_arun_2010(
        topic_word_distrib=lda_models[i].components_, 
        doc_topic_distrib=lda_models[i].doc_topic_, 
        doc_lengths=dtm.sum(axis=1)) for i in num_topics]
    
mimno_2011 = [tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word_distrib=lda_models[i].components_,
        dtm=dtm, 
        top_n=20,
        eps=1, #uses a different epsilon by default (set eps=1 for original)
        normalize = False, #uses a normalizing constant by default (set normalize=False for original)
        return_mean=True) for i in num_topics] #D. Mimno, H. Wallach, E. Talley, M. Leenders, A. McCullum 2011: Optimizing semantic coherence in topic models
    
DF_metrics = pd.DataFrame([num_topics,alphas,betas,loglikelihoods,perplexities,roder_2015,mimno_2011,arun_2010,cao_2009,griffiths_2004]).T
DF_metrics.columns=['k_topics','alpha','beta','loglikelihood','perplexity','roder_2015','mimno_2011','arun_2010','cao_2009','griffiths_2004']
    
#==============================================================================
# ################################################################ Save results
#==============================================================================
    
with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_lda_models_k.pbz2"), "w") as f:
    pickle.dump(lda_models, f, pickle.HIGHEST_PROTOCOL)
    
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "0. Data",
                                     "BIG3_k_metrics.xlsx"))
DF_metrics.to_excel(writer,'Metrics',encoding='utf8')
writer.save()
writer.close()

#==============================================================================
# ################################################################ Multiple LDA
#==============================================================================

#number of alphas to consider
k_topic = 25
num_alphas = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]

lda_models = {}
time_start = datetime.datetime.now()
for i in num_alphas:
    ldamodel = lda.LDA(n_topics=k_topic,
                       n_iter=1000, #default 2000                       
                       alpha=i, #default 0,1, 50/i in Griffiths TL, Steyvers M (2004). “Finding Scientific Topics.” Proceedings of the National Academy of Sciences of the United States of America, 101, 5228–5235.
                       eta=0.01, #default 0,01, 1/i in sklearn, some suggest 200/words in dict.
                       random_state=1234) 
    lda_models[i] = ldamodel.fit(dtm.toarray().astype(int))
    sys.stdout.write("\rModeling till "+str(i)+" topics took %s"%(str(datetime.datetime.now() - time_start)))
    sys.stdout.flush() 
    
#==============================================================================
# ################################################################### Coherence
#==============================================================================

roder_2015 = [tmtoolkit.topicmod.evaluate.metric_coherence_gensim(
        measure='c_v',
        topic_word_distrib=lda_models[i].components_,
        vocab=np.array(vocab),
        dtm=dtm, 
        top_n=20,
        texts=DF.Lemma,
        return_mean=True) for i in num_alphas] #Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. WSDM 2015 - Proceedings of the 8th ACM International Conference on Web Search and Data Mining, 399–408. 

#==============================================================================
# ############################################################### Other metrics
#==============================================================================
   
alphas = [lda_models[i].alpha for i in num_alphas]

betas = [lda_models[i].eta for i in num_alphas]

loglikelihoods = [lda_models[i].loglikelihood() for i in num_alphas]

perplexities = ['n/a' for i in num_alphas]

griffiths_2004 = [tmtoolkit.topicmod.evaluate.metric_griffiths_2004(lda_models[i].loglikelihoods_) for i in num_alphas] #Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of the National Academy of Sciences 101, suppl 1: 5228–5235.

cao_2009 = [tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(topic_word_distrib=lda_models[i].components_) for i in num_alphas] #Cao Juan, Xia Tian, Li Jintao, Zhang Yongdong, and Tang Sheng. 2009. A density-based method for adaptive LDA model selection. Neurocomputing — 16th European Symposium on Artificial Neural Networks 2008 72, 7–9: 1775–1781.

arun_2010 = [tmtoolkit.topicmod.evaluate.metric_arun_2010(
        topic_word_distrib=lda_models[i].components_, 
        doc_topic_distrib=lda_models[i].doc_topic_, 
        doc_lengths=dtm.sum(axis=1)) for i in num_alphas]
    
mimno_2011 = [tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word_distrib=lda_models[i].components_,
        dtm=dtm, 
        top_n=20,
        eps=1, #uses a different epsilon by default (set eps=1 for original)
        normalize = False, #uses a normalizing constant by default (set normalize=False for original)
        return_mean=True) for i in num_alphas] #D. Mimno, H. Wallach, E. Talley, M. Leenders, A. McCullum 2011: Optimizing semantic coherence in topic models
    
DF_metrics = pd.DataFrame([[k_topic for x in num_alphas],alphas,betas,loglikelihoods,perplexities,roder_2015,mimno_2011,arun_2010,cao_2009,griffiths_2004]).T
DF_metrics.columns=['k_topics','alpha','beta','loglikelihood','perplexity','roder_2015','mimno_2011','arun_2010','cao_2009','griffiths_2004']

#==============================================================================
# ################################################################ Save results
#==============================================================================
    
with bz2.BZ2File(os.path.join(main_path,
                              "0. Data",
                              "BIG3_lda_models_alpha.pbz2"), "w") as f:
    pickle.dump(lda_models, f, pickle.HIGHEST_PROTOCOL)
    
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "0. Data",
                                     "BIG3_alpha_metric.xlsx"))
DF_metrics.to_excel(writer,'Metrics',encoding='utf8')
writer.save()
writer.close()

