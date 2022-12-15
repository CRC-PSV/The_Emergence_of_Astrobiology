# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Jun 29 11:44:54 2021
@author: Francis Lareau
This is Project Astrobiology.
Corpus assembly and cleaning
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import pandas as pd
import re
import lxml.html as lh
import nltk
import spacy

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("your_main_path")
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

# import with pandas a dataframe structure
DF_OLEB = pd.read_pickle(os.path.join(main_path,
                                      "0. Data",
                                      "Private",
                                      "DataFrame_OLEB_1970-2020_cleaned.pkl"))

DF_IJA = pd.read_pickle(os.path.join(main_path,
                                     "0. Data",
                                     "Private",
                                     "DataFrame_IJA_2002-2021_cleaned.pkl"))

DF_Astro = pd.read_pickle(os.path.join(main_path,
                                       "0. Data",
                                       "Private",
                                       "DataFrame_Astro_2001_2020_cleaned.pkl"))

#==============================================================================
# ############################################ Function to clean truncated word
#==============================================================================

# Set english vocabulary for nltk
english_vocab_nltk = set(w.lower() for w in nltk.corpus.words.words())
nlp = spacy.load("en_core_web_sm")
def remove_multiline(string, nlp):
    """Function for removing "-", "- " or "-\n" in words. If word's in english 
    dictionnary, it return the word, else it returns "-" without space"""    
    temp_string = re.sub('-\s*','',string)#string.replace("- ","")
    temp_string_lem = nlp(temp_string)[0].lemma_
    check_mot = temp_string_lem.lower() in english_vocab_nltk
    if check_mot == True:
        returned_value = temp_string
    if check_mot == False:
        returned_value = re.sub("\s+","",string)
    return returned_value

def remove_dash(string, nlp):
    """Higher level of function remove_multiline"""
    if re.findall('\w{2,}-\s*\w{2,}',str(string)):        
        list_mot_multiline_string = re.findall("\w{2,}-\s*\w{2,}",string)
        for pattern_multiline in list_mot_multiline_string:
            string = re.sub(pattern_multiline,remove_multiline(pattern_multiline, nlp),string)
        return string
    else:
        return string

#==============================================================================
# ####################################################### Astro data extraction
#==============================================================================

for i in range(len(DF_Astro)):
    #parsing
    tree = lh.fromstring(DF_Astro.loc[i]['HTML'])
    #deleting some element (but keeping tail)
    [e.drop_tree() for e in tree.xpath('//div[@class="article-table-content"]')]
    [e.drop_tree() for e in tree.xpath('//figure[@class="article__inlineFigure"]')]
    [e.drop_tree() for e in tree.xpath('//div[@class="footnote"]')]
    [e.drop_tree() for e in tree.xpath('//a[@class="ref fn"]')]
    #constructing text
    article=''   
    for para in tree.xpath('//div[@class="hlFld-Fulltext"]//*[self::p or self::h2]'):
        para = para.text_content()
        para = ''.join(para)
        para=re.sub('^[\d\.\s]+','',para)
        para=re.sub('\n',' ',para)
        para=re.sub('\s+',' ',para)
        article=article+para+'\n\n'
    #cleaning text
    article=re.sub('\xa0',' ',article) # delet non-breaking space
    article=re.sub('\n\n\n+','\n\n',article) # delet repeating \n\n
    article=re.sub('^Introduction\n\n','',article,flags=re.DOTALL) # delet introduction
    article=re.sub('\n\n.{,3}Appendix.+','',article,flags=re.DOTALL) # delet appendix
    article=re.sub('\n\n.{,3}Acknowledge?ments?.+','',article,flags=re.DOTALL) # delet acknowledgement
    article=re.sub('\n\n.{,3}Abbreviations.+','',article,flags=re.DOTALL) # delet abbreviation
    article=re.sub('\n\n.{,3}Author Disclosure.+','',article,flags=re.DOTALL) # delet disclosure
    article=re.sub('\n\n.{,3}Corresponding author:.+','',article,flags=re.DOTALL) # delet correspondance
    DF_Astro.Article_html[i]=article
    
DF_Astro['Article_pdf']=''
for i in range(len(DF_Astro)):
    if DF_Astro.Article_html[i]=='':
        DF_Astro.Article_pdf[i] = DF_Astro.PDF[i]

#Head
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^.*?Downloaded by Christophe Malaterre from www.liebertpub.com at \d+/\d+/\d+. For personal use only.\n','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^.*?1?\.? ?I[nN][tT][rR][oO][dD][uU][cC][tT][iI][oO][nN][^\n]{,50}\n','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^.{,4000}K[eE][yY] ?[wW][oO][rR][dD][sS]?[^\n]+\n','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^.{,500}\n[A-Z ,\-and]{5,}\n','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^.+?\nAstrobiology LiteratureWatch\n','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^.+?\nCorrections?\n','',x,flags=re.DOTALL))
#Body
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[^\n]{,909}\n[\n\d]+\nDownloaded by Christophe Malaterre from www.liebertpub.com at \d+/\d+/\d+. For personal use only.\n','\n',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n(Downloaded by Christophe Malaterre from www.liebertpub.com at \d+/\d+/\d+. For personal use only.\n)+','\n',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\nFIG\. *\d+[^\n]*',' ',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\nTABLE *\d+[^\n]*',' ',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n\d[A-Z][^\n]{,400}\n','\n',x,flags=re.DOTALL))
#Tail
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[bB][bB][rR][eE][vV][iI][aA][tT][iI][oO][nN][sS]?.+',' ',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[cC][kK][nN][oO][wW][lL][eE][dD][gG][mM][eE][nN][tT][sS]?.+',' ',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*R[eE][fF][eE][rR][eE][nN][cC][eE][sS]?.+',' ',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[pP][pP][eE][nN][dD][iI][xX].+',' ',x,flags=re.DOTALL))
#Last
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('[\t ]+',' ',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n ','\n',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^ ','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n\d[A-Z][^\n]{,400}\n','\n',x,flags=re.DOTALL))#footnote, title
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n\*[A-Z][^\n]{,600}\n','\n',x,flags=re.DOTALL))#footnote
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n+','\n',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+\n','\n',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+$','',x,flags=re.DOTALL))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('^[^a-z]+\n','',x,flags=re.DOTALL))

#Note: article manual fix
for i in [13,33,149,160,170,171,234,245,247,253,272,296,313,317,319,321,332,333,341,365,391,524,551,1142,1334,1470]:
    with open(os.path.join(main_path,
                           "0. Data",
                           "Private",
                           "Astro_cleaning",
                           DF_Astro.PDF_name[i][:-3]+'txt'),encoding='utf_16_le') as f:
        article=f.read()
        article=re.sub('\n+','\n',article)
        DF_Astro.Article_pdf[i]=article

# Cleaning truncated words separated by '-' or '- ' and fix
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: remove_dash(x,nlp))
DF_Astro.Article_pdf=DF_Astro.Article_pdf.apply(lambda x: re.sub('([^\n]{125,})\n+','\g<1>\n\n',x,flags=re.DOTALL))

#==============================================================================
# ######################################################## OLEB data extraction
#==============================================================================

for i in range(len(DF_OLEB)):
    #parsing
    tree = lh.fromstring(DF_OLEB.loc[i]['HTML'])
    #deleting some element (but keeping tail)
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="Abs1"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Résumé"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="Bib1"]')]
    [e.drop_tree() for e in tree.xpath('//section[starts-with(@data-title, "Acknowledg")]')]
    [e.drop_tree() for e in tree.xpath('//section[starts-with(@data-title, "Electronic")]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Notes"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Abbreviations"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Ethics declarations"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="author-information"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="appendices"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Additional information"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Rights and permissions"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="article-info"]')]
    [e.drop_tree() for e in tree.xpath('//div[@class="c-article-section__figure-description"]')]
    #constructing text
    article=''   
    for para in tree.xpath('//main//section//*[self::p or self::h2]'):
        para = para.xpath('./descendant-or-self::*/text()')
        #para = para.text_content()
        para = ''.join(para)
        para=re.sub('^[\d\.\s]+','',para)
        para=re.sub('\s+',' ',para)
        article=article+para+'\n\n'
    #cleaning text
    article=re.sub('\xa0',' ',article) # delet non-breaking space
    article=re.sub('\n\n\n+','\n\n',article) # delet repeating \n\n
    article=re.sub('^Introduction\n\n','',article,flags=re.DOTALL) # delet introduction

    if article != '':
        DF_OLEB.Article_html[i]=article
    else:
        for para in tree.xpath('//div[starts-with(@class, "c-article-section__content")]//*[self::p or self::h2]'):
            para = para.xpath('./descendant-or-self::*/text()')
            para = ''.join(para)
            para=re.sub('^[\d\.\s]+','',para)
            para=re.sub('\s+',' ',para)
            article=article+para+'\n\n'
            #cleaning text
            article=re.sub('\xa0',' ',article) # delet non-breaking space
            article=re.sub('\n\n\n+','\n\n',article) # delet repeating \n\n
            article=re.sub('^Introduction\n\n','',article,flags=re.DOTALL)
        DF_OLEB.Article_html[i]=article

for i in [1906,1913,1918,1921,1932,1963]: #special traitement for exeptions
    #parsing
    tree = lh.fromstring(DF_OLEB.loc[i]['HTML'])
    #deleting some element (but keeping tail)
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="Abs1"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Résumé"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="Bib1"]')]
    [e.drop_tree() for e in tree.xpath('//section[starts-with(@data-title, "Acknowledg")]')]
    [e.drop_tree() for e in tree.xpath('//section[starts-with(@data-title, "Electronic")]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Notes"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Abbreviations"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Ethics declarations"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="author-information"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="appendices"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Additional information"]')]
    [e.drop_tree() for e in tree.xpath('//section[@data-title="Rights and permissions"]')]
    [e.drop_tree() for e in tree.xpath('//section[@aria-labelledby="article-info"]')]
    [e.drop_tree() for e in tree.xpath('//div[@class="c-article-section__figure-description"]')]
    #constructing text
    article=''   
    for para in tree.xpath('//div[starts-with(@class, "c-article-body")]//*[self::p or self::h2]'):
        para = para.xpath('./descendant-or-self::*/text()')
        para = ''.join(para)
        para=re.sub('^[\d\.\s]+','',para)
        para=re.sub('\s+',' ',para)
        article=article+para+'\n\n'
        #cleaning text
        article=re.sub('\xa0',' ',article) # delet non-breaking space
        article=re.sub('\n\n\n+','\n\n',article) # delet repeating \n\n
        article=re.sub('^Introduction\n\n','',article,flags=re.DOTALL)
        DF_OLEB.Article_html[i]=article
        
DF_OLEB['Article_pdf']=''
for i in range(len(DF_OLEB)):
    if DF_OLEB.Article_html[i]=='':
        DF_OLEB.Article_pdf[i] = DF_OLEB.PDF[i]

#Head
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.*?Preface\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.*?1?\.? ?I[nN][tT][rR][oO][dD][uU][cC][tT][iI][oO][nN][^\n]{,50}\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.{,4000}K[eE][yY] ?[wW][oO][rR][dD][sS]?[^\n]+\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.{,4000}A[bB][sS][tT][rR][aA][cC][tT] ?\n[^\n]+\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.{,4000}\s+A[bB][sS][tT][rR][aA][cC][tT]\.[^\n]+\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.{,500}\n[A-Z ,\-and]{5,}\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.{,500}\n1\. *','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.+\n\(Received[^\n]+','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^.{,500}\n[^a-z\n]+and[^a-z\n]+','',x,flags=re.DOTALL))
#Body
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\nFig\. *\d+[^\n]*',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\nT[aA][bB][lL][eE] *\d+[^\n]*',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\*]+[A-Z][a-z][^\n]{,400}\n','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[^\n]+All Rights Reserved\n','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\nCopyright[^\n]+\n','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\nOrigins of Life[^\n]+\n','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('[^\n]{,200}©[^\n]{,200}\n','\n',x,flags=re.DOTALL))
#Tail
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[bB][bB][rR][eE][vV][iI][aA][tT][iI][oO][nN][sS]?.+',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[cC][kK][nN][oO][wW][lL][eE][dD][gG][eE]?[mM][eE][nN][tT][sS]?.+',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*R[eE][fF][eE][rR][eE][nN][cC][eE][sS]?.+',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*B[iI][bB][lL][iI][oO][gG][rR][aA][pP][hH][yY]?.+',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[pP][pP][eE][nN][dD][iI][xX].+',' ',x,flags=re.DOTALL))
#Last
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('[\t ]+',' ',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n ','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^ ','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\*]+ ?[A-Z][a-z ][^\n]{,400}\n','\n',x,flags=re.DOTALL))#footnote, title
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[\d\*]+ ?[A-Z][a-z][^\n]{,400}\n','\n',x,flags=re.DOTALL))#footnote, title
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n+','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+\n','\n',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+$','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('^[^a-z]+\n','',x,flags=re.DOTALL))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+[a-z][^a-z]+\n','\n',x,flags=re.DOTALL))

#Note: article manual fix
for i in [12,14,97,157,188,290,303,553,584,910,927,996,1073,1243,1270,1279,1436,1906,1913,1921,1932,1963]:
    with open(os.path.join(main_path,
                           "0. Data",
                           "Private",
                           "OLEB_cleaning",
                           DF_OLEB.PDF_name[i][:-3]+'txt'),encoding='utf_16_le') as f:
        article=f.read()
        article=re.sub('\n+','\n',article)
        DF_OLEB.Article_pdf[i]=article

# Cleaning truncated words separated by '-' or '- ' and fix
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: remove_dash(x,nlp))
DF_OLEB.Article_pdf=DF_OLEB.Article_pdf.apply(lambda x: re.sub('([^\n]{125,})\n+','\g<1>\n\n',x,flags=re.DOTALL))

#==============================================================================
# ######################################################### IJA data extraction
#==============================================================================

for i in range(len(DF_IJA)):
    #parsing
    tree = lh.fromstring(DF_IJA.loc[i]['HTML'])
    #deleting some element (but keeping tail)
    [e.drop_tree() for e in tree.xpath('//div[@class="caption"]')]
    [e.drop_tree() for e in tree.xpath('//div[@class="fig fig"]')]
    #constructing text
    article=''   
    for para in tree.xpath('//div[@class="body"]//*[self::p or self::h2]'):
        para = para.xpath('./descendant-or-self::*/text()')
        #para = para.text_content()
        para = ''.join(para)
        para=re.sub('^[\d\.\s]+','',para)
        para=re.sub('\s+',' ',para)
        article=article+para+'\n\n'
    #cleaning text
    article=re.sub('\xa0',' ',article) # delet non-breaking space
    article=re.sub('\n\n\n+','\n\n',article) # delet repeating \n\n
    article=re.sub('^\s*Introduction\n\n','',article,flags=re.DOTALL) # delet introduction
    DF_IJA.Article_html[i]=article
    
DF_IJA['Article_pdf']=''
for i in range(len(DF_IJA)):
    if DF_IJA.Article_html[i]=='':
        DF_IJA.Article_pdf[i] = DF_IJA.PDF[i]

#Head
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.{,5000}?1?\.? ?I[nN][tT][rR][oO][dD][uU][cC][tT][iI][oO][nN][^\n]{,50}\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.{,4000}K[eE][yY] ?[wW][oO][rR][dD][sS]?[^\n]+\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.{,4000}\s+A[bB][sS][tT][rR][aA][cC][tT][^\n]+\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.{,1600}\n[A-Z ,\-and]{5,}\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^[^T][^v].{,1000}\n[^a-z]+\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.+\nDOI[^\n]{,200}\n[^\n]{,200}\n[^\n]+\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.+\ndoi[^\n]{,200}\n[^\n]{,200}\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^.{,1000}[eE]-mail[^\n]+','',x,flags=re.DOTALL))
#Body
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\nFig\. *\d+[^\n]*',' ',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\nT[aA][bB][lL][eE] *\d+[^\n]*',' ',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n\d{1,2}\t[A-Z][^\n]{,200}\n','\n',x,flags=re.DOTALL))

#Tail
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('ADVERTISING.+','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[cC][kK][nN][oO][wW][lL][eE][dD][gG][eE]?[mM][eE][nN][tT][sS]?.+','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*R[eE][fF][eE][rR][eE][nN][cC][eE][sS]?.+','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*B[iI][bB][lL][iI][oO][gG][rR][aA][pP][hH][yY]?.+','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[\d\. ]*A[pP][pP][eE][nN][dD][iI][xX][\.:\n].+','',x,flags=re.DOTALL))
#Last
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('[\t ]+',' ',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n +','\n',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^ +','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[\d\*]+ ?[A-Z][a-z ][^\n]{,400}\n','\n',x,flags=re.DOTALL))#footnote, title
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[\d\*]+ ?[A-Z][a-z][^\n]{,400}\n','\n',x,flags=re.DOTALL))#footnote, title
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n+','\n',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+\n','\n',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+$','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('^[^a-z]+\n','',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('\n[^a-z]+[a-z][^a-z]+\n','\n',x,flags=re.DOTALL))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('Downloaded[^\n]+','',x,flags=re.DOTALL))

# Cleaning truncated words separated by '-' or '- ' and fix
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: remove_dash(x,nlp))
DF_IJA.Article_pdf=DF_IJA.Article_pdf.apply(lambda x: re.sub('([^\n]{125,})\n+','\g<1>\n\n',x,flags=re.DOTALL))
    
#==============================================================================
# ###################################################### Concatenate dataframes
#==============================================================================

DF = pd.concat([DF_Astro,DF_OLEB,DF_IJA], ignore_index=True, sort=False, axis=0)
DF['Article'] = DF.apply(lambda row: row.Article_html if row.Article_html!='' else row.Article_pdf, axis=1)
DF=DF[DF.Statut=='yes'] #exclude 'NON'
DF=DF[DF.Year!='2021'] #exclude '2021'
DF.reset_index(inplace=True)

#==============================================================================
# ################################################################### Save Data
#==============================================================================

pd.to_pickle(DF,os.path.join(main_path,
                             "0. Data",
                             "Private",
                             "DF_Consolidation_Astrobiology.pkl"))
