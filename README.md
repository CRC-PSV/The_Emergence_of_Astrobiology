# The_Emergence_of_Astrobiology
## Abstract
Astrobiology is often defined as the study of the origin, evolution, distribution, and future of life on Earth and in the universe. As a discipline that emerged in the last decades of the 20th century, its contours have not always been straightforward, resulting from the interweaving of several lines of research as early as the 1960s. By applying computational topic-modeling approaches to the complete full-text corpus of three flagship journals in the field, <em>Origins of Life and Evolution of Biospheres</em> (1968-2020), <em>Astrobiology</em> (2001-2020), and the <em>International Journal of Astrobiology</em> (2002-2020), we identify specific topics that characterize the early blossoming of the discipline. We also map their evolution through time, as emphasis changed between different readings of astrobiology, from an exobiology and origins-of-life perspective to a more space- and planetarysciences view of the discipline.

## Requirements
This code was tested on Python 3.7.3. Other requirements are as follows (see requirements.txt):

lda
numpy
pandas
sklearn
tmtoolkit
treetaggerwrapper
##Quick Start
Install libraries: pip install -r requirements.txt
Install TreeTagger
### 1. Corpus assembly and cleaning*
Execute to replicate research : Corpus_assembly_and_cleaning.py
### 2. Preprocessing*
Execute to replicate research : Preprocessing.py
### 3. Topic modeling
#### 3.1 Coherence*
Execute to replicate research : Coherence.py
#### 3.2 LDA
Execute to replicate research : LDA.py
### 3.3 Topic interpretation and correlation
Execute to replicate research : Topic_interpretation_and_correlation.py
### 4. Diachronic modeling
Execute to replicate research : Diachronic_modeling.py
### 5. Journal profiling
Execute to replicate research : Journal_profiling.py
*Note that for legal issues, the complete full-text of journal articles could not be included with the dataset (but can be retrieved by asking the respective publishers).

## Citation
...

## Authors
Christophe Malaterre
Email: malaterre.christophe@uqam.ca
Francis Lareau
Email: francislareau@hotmail.com
## Acknowledgments
F.L. acknowledges funding from the Fonds de recherche du Québec - Société et culture (FRQSC-276470) and the Canada Research Chair in Philosophy of the Life Sciences at UQAM. C.H.P. and C.M. acknowledge from Canada Foundation for Innovation (Grant 34555) and Canada Research Chairs (CRC-950-230795).
