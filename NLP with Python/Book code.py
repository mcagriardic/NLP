# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:08:34 2019

@author: Cagri
"""

import nltk
nltk.download()
from nltk.book import *

text1

text1.concordance("monstrous")
text2.concordance("affection")

text4.concordance("terror")

text4.dispersion_plot(["terror", "nation"])

sorted_first_50 = sorted(set(text3))[:50]

"""
a measure of the lexical richness of the text.
"""
from __future__ import division
len(text3) / len(set(text3))

fdist1 = FreqDist(text1)
vocabulary1 = fdist1.keys()
vocabulary1[:50]

text4.collocations()










