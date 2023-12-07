# N-Gram Sentiment Analysis
This project aims to perform sentiment analysis using N-gram models in Natural Language Processing (NLP). It utilizes various N-gram techniques to analyze the sentiment of text data.

## Overview
Sentiment analysis is the process of determining the sentiment or opinion expressed in a piece of text. In this project, we leverage N-gram models to analyze the sentiment of textual data. N-grams are sequences of N words that are used to understand the context and relationships between words in a text.

## Features
N-gram Analysis: Utilizes N-grams (unigrams, bigrams, trigrams) to capture patterns and context in the text.
Sentiment Classification: Classifies text into negative sentiments based on the analyzed N-gram features. This is due to its focus being to analyze and identify cyberbullying and hate speech from the tweets in the dataset.
Dataset: A sample dataset is included for demonstration purposes. The dataset is extracted from [Kaggle] (https://drive.google.com/uc?id=1iKVted6TqDRk6cYnP4vuQ-Gix2KPASkO). In this dataset, it is classified into cyberbullying and not cyberbullying tweets in separate columns. This eases the process of sentiment analysis using the n-gram analysis (unigram, bigram and trigram)

## Library Requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import spacy

# Contibutors
  1. Veytri
  2. Eileen
