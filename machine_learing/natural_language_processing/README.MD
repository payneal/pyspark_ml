# Natural Language Processings
* this is the field of machine learning that focuses on creating models from a text data source(straight from articles of words).
* very large field of machine learning with its own unique challenges and sets of algos and features so what we cover here will be scratching just the surface

# optional reading suggestions:
* Wikipedia Article on NL
* NLTK Book (seperate python libary)
* Foundations of Statistical Natural Language Processing (Manning)

# examples of NLP
* clustering news articles
* suggesting similar books
* grouping legal Documents
* Analyzing Consumer Feedback
* Spam Email Detection

# basic process for NLP:
* compile all documents(Corpus)
* featurize the words to numerics
* Compare features of documents
* a standard way of doing this is through the use of what is known as "TF-IDF" methods
* TF_IDF stands for Term Frequency-Inverse Document Frequency 

# NLP simple example
* two documents: Blue house, Red house
* Featurize based on word count: 
*   "blue house" -> (red,blue,house) -> (0,1,1)
*   "red house" -> (red,blue,house) -> (1,0,1)
* a document represented as a vector of word counts is called a "Bag of Words"
* these are now vectors in an N-dimensional space, we can compare vectors with  cosine similarity
* we can improve on Bag of Words byadjusting word counts based on their frequency in corpus (the group of all the documents)
* wecan use TF_IDF ( Term Frequency-Inverse Document Frequency)

# term frequency(TF)
* Importance of the term within that document
* TF(x,y) = Number of ocurrences of term x in document y

# Inverse Document Frequenc(IDF):
* Importance of the term in the corpus
* IDF(t) = log(N/dfx) where
* N = total number of documents
* dfx = number of documents with the term

