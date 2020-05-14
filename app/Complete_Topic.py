# Jiefei Liu
# Get the complete topics for the dataset and store them into pickle

import json
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import string
import gensim
import pickle
import time


def load():
    with open('arxivData.json') as f:
        data = json.load(f)
        return data


def PreprocessData_tolist(data):
    result = []
    size = len(data)
    for index in range(size):
        str = ""
        str += (getAuthorName(data[index]["author"]))
        str += (data[index]["title"])
        str += (data[index]["summary"])
        result.append(str)
    return result


def getAuthorName(author):
    temp = author.split("'")
    result = ""
    index = 3
    while index < len(temp):
        result += temp[index] + ", "
        index += 4
    return result


# LDA Preprocess the input
def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def LDA(dataset):
    doc_complete = PreprocessData_tolist(dataset)
    doc_clean = [clean(doc).split() for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    """
        Running LDA Model
        LDA model and train it on Document-Term matrix
    """
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=100, id2word=dictionary, passes=50, minimum_probability=0.0)
    results = []            # Store the results
    model = ldamodel        # Store the LDA model
    com_topic = []          # com_topics
    dict_index = {}         # 20 topics stores in dict
    # Go through the document one by one
    for i in range(len(doc_complete)):
        # training the model
        temp_result = ldamodel[doc_term_matrix[i]]
        max_value = 0
        max_index = -1
        temp = []
        # for each document
        for j in range(len(temp_result)):
            # Get the max value with index
            if temp_result[j][1] >= max_value:
                max_value = temp_result[j][1]
                max_index = temp_result[j][0]
            # get the prob for each document
            temp.append(temp_result[j][1])
        com_topic.append(temp)
        # Build the dictionary with the key is topic value is a list of the max_index with prob
        temp_list = [i, max_value]
        if max_index in dict_index:
            dict_index[max_index].append(temp_list)
        else:
            dict_index[max_index] = [temp_list]
    results.append(model)
    results.append(dict_index)
    results.append(com_topic)
    return results


# Load the complete topic from the pickle file
def GetCompleteTopic():
    with open('complete_topic.pickle', 'rb') as file:
        Complete_Topic = pickle.load(file)
    return Complete_Topic


# Load the topic from the pickle file
def GetTopics():
    with open('LDA.pickle', 'rb') as file:
        Topic = pickle.load(file)
    return Topic




"""main"""
start = time.time()
data = load()
'''
result = LDA(data)
"""Store the result into pickle file"""
file = open('100_test.pickle', 'wb')
pickle.dump(result, file)
file.close()
'''
#LDA_vectors = GetCompleteTopic()
#LDA_topic = GetTopics()
end = time.time()
print("running time = ", end-start)
