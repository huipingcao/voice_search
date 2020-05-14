# WebServer
# Jiefei Liu

import json
import string
import pickle
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# load json data from file
def load():
    with open('arxivData.json') as f:
        data = json.load(f)
        return data


# Preprocess data convert data from dict to string
def PreprocessData_toString(data):
    str = ""
    str += getAuthorName(data["author"]) + data["title"] + data["summary"]
    return str


# Extract Authors name and convert to string
def getAuthorName(author):
    temp = author.split("'")
    result = ""
    index = 3
    while index < len(temp):
        result += temp[index] + ","
        index += 4
    return result


# Load the topic from the pickle file
def GetTopics():
    with open('50_test.pickle', 'rb') as file:
        Topic = pickle.load(file)
    return Topic


# TFIDF preprocess the data
'''Returns 2 values which are 
   Preprocess Name and Preprocess Results
   '''


def TFIDF(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    return denselist


# LDA clean, clean the punctuation and find stop words
def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# LDA method to get the Query topic
def LDA(doc_complete, model):
    doc_clean = [clean(doc).split() for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    """
        Running LDA Mode
        LDA model and train it on Document-Term matrix
    """
    whole_dis = model[doc_term_matrix[0]]
    max = 0
    max_index = -1
    for i in range(len(whole_dis)):
        if whole_dis[i][1] >= max:
            max = whole_dis[i][1]
            max_index = whole_dis[i][0]
    return max_index


# Get the complete vectors of LDA topics
def LDA_complete(doc_complete, model):
    doc_clean = [clean(doc).split() for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    """
        Running LDA Model
        LDA model and train it on Document-Term matrix
    """
    results = []
    for i in range(len(doc_complete)):
        temp_result = model[doc_term_matrix[i]]
        temp = []
        for j in range(len(temp_result)):
            temp.append(temp_result[j][1])
        results.append(temp)
    return results


# Find the results
def GetResult(queryInput, data):
    """
    input: query, string
            The in put data

           data, array
            The whole dataset

           TopicsOfData, dictionary,{key: list[list]}
             The topics of data, key is topic, value is the lists which contains index of the data and probability of this topic

    output: Result, list
            The top 10 data matched query
    """
    Offline = GetTopics()
    TopicsOfData = Offline[1]
    query = queryInput[0]
    lam = queryInput[1]
    lam = float(lam.replace('"', ''))
    lam = lam / 10.0
    document = [query]
    queryTopics = LDA(document, Offline[0])
    Result = []
    MatchedTopics = []
    if queryTopics in TopicsOfData:
        MatchedTopics.append(query)
        # Use LDA to get the vectors of the whole dataset
        whole_LDA_vectors = Offline[2]
        LDA_vectors = []
        LDA_vectors.append(LDA_complete(document, Offline[0])[0])
        for i in TopicsOfData[queryTopics]:
            # i is the matched topic data index
            LDA_vectors.append(whole_LDA_vectors[i[0]])
            MatchedTopics.append(PreprocessData_toString(data[i[0]]))
        # Use cosine function get the similarity of results, cosine will be n*n matrix
        cosine_LDA = cosine_similarity(LDA_vectors)
        # Use TFIDF to train the matched data
        dataVector = TFIDF(MatchedTopics)
        # Use cosine function get the similarity of results, cosine will be n*n matrix
        cosine_TFIDF = cosine_similarity(dataVector)
        # Combine the LDA and TFIDF cosine similarity
        cosine = lam * cosine_LDA + (1-lam) * cosine_TFIDF
        # We only need the first row of the matrix, now the cosine is 1*n-1 array
        cosine = cosine[0, 1:]
        position = []
        Top = 10
        for j in range(Top):
            # find index of max value
            index = np.where(cosine == np.max(cosine))
            # set the max number to be 0
            cosine[index] = 0
            # append the index value into position
            position.append(index[0][0])
        # get the list where was stored in dictionary
        Templist = TopicsOfData[queryTopics]
        # get the index of data base on the position
        for k in position:
            TempIndex = Templist[k][0]
            Result.append(data[TempIndex])
    else:
        print("Does not find the matched topic.")
    return Result


# convert list to JSON format
def listToJson(mylist):
    FinalResult = []
    for i in mylist:
        temp = json.dumps(i)
        FinalResult.append(temp)
    return FinalResult


app = Flask(__name__)


@app.route('/home')
def hello_name():
    return render_template('hello.html')


@app.route("/hello/world", methods=['GET', 'POST'])
def Hello():
    receive_data = request.get_data()
    receive_data = receive_data.decode("utf-8")
    receive_data = receive_data.split(';')
    print(receive_data)
    if len(receive_data[1]) > 2:
        dataset = load()
        result = GetResult(receive_data, dataset)
        return dict(data=result)
    else:
        print("Receive data is empty.")
        results = [{'author': 'Ahmed Osman', 'year': '2018',
                    'name': 'Dual Recurrent Attention Units for Visual Question Answering',
                    'url': 'http://arxiv.org/abs/1802.00209v1'},
                   {'author': 'Ji Young Lee', 'year': '2016',
                    'name': 'Sequential Short-Text Classification with Recurrent and Convolutional',
                    'url': 'http://arxiv.org/abs/1603.03827v1'},
                   {'author': 'Iulian Vlad Serban', 'year': '2016',
                    'name': 'Multiresolution Recurrent Neural Networks: An Application to DialogueResponse Generation',
                    'url': 'http://arxiv.org/abs/1606.00776v2'},
                   {'author': 'Sebastian Ruder', 'year': '2017',
                    'name': 'Learning what to share between loosely related tasks',
                    'url': 'http://arxiv.org/abs/1705.08142v2'},
                   {'author': 'Iulian V. Serban', 'year': '2017', 'name': 'A Deep Reinforcement Learning Chatbot',
                    'url': 'http://arxiv.org/abs/1709.02349v2'},
                   {'author': 'Kelvin Guu', 'year': '2017', 'name': 'Generating Sentences by Editing Prototypes',
                    'url': 'http://arxiv.org/abs/1709.08878v1'},
                   {'author': 'Iulian V. Serban', 'year': '2018',
                    'name': 'A Deep Reinforcement Learning Chatbot (Short Version)',
                    'url': 'http://arxiv.org/abs/1801.06700v1'},
                   {'author': 'Darko Brodic', 'year': '2016',
                    'name': 'Document Image Coding and Clustering for Script Discrimination',
                    'url': 'http://arxiv.org/abs/1609.06492v1'}]
        return dict(data=results)


if __name__ == '__main__':
    app.run(debug=True)
