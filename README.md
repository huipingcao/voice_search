# voice_search

## Introduction 
The Voice System is helping users to find the papers easily, there are two  ways to input query, first way you can input the query into the input bar, the second way is the user can click the "click to speak" bottom and speak the query, the system will translate the voice to text, and text will be used for the input query. And here has another parameter called "Sensitivity", this parameter can control the results to focus more on the words or the topics. 

Here's a Flow Chart for this system shows how does this system works. In the Flow Chart LDA represents "Latent Dirichlet Allocation", TFIDF represents "term frequency–inverse document frequency". 
![image](https://github.com/huipingcao/voice_search/blob/master/Flow_Chart.png)

## Run the System 
Before running this system, please make sure your file directory is exactly same as the Github and the python version should be 3.5 or higher, you also need to install the following python libraries: pickle, gensim, nltk, flask, sklearn. Please make sure you download the whole nltk words libraries, you can use command line: '''nltk.download()''' to download the libraries. 

In order to reduce the processing time, we preprocessed LDA method and stored it into a pickle file, I attached a link of pickle file too, so you do not need to run the Complete_Topic.py again. Due to the size limitation you need to download the original dataset and pickle file and put them into the app file directory. 
You can download the original dataset from : [https://drive.google.com/open?id=1n38NkWdkC8q3YwCJPzP6Lqr7B8094nL7]
And the pickle file from: [https://drive.google.com/open?id=1nS7aLEzRneNtE4ReTX7_TxK_cSDP66Ot]

To run this voice system, you only need to go to the app file directory and use command line: python app.py. And then you will find the url from the command line, copy the url to your browser and add the 'home' at end of the url. For example, the Webserver shows 
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
You can use the url: http://127.0.0.1:5000/home 

And after the system find the results, you can click "Speak Results" to let the system read how many results were found. 
