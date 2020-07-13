import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)

from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
embedmodel = SentenceTransformer('bert-base-nli-mean-tokens')
from flask import Flask, request, render_template

from tensorflow.keras.models import load_model

from selenium import webdriver
import cv2
import os
import numpy as np
import re
import nltk
import pickle
import scipy.spatial
import networkx as nx

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from bs4 import BeautifulSoup
import requests
from googlesearch import search


app = Flask(__name__)
model = load_model('D:\\Github Repository\\Fake-News-Classifier\\flask\\model.h5')
data=pickle.load(open('D:\\Github Repository\\Fake-News-Classifier\\flask\\text.pkl', 'rb'))

def DataPreProcess(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    text = text.lower()
    text =text.split()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    data.append(text)
    cv = CountVectorizer(max_features=8000)
    X = cv.fit_transform(data).toarray()
    temp = np.array([X[-1]])
    return temp

# Write Code For Summarizing Article Here
# This function take whole article as parameter and returns summary of that article

def embed(sentence):
    tokens = embedmodel.encode(sentence)
    return tokens


def sentence_similarity(sent1, sent2, stopwords=None):
    
    emb1=embed([' '.join(sent1)])
    emb2=embed([' '.join(sent2)])
    distance = scipy.spatial.distance.cdist(emb1, emb2, "cosine")[0]
    return 1 - distance

def build_similarity_matrix(sentences, stop_words):
   
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if sentences[idx1] == sentences[idx2]:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def summarizer(paragraph, top_n=2):
    print('paragraph = ',paragraph)
    stop_words = stopwords.words('english')
    summarize_text = []


    q=[]
    q.append(paragraph)

    # Step 1 - Read text anc split it
    article = q[0].split(". ")
    sentences = []
    print('****************************************************************')
    for sentence in article:
        # print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    print('???????????????????????????????????????')
    print('sentences = ', sentences)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    print('(((((((((((((((((((((((((((((((((((')
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    print('))))))))))))))))))))))))))))))))))))))))')
    scores = nx.pagerank(sentence_similarity_graph)
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    print("PRINTING SUMMZAIRIZNAJN",". ".join(summarize_text))

    return ". ".join(summarize_text)


def difference(paragraph1, paragraph2):

    q1=embed([paragraph1])
    q2=embed([paragraph2])
    diff = scipy.spatial.distance.cdist(q1,q2, "cosine")[0]
    return 1 - diff


# def related_news(query="google"):
def related_news(query):  
    websites={"link":[],"Title":[],"Heading1":[],"Heading2":[],"Paragraph":[]}
    print('query ', query)
    for link in search(query, tld="co.in", num=3, stop=3, pause=2):
        print('LINK =', link)
        paragraphs=[] 
        source=requests.get(link).text
        soup=BeautifulSoup(source,'lxml')
        print("soup", soup.title)
        # websites["Title"].append()
        if soup.title:
            websites["Title"].append(soup.title.text)
        else:
            websites['Title'].append('No Title provided by source')
        print('website title', websites["Title"])
        if soup.find('h1'):
            # article_heading1=soup.find('h1').text
            websites['Heading1'].append(soup.find('h1').text)
        else:
            websites['Heading1'].append('Header')
        for paras in soup.find_all("p"):
            paragraphs.append(paras.text)
        websites["link"].append(link)
        for i in paragraphs:
            print('i = ', i)
            if len(i)>240:
                websites["Paragraph"].append(i)
                break
        else:
            s=" "
            websites["Paragraph"].append(s.join(paragraphs[0:4]))

    return websites


# New function related news
def single_news(query):  
    # websites={"link":[],"Title":[],"Heading1":[],"Heading2":[],"Paragraph":[]}
    # print('query ', query)
    para = None
    for link in search(query, tld="co.in", num=3, stop=3, pause=2):
        # print('LINK =', link)
        paragraphs=[] 
        source=requests.get(link).text
        soup=BeautifulSoup(source,'lxml')
        # print("soup", soup.title)
        # websites["Title"].append(soup.title.text)
        # print('website title', websites["Title"])
        # article_heading1=soup.find('h1').text
        
        for paras in soup.find_all("p"):
            paragraphs.append(paras.text)
        para=''.join(paragraphs)
        # print('************************************para = ', para)
        # websites["link"].append(link)
        # websites["Heading1"].append(article_heading1)
        # for i in paragraphs:
        #     print('i = ', i)
        #     if len(i)>240:
        #         websites["Paragraph"].append(i)
        #         break
        # else:
        #     s=" "
        #     websites["Paragraph"].append(s.join(paragraphs[0:4]))

    return para
# End of function



@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/icons.html')
def icons():
    return render_template('icons.html')

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/news.html')
def news():
    return render_template('news.html')

@app.route('/notifications.html')
def notifications():
    return render_template('notifications.html')

@app.route('/tables.html')
def tables():
    return render_template('tables.html')

@app.route('/typography.html')
def typography():
    return render_template('typography.html')

@app.route('/upgrade.html')
def upgrade():
    return render_template('upgrade.html')

@app.route('/user.html')
def user():
    return render_template('user.html')

@app.route('/whatsapp.html')
def whatsapp():
    return render_template('whatsapp.html')

@app.route('/article_user', methods=['GET', 'POST'])
def article_user():
    if request.method == 'POST':
        article=request.form["text"]
        summary=summarizer(article)
        websites=related_news(summary)
        diff=difference(article, single_news(summary))[0]*100
        diff=round(diff, 2)
        s=str(diff)+' % match found.'
        # print('single_news(summary) = ', single_news(summary))
        # print("Diff = ", difference(article, single_news(summary)))
        # print('summary = ', summary)
    return render_template('user.html', news=websites, diff=s)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = [x for x in request.form.values()]
        print('text from html =',text)
        print('Type of text from html =',type(text))
        s=text[0]
        text = DataPreProcess(text[0])
        print('text after DataPreProcess =', text)
        output = model.predict_classes(text)
        print('Output = ',output)
        # 1- Unreliable
        # 0- Reliable
        if output[0][0]==1:
            prediction_text = 'This seems to be fake'
            websites=''
        else:
            prediction_text = 'This seems to be real'
            article=request.form["text"]
            summary=summarizer(article)
            websites=related_news(summary) 
    return render_template('index.html', prediction_text=prediction_text,news=websites)
    
file_img = None
UPLOAD_FOLDER = 'D:/Github Repository/Fake-News-Classifier/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/whatsappp',methods=['GET', 'POST'])
def whatsappp():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']

        path = os.path.join(app.config['UPLOAD_FOLDER'],file1.filename)
        file1.save(path)
        file_img = file1.filename
        img_url = passageForLink(imageUrl(path))
        print('img url = ', img_url)
        diff = difference(img_url, (request.form["text"]))
        # print('Request form', summarizer_img(request.form["text"]))
        # print('SUmmarizer for passageforlink',summarizer_img(passageForLink(imageUrl(path))))
        print('DIFFERENCE in WHatsapp = ', diff)
        ne=related_news(summarizer(img_url))
        if diff > 0.48:
            text = 'It seems real'
        else:
            text = 'It seems fake'
        # return request.form["text"]
    return render_template('whatsapp.html', prediction_text= text, news=ne)

def imageUrl(filepath):
    searchUrl="http://www.google.hr/searchbyimage/upload"
    multipart={'encoded_image':(filepath,open(filepath,'rb')),'image_content':''}
    response=requests.post(searchUrl,files=multipart,allow_redirects=False)
    fetchUrl=response.headers['Location']
    driver = webdriver.Chrome("D:\Github Repository\Fake-News-Classifier\chromedriver_win32\chromedriver.exe")
    driver.get(fetchUrl)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    link=""
    i=0
    for divs in soup.find_all("div",attrs={'class':'r'}):
        for link in divs.find_all("a"):
            if link['href']!="#":
                i=1
                link=link['href']
                break
        if i==1:
            break
    return link

def passageForLink(link):
    source=requests.get(link).text
    soup=BeautifulSoup(source,'lxml')
    paragraphs=[]
    for paras in soup.find_all("p"):
        print('paras = ', paras)
        paragraphs.append(paras.text)
    return " ".join(paragraphs[0:(len(paragraphs)//3)])


def summarizer_img(paragraph):
    stop_words = stopwords.words('english')
    summarize_text = []
    q=[]
    q.append(paragraph)

    # Step 1 - Read text anc split it
    article = q[0].split(". ")
    
    sentences = []
    print('Inside summarizer')
    for sentence in article:
        # print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    
    for i in range(1):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    print("PRINTING SUMMZAIRIZNAJN",". ".join(summarize_text))

    return ". ".join(summarize_text)


if __name__ == '__main__':
    app.run(debug=True)