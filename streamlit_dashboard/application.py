import streamlit as st
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
import time
from tkinter import Tk, RIGHT, BOTH, RAISED
from tkinter.ttk import Frame, Button, Style

#progress bar
with st.spinner('Wait for it...'):
    time.sleep(5)
    st.success('Done!')
#about the project(information)

def main():
    html_temp = """
  <title>COVID-19 Tweets Analysis</title>
    <h2 >The Problem </h2></div>
    <br>
   <div>
    Sentiment classification is a classic problem in NLP regarding understanding whether a sentence is positive or negative. In this visualizer,we're displaying the ratio of the multitude of emotions that people were going through,when they composed these tweets related to the global pandemic of COVID-19. This is just a small attempt at understanding how the general public feels about the changing,uncertain times we live in.
       
   </div>
<br><br><br>
   <div style="background-color:#212529 ;padding:3px">
    <h2 style="color:white;text-align:center;">The Team </h2>
    <h4 style="color:white;text-align:center;"> <p>Swati Thakur</p><p>  Shaona Kundu</p> <p> Antaripa Saha</p><p>  Indrashis Mitra</p></h4>
    </div>


    """
    st.markdown(html_temp, unsafe_allow_html=True)

if __name__=='__main__':
    main()

#providing title

st.title("SENTIMENT ANALYSIS OF COVI-19 TWEETS🦠")
st.sidebar.title("Sentiment Analysis")
st.sidebar.markdown("Streamlit Dashboard To Analyze The Sentiments")

st.markdown(os.listdir())

data = pd.read_csv('Output.csv')  #read the file

@st.cache(persist=True)
def load_data():
    return data

data = load_data()

#providing subheader
st.header("ANALYSIS:")

#hide the hamburger menu provided by the streamlit
#hide the made by streamlit mark(as per your requirements)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

#providing colour to the writeups( you may adjust it as per ur way)
st.markdown('<style>h1{color: black;}</style>', unsafe_allow_html=True)
st.markdown('<style>h2{color: black;}</style>', unsafe_allow_html=True)   
st.markdown('<style>h3{color: black;}</style>', unsafe_allow_html=True)   
#importing images 
from PIL import Image
image = Image.open('corona.jpg')
st.image(image, caption='corona',
       use_column_width=True)



#number of tweets by sentiments
st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['Polarity'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)

        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)
        st.markdown(str(sys.path))

#breakdown the  sentiments
df = data.copy()
@st.cache(persist=True)

def plot_sentiment(Polarity):
    df['Polarity'][df['Polarity']!=Polarity] = 'Others'
    # df = data
    count = df['Polarity'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
    return count        



st.sidebar.subheader("Breakdown sentiments")
choice = st.sidebar.multiselect('Pick sentiments', ('Positive','Negative','Neutral'))
if len(choice) > 0:
    st.subheader("Breakdown sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='2')
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_3.update_layout(height=500, width=700)
        st.plotly_chart(fig_3)
    else:
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                    i+1, j+1
                )
        fig_3.update_layout(height=500, width=700)
        st.plotly_chart(fig_3)

#wordcloud (each for positive, negative and neutral)


def Polarity_score(data):
  if data['Polarity'] == 0:
    return 'Neu'
  elif data["Polarity"] > 0:
    return 'Pos'
  else:
    return 'Neg'
    st.markdown(data)

#sidebar header fro the word cloud
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for ',('Positive','Negative','Neutral'))
if not st.sidebar.checkbox("Close", True, key='3'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['Polarity']==word_sentiment]
    df = df.applymap(str)
    words = ' '.join(df['cleaned_text'])
    
    # st.markdown(words)
    wc = WordCloud(background_color='black',width=1600,height=640)

    try:
        cloud = wc.generate(words)
        plt.imshow(cloud)

        st.pyplot()
    except:
        st.markdown(f'No words found with {word_sentiment} sentiment.')
