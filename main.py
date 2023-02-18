import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import lyricsgenius as lg



# importing pickle for models
import pickle

# importing spotify Oauth
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# TOKENS
cid = 'c3e43077142f4e7bb70601747bd1f0d2'
secret = '925f7025c8e6420f9be665401784c0f6'
spotipy_redirect_uri = 'https://google.com'
genius_access_token = 'jVHnDFkBvrvEsj_M8YUxbQ9t8mo2zIG19giBxlX_g7NeRz4W51gedM1YHTBX_U-4'

#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# loading models
ada_model = pickle.load(open('C:\\Users\\91635\\Documents\\SentimentAnalysisOfPlaylist\\models\\ada.pkl','rb'))
gnb_model = pickle.load(open('C:\\Users\\91635\\Documents\\SentimentAnalysisOfPlaylist\\models\\gnb.pkl','rb'))
knn_model = pickle.load(open('C:\\Users\\91635\\Documents\\SentimentAnalysisOfPlaylist\\models\\knn.pkl','rb'))
rf_model = pickle.load(open('C:\\Users\\91635\\Documents\\SentimentAnalysisOfPlaylist\\models\\rf.pkl','rb'))
dt_model = pickle.load(open('C:\\Users\\91635\\Documents\\SentimentAnalysisOfPlaylist\\models\\dt.pkl','rb'))



#*-------------------------------------------------------------------
#*                              START
#*-------------------------------------------------------------------

# Title
st.title('Sentiment Analysis of song playlist')

# sidebar to select models
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ( 'KNN','Gaussian naive bayes', 'AdaBoost','Random Forest',"Decision Tree")
)

playlist_link = st.text_input("Enter the playlist link")
# trial_link = 'https://open.spotify.com/playlist/7un2hQ0LxSRyQpHCQE053M?si=f9e08220eadb4081'
playlist_URI = playlist_link.split("/")[-1].split("?")[0]


tracks = []
artists= []

# function to get playlist songs
def get_playlist(playlist_URI):
    for track in sp.playlist_tracks(playlist_URI)["items"]:
        #Track name
        track_name = track["track"]["name"]
        tracks.append(track_name)
        # Artist Name
        artist_name = track["track"]["artists"][0]["name"]
        artists.append(artist_name)

# when link is entered
if(playlist_link):   
    # fetching the data  
    get_playlist(playlist_URI)   
    df = pd.DataFrame(list(zip(tracks, artists)),
               columns =['track', 'artists'])
    
    #-----------------------------------------
    #           fetching lyrics
    #-----------------------------------------
    genius = lg.Genius(genius_access_token)
    
    l=[]
    for i in range(len(df['track'])):
        song = genius.search_song(title=df['track'][i],artist=df['artists'][i])
        if song != None:
            lyrics = song.lyrics
            l.append(lyrics)
        else:
            l.append(None)

    lyrics_df = pd.DataFrame()
    lyrics_df['lyrics'] = l
    
    # join both data frames
    df = pd.concat([df, lyrics_df], axis=1, join='inner')
    
    
    # ----------------------------------------------------------------
    #                        Pre-processing
    # ----------------------------------------------------------------
    
    # DATA CLEANING
    df.isna().sum()
    df.dropna(how='any',inplace=True)
    df['track'].duplicated().sum()
    df.drop_duplicates(subset='track',keep='last',inplace=True)

    # TOKENIZATION
    import re
    import string
    
    def preprocessText(text, remove_stops=False):
        
        # Remove everything between hard brackets
        text = re.sub(pattern="\[.+?\]( )?", repl='', string=text)

        # Change "walkin'" to "walking", for example
        text = re.sub(pattern="n\\\' ", repl='ng ', string=text)

        # Remove x4 and (x4), for example
        text = re.sub(pattern="(\()?x\d+(\))?", repl=' ', string=text)

        # Fix apostrophe issues
        text= re.sub(pattern="\\x91", repl="'", string=text)
        text = re.sub(pattern="\\x92", repl="'", string=text)
        text= re.sub(pattern="<u\+0092>", repl="'", string=text)
        
        # Make lowercase
        text = text.lower()

        # Remove \n from beginning
        text = re.sub(pattern='^\n', repl=' ', string=text)

        # Strip , ! ?, : and remaining \n from lyrics
        text = ''.join([char.strip(",!?:") for char in text])
        text = text.replace('\n', '  ')
        
        # Remove contractions
        # specific
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"won\’t", "will not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"can\’t", "can not", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"let\’s", "let us", text)
        text = re.sub(r"ain't", "aint", text)
        text = re.sub(r"ain\’t", "aint", text)
        text = re.sub(r"wanna", "want to", text)
        text = re.sub(r"gonna", "going to", text)
        text = re.sub(r"gotta", "go to", text)
        
        # general
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"n\’t", " not", text)
        text = re.sub(r"\’re", " are", text)
        text = re.sub(r"\’s", " is", text)
        text = re.sub(r"\’d", " would", text)
        text = re.sub(r"\’ll", " will", text)
        text = re.sub(r"\’t", " not", text)
        text = re.sub(r"\’ve", " have", text)
        text = re.sub(r"\’m", " am", text)
        text = re.sub(r"\â", "a", text)
        
        # Remove Symbols
        text = re.sub(r"[^\w\s]","", text)
        
        # remove single char
        text = re.sub(r"\b[a-zA-Z]\b","", text)
    
        #remove number
        text = re.sub(r"\d+", "", text)
        
        #remove duplicate char(oohhh,ohhhh,oohhhh,etc)
        text = re.sub(r"o+h", "oh", text)
        text = re.sub(r"oh+", "", text)
        
        # Remove remaining punctuation
        punc = string.punctuation
        text = ''.join([char for char in text if char not in punc])

        # Remove double spaces and beginning/trailing whitespace
        text = re.sub(pattern='( ){2,}', repl=' ', string=text)
        text = text.strip()
        
        return(text)

    df['lyrics'] = df['lyrics'].apply(preprocessText)
    
    # STOP WORDS REMOVAL
    import neattext.functions as nfx
    df['lyrics'] = df['lyrics'].apply(nfx.remove_stopwords)

    st.write(df)
 
    # choosing classifier function
    def get_classifier(clf_name):
        clf = None
        if clf_name == 'adaBoost':
            clf = ada_model
        elif clf_name == 'KNN':
            clf = knn_model
        elif clf_name == 'Random Forest':
            clf = rf_model
        elif clf_name == 'Decision Tree':
            clf = dt_model
        else:
            clf = gnb_model
        return clf

    clf = get_classifier(classifier_name)

    #---------------------------------------------------------------------
    #                       CLASSIFICATION 
    #---------------------------------------------------------------------
    
    
    test_classifier = st.write(f'Classifier = {classifier_name}')
    
    # lyrics into a list
    corpus = []
    for sentence in df['lyrics']:
        corpus.append(sentence)
    
    # count vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(1,2))
    X = cv.fit_transform(corpus).toarray()

    # predicting the sentiment
    y_pred = clf.predict(X)
    
    # displaying the final df
    df[test_classifier] = y_pred.tolist()
