import json
import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams 
from nltk import everygrams
from textblob import TextBlob
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
from nltk.tokenize import MWETokenizer
tokenizer = MWETokenizer(separator=' ')
import warnings
warnings.filterwarnings("ignore")
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import datetime as dt
import os
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentifish import Sentiment
from tqdm import tqdm
#%load_ext autotime
vader = SentimentIntensityAnalyzer()
from afinn import Afinn
import pymssql
from datetime import datetime as dt  
import re
from email.mime.multipart import MIMEMultipart
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib
from email import encoders 
from datetime import datetime

af = Afinn()

from wordphrasesupd import matcher


connection1 = pymssql.connect(server='100.22.1.57',user='aravindh',password='c4o@ER',database ='djuboreview',port='1443')

query1 ='select hotelcode,review_content,review_header,sentiment_score,polarity,properties,reviewid_new,id from djuboreview.dbo.reviewdetails_bak with (nolock) where dtcollected ='+dt.today().strftime("'%Y-%m-%d'")

review=pd.read_sql(query1,connection1)





review['strength']=''

review['sentiment_score']=0

review['polarity']=''

review['properties']=''

review.columns

review['hotelcode'].nunique()

Corpus=pd.read_excel('trainingset_combined1_02222022.xlsx')




Corpus=Corpus[Corpus['text']!='served']







# training_set
Train_X = Corpus['text_final']
Train_Y = Corpus['class']
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Tfidf_vect = TfidfVectorizer(max_features=3000)
Tfidf_vect.fit(Corpus['text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)

# model

def modelbuilding(klm):
    fdata=pd.DataFrame(klm)
    fdata.rename(columns={0:"start"},inplace=True)
    fdata.rename(columns={1:"end"},inplace=True)
    fdata.rename(columns={2:"text"},inplace=True)
    fdata['text']=fdata['text'].astype(str)
    fdata.drop_duplicates('text',keep='first',inplace=True)
    fdata.drop_duplicates('start',keep='last',inplace=True)
    fdata.drop_duplicates('end',keep='first',inplace=True)
    fdata.reset_index(inplace = True, drop = True)

    ffq=[]
    for n in range(0,len(fdata)):
        fq = re.sub("[^A-Za-z" "]+"," ",fdata['text'][n])
        ffq.append(fq)
    fdata['content']=pd.DataFrame(ffq)
    fdata['content'].dropna(inplace=True)
    fdata['content'] = [entry.lower() for entry in fdata['content']]
    fdata['content']= [word_tokenize(entry) for entry in fdata['content']]
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(fdata['content']):
        Final_words1 = []
        word_Lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            #if word not in stopwords.words('english') and word.isalpha():
            word_Final1 = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words1.append(word_Final1)
        fdata.loc[index,'text_final'] = str(Final_words1)

    Test_X=fdata['text_final']
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    model = DecisionTreeClassifier(criterion = 'entropy')
    model.fit(Train_X_Tfidf,Train_Y)
    predictions_DT = model.predict(Test_X_Tfidf)
    fdata["pred_DT"]= predictions_DT

    fdata.loc[fdata['pred_DT']==0,'class']='bathroom'
    fdata.loc[fdata['pred_DT']==1,'class']='bed'
    fdata.loc[fdata['pred_DT']==2,'class']='cleanliness'
    fdata.loc[fdata['pred_DT']==3,'class']='elevator'
    fdata.loc[fdata['pred_DT']==4,'class']="facilities & surroundings"
    fdata.loc[fdata['pred_DT']==5,'class']="fnb & bar"
    fdata.loc[fdata['pred_DT']==6,'class']="fnb & food"
    fdata.loc[fdata['pred_DT']==7,'class']="fnb & restaurant"
    fdata.loc[fdata['pred_DT']==8,'class']="internet"
    fdata.loc[fdata['pred_DT']==9,'class']="location"
    fdata.loc[fdata['pred_DT']==10,'class']="tranquility(noise-free)"
    fdata.loc[fdata['pred_DT']==11,'class']="parking"
    fdata.loc[fdata['pred_DT']==12,'class']="room"
    fdata.loc[fdata['pred_DT']==13,'class']="staff"
    fdata.loc[fdata['pred_DT']==14,'class']="value for money"
    fdata.loc[fdata['pred_DT']==15,'class']="z"

    fdata.drop(['start', 'end','content','pred_DT'], axis=1,inplace=True)
    fdata=fdata[fdata['class']!='z']
    fdata.reset_index(inplace = True, drop = True)

    return fdata



def store(rev,i_d):
    ijk=rev
    df=pd.DataFrame(columns=['sentiment_score','strength','polarity','properties','id'])
    if ijk==0:
        df.loc[0,'sentiment_score']=1000
        df.loc[0,'strength']=1000
        df.loc[0,'polarity']=1000
        df.loc[0,'properties']=1000
        df.loc[0,'id']=i_d

    else:
        ijk=pd.Series(ijk).str.replace(',','.').astype(str)
        ijk=pd.Series(ijk).str.replace('.','. ').astype(str)
        ijk=ijk[0]
        doc=nlp(ijk)
        matches = matcher(doc)
        klm=[]       
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  
            k=(start),( end),(span.text)
            klm.append(k)

        if len(klm)>=1:
            fdata=modelbuilding(klm)
            if len(fdata)>0:
                fdata['polarity']=0.0
                fdata['sentifish_score']=np.nan
                fdata['textblob_score']=np.nan
                fdata['vader_score']=np.nan
                fdata['afinn_score']=np.nan
                for i in range(0,len(fdata)):
                    try:
                        fdata['sentifish_score'][i]=Sentiment(fdata['text'][i]).analyze( )
                        fdata['textblob_score'][i]=TextBlob(fdata['text'][i]).sentiment.polarity 
                        fdata['afinn_score'][i]=af.score(fdata['text'][i])
                        #fdata['vader_score'][i]=vader.polarity_scores(fdata['text'][i])['compound']
                    except:
                        fdata['sentifish_score'][i]=TextBlob(fdata['text'][i]).sentiment.polarity
                        fdata['textblob_score'][i]=fdata['sentifish_score'][i]
                        fdata['afinn_score'][i]=af.score(fdata['text'][i])
                        #fdata["flair_score"] = fdata["text"].apply(flair_prediction)
                        #fdata['vader_score'][i]=vader.polarity_scores(fdata['text'][i])['compound']
                        
                fdata['afinn_score']=fdata['afinn_score']/10
                fdata['polarity']=(fdata['sentifish_score']+fdata['textblob_score']+fdata['afinn_score'])
                fdata['polarity']=round(fdata['polarity'], 3)
                fdata.loc[fdata['polarity']< 0,'sentiment']="negative"
                fdata.loc[(fdata['polarity']> 0.10),'sentiment']="positive"
                fdata.loc[(fdata['polarity']<=0.10) & (fdata['polarity']>= 0),'sentiment']="neutral"
                fdata=fdata[(fdata['text']!='good reviews')&(fdata['text']!='Negative')&(fdata['text']!='We booked this property')&(fdata['text']!='seeing good')&(fdata['text']!='very old property')&(fdata['text']!='need renovation)c')&(fdata['text']!='better')]
                fdata['polarity']=round(fdata['polarity'], 2)
                kk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class']).polarity.mean().reset_index()
                aaab=pd.DataFrame(kk.groupby(['class']).polarity.mean()).to_json(orient='index')
                kkk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class','sentiment']).polarity.mean().reset_index()
                asd=kkk.rename(columns={"class": "properties", "sentiment": "polarity",'polarity':'strength'}).to_dict('records') 
                

                df.loc[0,'properties']=str(aaab)
                df.loc[0,'strength']=str(asd)
                df.loc[0,'id']=i_d
                
            else:
                df.loc[0,'strength']=1000
                df.loc[0,'properties']=1000
                df.loc[0,'id']=i_d
            
        else:
            df.loc[0,'strength']=1000
            df.loc[0,'properties']=1000
            df.loc[0,'id']=i_d
   
    return df

review.loc[review['review_content']==0,'review_content']='[]'
review.loc[review['sentiment_score']==1000,'sentiment_score']='[]'
review.loc[review['strength']==1000,'strength']='[]'
review.loc[review['polarity']==1000,'polarity']='[]'
review.loc[review['properties']==1000,'properties']='[]'

#review=review[review['Review_Header'].notnull()]

review.loc[review['review_content'].isnull(),'review_content']=0
review['sentiment_score']=review['sentiment_score'].astype(float)

ree=review[(review['review_content']!='0')&(review['review_content']!='There are no comments available for this review')]




ss=ree['id'].tolist()
ss1=ree['review_content'].tolist()





import multiprocessing as mp
from time import sleep

pool = mp.Pool(35)
examp = [pool.apply_async(store,args=(s,s1)) for s,s1 in tqdm(zip(ss1,ss))]
examp = [r.get() for r in examp]
pool.close()
pool.join()



df33=pd.concat(examp)

def store(rev,i_d):
    ijk=rev
    df=pd.DataFrame(columns=['sentiment_score','strength','polarity','properties','id'])
    if ijk==0:
        df.loc[0,'sentiment_score']=1000
        df.loc[0,'strength']=1000
        df.loc[0,'polarity']=1000
        df.loc[0,'properties']=1000
        df.loc[0,'id']=i_d

    else:
        ijk=pd.Series(ijk).str.replace(',','.').astype(str)
        ijk=pd.Series(ijk).str.replace('.','. ').astype(str)
        ijk=ijk[0]
        doc=nlp(ijk)
        matches = matcher(doc)
        klm=[]       
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  
            k=(start),( end),(span.text)
            klm.append(k)

        if len(klm)>=1:
            fdata=modelbuilding(klm)
            if len(fdata)>0:
                fdata['polarity']=0.0
                fdata['sentifish_score']=np.nan
                fdata['textblob_score']=np.nan
                fdata['vader_score']=np.nan
                fdata['afinn_score']=np.nan
                for i in range(0,len(fdata)):
                    try:
                        fdata['sentifish_score'][i]=Sentiment(fdata['text'][i]).analyze( )
                        fdata['textblob_score'][i]=TextBlob(fdata['text'][i]).sentiment.polarity 
                        fdata['afinn_score'][i]=af.score(fdata['text'][i])
                        #fdata['vader_score'][i]=vader.polarity_scores(fdata['text'][i])['compound']
                    except:
                        fdata['sentifish_score'][i]=TextBlob(fdata['text'][i]).sentiment.polarity
                        fdata['textblob_score'][i]=fdata['sentifish_score'][i]
                        fdata['afinn_score'][i]=af.score(fdata['text'][i])
                        #fdata["flair_score"] = fdata["text"].apply(flair_prediction)
                        #fdata['vader_score'][i]=vader.polarity_scores(fdata['text'][i])['compound']
                        
                fdata['afinn_score']=fdata['afinn_score']/10
                fdata['polarity']=(fdata['sentifish_score']+fdata['textblob_score']+fdata['afinn_score'])
                fdata['polarity']=round(fdata['polarity'], 3)
                fdata.loc[fdata['polarity']< 0,'sentiment']="negative"
                fdata.loc[(fdata['polarity']> 0.10),'sentiment']="positive"
                fdata.loc[(fdata['polarity']<=0.10) & (fdata['polarity']>= 0),'sentiment']="neutral"
                fdata=fdata[(fdata['text']!='good reviews')&(fdata['text']!='Negative')&(fdata['text']!='We booked this property')&(fdata['text']!='seeing good')&(fdata['text']!='very old property')&(fdata['text']!='need renovation)c')&(fdata['text']!='better')]
                fdata['polarity']=round(fdata['polarity'], 2)
                kk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class']).polarity.mean().reset_index()
                aaab=pd.DataFrame(kk.groupby(['class']).polarity.mean()).to_json(orient='index')
                kkk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class','sentiment']).polarity.mean().reset_index()
                asd=kkk.rename(columns={"class": "properties", "sentiment": "polarity",'polarity':'strength'}).to_dict('records') 
                

                df.loc[0,'properties']=str(aaab)
                df.loc[0,'strength']=str(asd)
                df.loc[0,'id']=i_d
                
            else:
                df.loc[0,'strength']=1000
                df.loc[0,'properties']=1000
                df.loc[0,'id']=i_d
            
        else:
            df.loc[0,'strength']=1000
            df.loc[0,'properties']=1000
            df.loc[0,'id']=i_d
   
    return df

review.loc[review['review_content']==0,'review_content']='[]'
review.loc[review['sentiment_score']==1000,'sentiment_score']='[]'
review.loc[review['strength']==1000,'strength']='[]'
review.loc[review['polarity']==1000,'polarity']='[]'
review.loc[review['properties']==1000,'properties']='[]'

ret=review[(review['review_content']=='[]')|(review['review_content']==0)|(review['review_content']=='There are no comments available for this review')]




ret['review_header']=ret['review_header'].astype(str)

ss=ret['id'].tolist()
ss1=ret['review_header'].tolist()



import multiprocessing as mp
from time import sleep

pool = mp.Pool(35)
examp = [pool.apply_async(store,args=(ss,ss1)) for ss,ss1 in tqdm(zip(ss1,ss))]
examp = [r.get() for r in examp]
#res.append(results)
pool.close()
pool.join()

df34=pd.concat(examp)





review.drop(['sentiment_score', 'strength', 'polarity', 'properties'], axis=1,inplace=True)

ff=df33.merge(review, on='id')

ff1=df34.merge(review, on='id')

ff['review_content']=ff['review_content'].astype(str)



kk=ff

ff=pd.DataFrame()

kk['sentiment_score']=np.nan
kk['sentifish_score']=np.nan
kk['textblob_score']=np.nan
kk['vader_score']=np.nan
kk['afinn_score']=np.nan

kk=kk.reset_index()



for i in range(0,len(kk)):
    try:
        kk['sentifish_score'][i]=Sentiment(str(kk['review_content'])[i]).analyze( )
        kk['textblob_score'][i]=TextBlob(str(kk['review_content'])[i]).sentiment.polarity 
        kk['afinn_score'][i]=af.score(str(kk['review_content'])[i])
        #kk['vader_score'][i]=vader.polarity_scores(str(kk['review_content'])[i])['compound']
    except:
        kk['sentifish_score'][i]=TextBlob(str(kk['review_content'])[i]).sentiment.polarity
        kk['textblob_score'][i]=kk['sentifish_score'][i]
        kk['afinn_score'][i]=af.score(str(kk['review_content'])[i])
        #kk['vader_score'][i]=vader.polarity_scores(str(kk['review_content'])[i])['compound']

kk['afinn_score']=kk['afinn_score']/10
kk['sentiment_score']=(kk['sentifish_score']+kk['textblob_score']+kk['afinn_score'])/3

kk['sentiment_score']=round(kk['sentiment_score'], 2)

kk.loc[kk['sentiment_score']< 0,'polarity']="negative"
kk.loc[kk['sentiment_score']> 0.10,'polarity']="positive"
kk.loc[(kk['sentiment_score']<= 0.10) & (kk['sentiment_score']>= 0),'polarity']="neutral"

kk.loc[(kk['sentiment_score']==1000),'sentiment_score']="[]"
kk.loc[(kk['properties']==1000),'properties']="[]"



kk.loc[(kk['strength']==1000),'strength']="[]"




kk=kk[['sentiment_score', 'strength', 'polarity', 'properties',
       'hotelcode',
       'review_content','id','review_header']]







ff1['review_content']=ff1['review_content'].astype(str)

kk1=ff1

ff1=pd.DataFrame()

kk1['sentiment_score']=np.nan
kk1['sentifish_score']=np.nan
kk1['textblob_score']=np.nan
#kk1['vader_score']=np.nan
kk1['afinn_score']=np.nan

kk1=kk1.reset_index()



for i in range(0,len(kk1)):
    try:
        kk1['sentifish_score'][i]=Sentiment(str(kk1['review_header'][i])).analyze( )
        kk1['textblob_score'][i]=TextBlob(str(kk1['review_header'][i])).sentiment.polarity 
        kk1['afinn_score'][i]=af.score(str(kk1['review_header'][i]))
        #kk1['vader_score'][i]=vader.polarity_scores(str(kk1['review_header'][i]))['compound']
    except:
        kk1['sentifish_score'][i]=TextBlob(str(kk1['review_header'][i])).sentiment.polarity
        kk1['textblob_score'][i]=kk1['sentifish_score'][i]
        kk1['afinn_score'][i]=af.score(str(kk1['review_header'][i]))
        #kk1['vader_score'][i]=vader.polarity_scores(str(kk1['review_header'][i]))['compound']

kk1['afinn_score']=kk1['afinn_score']/10
kk1['sentiment_score']=(kk1['sentifish_score']+kk1['textblob_score']+kk1['afinn_score'])/3

kk1['sentiment_score']=round(kk1['sentiment_score'], 2)

kk1.loc[kk1['sentiment_score']< 0,'polarity']="negative"
kk1.loc[kk1['sentiment_score']>= 0.10,'polarity']="positive"
kk1.loc[(kk1['sentiment_score']< 0.10) & (kk1['sentiment_score']>= 0),'polarity']="neutral"

kk1.loc[(kk1['sentiment_score']==1000),'sentiment_score']="[]"
kk1.loc[(kk1['properties']==1000),'properties']="[]"



kk1.loc[(kk1['strength']==1000),'strength']="[]"

kk1=kk1[['sentiment_score', 'strength', 'polarity', 'properties',
       'hotelcode',
       'review_content','id','review_header']]

kk1.loc[kk1['review_header'].str.contains('Superb',case=False,na=False),'properties']='[]'
kk1.loc[kk1['review_header'].str.contains('Superb',case=False,na=False),'strength']='[]'
kk1.loc[kk1['review_header'].str.contains('Exceptional',case=False,na=False),'polarity']='positive'
kk1.loc[kk1['review_header'].str.contains('Very good',case=False,na=False),'properties']='[]'
kk1.loc[kk1['review_header'].str.contains('Very good',case=False,na=False),'strength']='[]'
kk1.loc[kk1['review_header'].str.contains('Poor',case=False,na=False),'properties']='[]'
kk1.loc[kk1['review_header'].str.contains('Poor',case=False,na=False),'strength']='[]'

kk1.loc[kk1['review_header'].str.contains('Very good',case=False,na=False),'polarity']='positive'
kk1.loc[kk1['review_header'].str.contains('Poor',case=False,na=False),'polarity']='negative'
kk1.loc[kk1['review_header'].str.contains('Good',case=False,na=False),'polarity']='positive'
kk1.loc[kk1['review_header'].str.contains('Excellant',case=False,na=False),'polarity']='positive'
kk1.loc[kk1['review_header'].str.contains('Fair',case=False,na=False),'polarity']='positive'

kk=pd.concat([kk,kk1])

kk=kk.reset_index()

kk.drop(columns='index',axis=1,inplace=True)





kk.to_excel("djubo"+dt.today().strftime("%d%m%Y")+'.xlsx',index=False)