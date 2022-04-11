import pandas as pd
import re,datetime
from nltk.tokenize import word_tokenize
import spacy
from spacy.matcher import Matcher
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
from sklearn.tree import  DecisionTreeClassifier
from wordphrase import matcher
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pymssql, sys, linecache
from support import supply

script, Status, Start_id, End_id, Inputtable =sys.argv

vader = SentimentIntensityAnalyzer()

db, cursor  = supply().mssql(1, 'djuboreview')
dtcollected   = datetime.datetime.today().strftime('%Y-%m-%d')
to_mail         = "girija@aggregateintelligence.in"
cc              = ['maruthupandian@aggregateintelligence.com','saravanakumarb@aggregateintelligence.in','pythonic@aggregateintelligence.in','perumaal@aggregateintelligence.in','rmhotels@aggregateintelligence.in']
message_subject = 'Sentiment_Status_Update_'+str(dtcollected)

query1 = "select * from "+str(Inputtable)+" with (nolock) where  status = '%s' and id between '%s' and '%s';" %(Status, Start_id, End_id)

review=pd.read_sql(query1,db)
query2 = 'select * from djuboreview.dbo.Combined with (nolock);'
Corpus=pd.read_sql(query2,db)

# training_set
Train_X = Corpus['text_final']
Train_Y = Corpus['class']
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Tfidf_vect = TfidfVectorizer(max_features=2500)
Tfidf_vect.fit(Corpus['text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)

for i in range(len(review)):
    id_up = review['id'][i]
    try:
        ijk=review['review_content'][i]
        ijk=str(ijk)
        if ijk=='':
            asd=[]
            sentiment_score=[]
            aaab=[]
            polarity=[]
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
                        if word not in stopwords.words('english') and word.isalpha():
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
                fdata.loc[fdata['pred_DT']==1,'class']='cleanliness'
                fdata.loc[fdata['pred_DT']==2,'class']='elevator'
                fdata.loc[fdata['pred_DT']==3,'class']="facilities & surroundings"
                fdata.loc[fdata['pred_DT']==4,'class']="fnb & bar"
                fdata.loc[fdata['pred_DT']==5,'class']="fnb & food"
                fdata.loc[fdata['pred_DT']==6,'class']="fnb & restaurant"
                fdata.loc[fdata['pred_DT']==7,'class']="internet"
                fdata.loc[fdata['pred_DT']==8,'class']="location"
                fdata.loc[fdata['pred_DT']==9,'class']="tranquility(noise-free)"
                fdata.loc[fdata['pred_DT']==10,'class']="parking"
                fdata.loc[fdata['pred_DT']==11,'class']="room"
                fdata.loc[fdata['pred_DT']==12,'class']="staff"
                fdata.loc[fdata['pred_DT']==13,'class']="bed"  
                fdata.loc[fdata['pred_DT']==14,'class']="value for money"
                fdata.loc[fdata['pred_DT']==15,'class']="z"
            
                fdata.drop(['start', 'end','content','pred_DT'], axis=1,inplace=True)
                fdata=fdata[fdata['class']!='z']
#                 fdata=fdata[fdata['class']!='unclassified']
                fdata.reset_index(inplace = True, drop = True)
            
                fdata['polarity']=0.0
                for i in range(0,len(fdata)):
                    fdata['polarity'][i]=vader.polarity_scores(fdata['text'][i])['compound']
                
                if len(fdata) > 0:
#                     fdata.loc[fdata['polarity']<0,'sentiment']="negative"
#                     fdata.loc[fdata['polarity']>0,'sentiment']="positive"
#                     fdata.loc[(fdata['polarity']== 0),'sentiment']="neutral"
#                     kk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class']).polarity.mean().reset_index()
#                     aaab=pd.DataFrame(kk.groupby(['class']).polarity.mean()).to_json(orient='index')
#                     kkk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class','sentiment']).polarity.mean().reset_index()
#                     asd=kkk.rename(columns={"class": "properties", "sentiment": "polarity",'polarity':'strength'}).to_dict('records') 
#                     sentiment_score=vader.polarity_scores(ijk)['compound']
                    
                    fdata.loc[fdata['polarity']< -0.25,'sentiment']="negative"
                    fdata.loc[fdata['polarity']> 0.25,'sentiment']="positive"
                    fdata.loc[(fdata['polarity']<=0.25) & (fdata['polarity']>= -0.25),'sentiment']="neutral"
                    sentiment_score=vader.polarity_scores(ijk)['compound']
    
                    fdata['polarity']=round(fdata['polarity'], 2)
                    kk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class']).polarity.mean().reset_index()
                    aaab=pd.DataFrame(kk.groupby(['class']).polarity.mean()).to_json(orient='index')
                    kkk=fdata[['sentiment','class','polarity']].drop_duplicates().groupby(['class','sentiment']).polarity.mean().reset_index()
                    asd=kkk.rename(columns={"class": "properties", "sentiment": "polarity",'polarity':'strength'}).to_dict('records')
                    
                    if sentiment_score<-0.25:
                        polarity='negative'
                    elif sentiment_score >0.25:
                        polarity='positive'
                    else:
                        polarity='neutral'
                else:
                    print('fdata:',fdata)
                    asd=""
                    sentiment_score=[]
                    aaab=""
                    polarity=""
            else:
                asd=""
                sentiment_score=[]
                aaab=""
                polarity=""
        
        properties_column=re.sub(r"'","''",str(aaab))
        strength_column=re.sub(r"'","''",str(asd))
        sentiment_score_column=re.sub(r"'","''",str(sentiment_score))
        polarity_column=re.sub(r"'","''",str(polarity))
        
        if '[]' in str(sentiment_score_column):
            sentiment_score_column = 0
        
        if sentiment_score_column==0:
            polarity_column = 'neutral'
        
#         print('properties_column:',properties_column)
#         print('strength_column:',strength_column)
#         print('sentiment_score_column:',sentiment_score_column)
#         print('polarity_column:',polarity_column)
#         print(("update "+str(Inputtable)+" set properties = '%s',strength = '%s', sentiment_score = '%s', polarity = '%s' where id=%s"%(properties_column,strength_column,sentiment_score_column,polarity_column,id_up)))
        
        cursor.execute("update "+str(Inputtable)+" set properties = '%s',strength_new = '%s', sentiment_score = '%s', polarity = '%s', status=1 where id=%s"%(properties_column,strength_column,sentiment_score_column,polarity_column,id_up))
        db.commit()
        print('update')
    except Exception as e:
        frame   = sys.exc_info()[2].tb_frame
        lineno  = sys.exc_info()[2].tb_lineno
        filename= frame.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, frame.f_globals)
        print ('EXCEPTION_LINE_NO-{}\n{}\n{}\n{}'.format(lineno, line.strip(), sys.exc_info()[1], filename))
        cursor.execute("update "+str(Inputtable)+" set status=2 where id=%s"%(id_up))
        db.commit()
        print('update2')

cursor.execute("select count(1) from djuboreview.dbo.reviewdetails_bak where dtcollected=CAST(GETDATE() as date) and isnull(review_content,'')!='' and sentiment_Score is null  and status = 0")
result = cursor.fetchall()
print(result[0][0])
if result[0][0] ==0:
    try:

        message_text    = "Hi Team ,\n\nSentiment updated successfully."
        supply().mailsend(to_mail, cc, message_subject, message_text)  
    except:
        supply.eHandling()
        supply().mailsend(to_mail, cc, message_subject, f'Hi Team,\n\nPending count {result[0][0]} .')
