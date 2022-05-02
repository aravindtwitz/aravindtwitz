#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
from statistics import mean

"""We import a specific library for sentence distance estimation"""
from textdistance import levenshtein
from textdistance import ratcliff_obershelp

"To save and load the pretrained model"
import pickle


filename="Venue_Maper_SVC_model.sav"
std_svc=pickle.load(open(filename, 'rb'))



"""
The following functions are necesary to create the features derived from the dataset
They are not implemented as transformers/predictors, but could be done in the future, although it is not necessary if 
we respect column names
"""

#Tis function will allow us to split those words that are joined together so we will recognise them as a sentence instead as a single word
def break_down_joined_names (s):
    try:
        if len(s.split(" "))<=1:
            st = ''
            for c in s:
                if c == c.upper():
                    st += ' '   
                st += c    
            return st.strip()
        else:
            return s
    except:
        return s

def Levensthein_dist_NA_handling(source, google):
    #This function returns a normalized levensthein distance from the whole imput
    #range(0,1)
    #DISTANCES CLOSE TO 0=VERY SIMILAR
    source=break_down_joined_names(source)
    google=break_down_joined_names(google)
    try: 
        return(levenshtein.distance(str(source.lower()),str(google.lower()))/max([len(str(source)),len(str(google))]))
    except:
        return(np.nan)


def Upper_Lev_dist(source, google):
    #This function returns a normalized levensthein distance of the initial letters of each word, that is we are just comparing
    #initial letters, so abbreviations do not influence distance
    #i.e. "The Kitchen of Peter"->"TKP"
    #range(0,2)
    #DISTANCES=0 mean the capital letters of the strings are the same
    
    capital_google=''.join([c for c in str(google) if c.isupper()])
    capital_source=''.join([c for c in str(source) if c.isupper()])
    try: 
        return(levenshtein.distance(capital_source,capital_google)/max([len(capital_source),len(capital_google)]))
    except:
        try:
            capital_google=''.join([c[0] for c in google.split(" ")])
            capital_source=''.join([c[0] for c in source.split(" ")])
            return(levenshtein.distance(capital_source,capital_google)/max([len(capital_source),len(capital_google)]))
        except:
            return(0)
        
def First_letter_coincidence(source, google):
    #This function returns a normalized levensthein distance of the initial letters of each word, only takes into account +4 letter words
    #initial letters, so abbreviations do not influence distance
    #i.e. "The Kitchen of Peter"->"TKP"
    #range(0,2)
    #DISTANCES=0 mean the capital letters of the strings are the same
    
    source=break_down_joined_names(source)
    google=break_down_joined_names(google)
    
    try: 
        source=sentence_to_list(source)
        google=sentence_to_list(google)
        source=" ".join([word for word in source if len(word)>4])
        google=" ".join([word for word in google if len(word)>4])
        source=set([c[0] for c in source.split(" ")])
        google=set([c[0] for c in google.split(" ")])
        return len(source & google)
    except:
        return(0)
        
def Levensthein_num_dist(source, google):
    #This function returns a normalized levensthein distance of numerical characters. Distance is normalized with average number
    #of numerical characters. i.e. "The Party 233 & 34"->"23334"
    #range(0,1), if there are no numbers output will be 0
    #DISTANCES=0 mean the capital letters of the strings are the same
    numeric_google=''.join([c for c in str(google) if c.isnumeric()])
    numeric_source=''.join([c for c in str(source) if c.isnumeric()])
    try: 
        return(levenshtein.distance(numeric_source,numeric_google)/max([len(numeric_source),len(numeric_google)]))
    except:
        return(np.nan)

def sentence_to_list(sentence):
    
    sentence=break_down_joined_names(sentence) #Tis line handles those inputs where the whole sentence was joined together
    sentence=''.join([c for c in str(sentence) if (c.isalpha() or c.isspace())])
    sentence=sentence.lower()
    sentence = sentence.split(" ")
    #returns list with "" elements removed
    return(list(filter(None,sentence)))
    
def Common_words(source, google):
    #This function returns the number of common words found in two strings and normalizes them with the length of the shortest string
    source=sentence_to_list(source)
    google=sentence_to_list(google)
    return((len(list(set(source)&set(google)))+1)/(min([len(source),len(google)])+1))

def Common_words_larger_than_4(source, google):
    #This function returns the number of common words found in two strings and normalizes them with the length of the shortest string
    try:
        source=sentence_to_list(source)
        google=sentence_to_list(google)
        source=" ".join([word for word in source if len(word)>4])
        google=" ".join([word for word in google if len(word)>4])
        source=sentence_to_list(source)
        google=sentence_to_list(google)
        return len(set(source)&set(google))
    except:
        return (0)
    
def Levensthein_num_address(source, google):
    #In this case we look for repeated numbers in both strings and perform the intersection to get what ratio of number match 
    #range(0,1), if there are no numbers output will be 0
    #The closer the output to 1, the higher the number coincidence ratio is
    try: 
        numeric_google=re.findall(r'\d+', google)
        numeric_source=re.findall(r'\d+', source)
        return(len(set(numeric_google).intersection(set(numeric_source)))/max([len(numeric_source),len(numeric_google)]))
    except:
        return(np.nan)

    
def split_dataset(dataset):
    # split into input (X) and output (y) variables
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:,-1]
    return X, y



"""
Now we load the data, it can be integrated with the SQL database if we use a python library enabling interaction with SQL server
and write the query to retrieve the data in question, for now I import it directly from a file
"""

fname=""
df = pd.read_csv(fname, sep=";", index_col=False, low_memory=False, encoding_errors="ignore")

columns=["Google  Name","Google city","Google  address","Google  Country","Google state","Google  Zip","Google  Latitude","Google  longitude"]

try:
    df.loc[df["Google  Name"]=="0","Google  Name"]=np.nan
    df.loc[df["Google city"]=="0","Google city"]=np.nan
    df.loc[df["Google  address"]=="0","Google  address"]=np.nan
    df.loc[df["Google  Country"]=="0","Google  Country"]=np.nan
    df.loc[df["Google state"]=="0","Google state"]=np.nan
    df.loc[df["Google  Zip"]=="0","Google  Zip"]=np.nan
    df.loc[df["Google  Latitude"]=="0","Google  Latitude"]=np.nan
    df.loc[df["Google  longitude"]=="0","Google  longitude"]=np.nan
except:
    for colname in columns:
        if colname not in df.columns:
            print("{} column is missing, there should be an input column named {}".format(colname,colname))


"""
At this point we have to create the features from the functions defined above, we will add these columns to the dataframe
"""
input_df=df.copy()
#Venue metrics
input_df["LevD_Name"]=df.apply(lambda x: Levensthein_dist_NA_handling(x["Input Location"], x["Google  Name"]), axis=1).to_frame()
input_df["Rat_Ob_Upper_Name"]=df.apply(lambda x: Upper_Lev_dist(x["Input Location"], x["Google  Name"]), axis=1).to_frame()
input_df["LevD_num_Name"]=df.apply(lambda x: Levensthein_num_dist(x["Input Location"], x["Google  Name"]), axis=1).to_frame()
input_df["LevD_num_set_Name"]=df.apply(lambda x: Levensthein_num_address(x["Input Location"], x["Google  Name"]), axis=1).to_frame()
input_df["Common_Words_Name"]=df.apply(lambda x: Common_words(x["Input Location"], x["Google  Name"]), axis=1).to_frame()
input_df["Common_first_letters_Name"]=df.apply(lambda x: First_letter_coincidence(x["Input Location"], x["Google  Name"]), axis=1).to_frame()
input_df["Common_words_Name_n"]=df.apply(lambda x: Common_words_larger_than_4(x["Input Location"], x["Google  Name"]), axis=1).to_frame()

#Now address metrics
input_df["LevD_Address"]=df.apply(lambda x: Levensthein_dist_NA_handling(x["Input Locationaddress"], x["Google  address"]), axis=1).to_frame()
input_df["Rat_Ob_Upper_Address"]=df.apply(lambda x: Upper_Lev_dist(x["Input Locationaddress"], x["Google  address"]), axis=1).to_frame()
input_df["LevD_num_set_Address"]=df.apply(lambda x: Levensthein_num_address(x["Input Locationaddress"], x["Google  address"]), axis=1).to_frame()
input_df["Common_Words_Address"]=df.apply(lambda x: Common_words(x["Input Locationaddress"], x["Google  address"]), axis=1).to_frame()
input_df["Common_first_letters_Address"]=df.apply(lambda x: First_letter_coincidence(x["Input Locationaddress"], x["Google  address"]), axis=1).to_frame()
input_df["Common_words_Address_n"]=df.apply(lambda x: Common_words_larger_than_4(x["Input Locationaddress"], x["Google  address"]), axis=1).to_frame()

#Now the city
input_df["LevD_City"]=df.apply(lambda x: Levensthein_dist_NA_handling(x["Input Locationcity"], x["Google city"]), axis=1).to_frame()


#This is the list of the newly created columns
new_columns=["LevD_Name","Rat_Ob_Upper_Name","LevD_num_Name","LevD_num_set_Name","Common_Words_Name","Common_first_letters_Name",
            "Common_words_Name_n","LevD_Address","Rat_Ob_Upper_Address","LevD_num_set_Address","Common_Words_Address",
             "Common_first_letters_Address","Common_words_Address_n","LevD_City"]


input_df["Mapping_status"]=std_svc.predict(input_df[new_columns])
input_df[["P1","P2"]]=std_svc.predict_proba(input_df[new_columns])



#As a brief note, note that Maping_status=0 is the equivalent of unmaped and Maping_status=1 correspond to maped events



#This block of code is meant to be used for later implementation, for now, if you want to see feature values and Probabilities (P1 & P2)
#Please use next block of code
"""
maped_venues=df.loc[input_df[(input_df["P2"]>0.5) & (input_df["Input Locationcountry"].str.lower()==input_df["Google  Country"].str.lower()) & (input_df["Input Locationcountry"].notna()) & (input_df["Input Locationaddress"].notna()) & (input_df["Google  address"].notna())].index].copy()
unmaped_venues=df.loc[input_df[((input_df["P1"]>0.7) & (input_df["Input Locationaddress"].notna() & input_df["Google  address"].notna()))|((input_df["Input Locationcountry"].str.lower()!=input_df["Google  Country"].str.lower())& input_df["Input Locationcountry"].notna())].index].copy()

#Now in the manual analyses data we can also find some events for which we are missing data such as google address, input address, so the 
#algorithm predicts them as unmaped because there is not enough information to consider them maped
manual_analyses2=df.loc[input_df[(input_df["Input Locationaddress"].isna() | input_df["Google  address"].isna() | input_df["Input Locationcountry"].isna())].index].copy()
manual_analyses=df.loc[input_df[((input_df["P1"]<0.8) & (input_df["P2"]<0.5))& (input_df["Input Locationaddress"].notna() & input_df["Google  address"].notna() & input_df["Input Locationcountry"].notna())].index].copy()
"""


maped_venues=input_df[(input_df["P2"]>0.5) & (input_df["Input Locationcountry"].str.lower()==input_df["Google  Country"].str.lower()) & (input_df["Input Locationcountry"].notna()) & (input_df["Input Locationaddress"].notna()) & (input_df["Google  address"].notna())].copy()
unmaped_venues=input_df[((input_df["P1"]>0.7) & (input_df["Input Locationaddress"].notna() & input_df["Google  address"].notna()))|((input_df["Input Locationcountry"].str.lower()!=input_df["Google  Country"].str.lower())& input_df["Input Locationcountry"].notna())].copy()
#Now in the manual analyses data we can also find some events for which we are missing data such as google address, input address, so the 
#algorithm predicts them as unmaped because there is not enough information to consider them maped
manual_analyses2=input_df[(input_df["Input Locationaddress"].isna() | input_df["Google  address"].isna() | input_df["Input Locationcountry"].isna())].copy()
manual_analyses=input_df[((input_df["P1"]<0.8) & (input_df["P2"]<0.5))& (input_df["Input Locationaddress"].notna() & input_df["Google  address"].notna() & input_df["Input Locationcountry"].notna())].copy()

