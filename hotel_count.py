import pymssql
from haversine import haversine
import pandas as pd
from tqdm import tqdm

connection = pymssql.connect(server='52.6.157.39',user='datascience',password='DS@Eh1-1',database ='reznext',port='7289')

query = 'select * from reznext.dbo.hotels where status=1;'

df =pd.read_sql(query,connection)

df['Lng'] = pd.to_numeric(df['Lng'].str.replace(",", ""), errors='coerce')

df['Lat'] = pd.to_numeric(df['Lat'].str.replace(",", ""), errors='coerce')

df['Lat']=df['Lat'].astype(float)
df['Lng']=df['Lng'].astype(float)
df['latlng']=df[['Lat','Lng']].apply(tuple,axis=1)

data=pd.read_excel('HotelInput.xlsx')

ab=df[df['HotelCode'].isin(data['HotelCode'])]

v=ab['HotelCode'].unique()


lst1=[]
z=ab
df2=df

for i in tqdm(range(0,len(v))):
    stlt=z[(z["HotelCode"]==v[i])]['latlng'].tolist()
    oo=df2[df2['Country']==z[(z["HotelCode"]==v[i])]['Country'].values[0]]
    lis=oo['latlng'].tolist()
    if len(stlt)>0:
        lst=[]
        c=0
        while c < len(lis):
            lst.append(haversine(stlt[0],lis[c],unit='mi'))
            c=c+1
    
        oo['proximity']=lst
        p=oo.sort_values(by='proximity')
        for j in range(1,20):
            p=oo[oo['proximity'] < j]
            p['HotelCode']=v[i]
            s=p.groupby('HotelCode').proximity.nunique().reset_index(name='hotel_count'+str(j))
            lst1.append(s)
aa=pd.concat(lst1)

aa=aa.pivot_table(index='HotelCode').reset_index()

       