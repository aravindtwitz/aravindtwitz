import time
import re
import pymssql
import pandas as pd


class OutputTableNotFoundError(Exception):
    pass

class EmptyDataFrameError(Exception):
    pass

class EmptyDataError(Exception):
    pass



def UploadTable_toDB(df=None, connection=None, Output_Table=None, daily_refresh_table=False, chuck_size=1000):
    
    start_time = time.time()
    
    if isinstance(df, pd.DataFrame):
        if df.empty:
            raise EmptyDataFrameError("The given dataFrame is Empty.")
    elif df==None:
        raise EmptyDataError("No dataframe is provided to upload")
    
    if Output_Table==None:
        raise OutputTableNotFoundError("Pass the target table to which the dataframe needs to be uploaded as an argument (Output_Table) to the 'UploadTable' function.")

    
    # Default DB User : DataScience.
    if connection==None:
        connection=connection1
        


    if daily_refresh_table:
        cursor.execute(f"truncate table {Output_Table};")
        connection.commit()
        print('- Table truncated...')
        time.sleep(10)


    print('- uploading dataframe...')

    no_of_chunk = df.shape[0]//chuck_size 

    columns = [i if len(i.split())==1 else f"[{i}]" for i in df.columns.tolist()]
    #print(columns)
    val_list = df.values.tolist()

    sampl_count = 0

    def insert(query):
        nonlocal sampl_count
        cursor.execute(query)
        connection.commit()
        sampl_count+=1

    insertty = []
    k = 0
    counter=1
    

    try:
        for value in val_list:
            k+=1
            temp = tuple(value)
            #print(temp)
            insertty.append(temp)
            if k==1000:
                insertty = re.sub(r"u'", "'", re.sub(r"\[|\]", r"", str(insertty)))
                #print(insertty)
                #insertty = re.sub('"', "'", insertty)
                insertQ = f"insert into {Output_Table} ({', '.join(columns)}) values {str(insertty)}"
                #print(insertQ)
                insertQ = re.sub(r"\snan([,|\)])", r" null\1", insertQ)
                #print(insertQ)
                insert(insertQ)
                k = 0
                insertty = []
                if not counter==no_of_chunk:
                    counter+=1

        insertty = re.sub(r"u'", "'", re.sub(r"\[|\]", r"", str(insertty)))
        #insertty = re.sub('"', "'", insertty)
        insertQ = f"insert into {Output_Table} ({', '.join(columns)}) values {str(insertty)}"
        #print(insertQ)
        insertQ = re.sub(r"\snan([,|\)])", r" null\1", insertQ)
        insert(insertQ)
        connection.close()
        
        print("- successfully updated table with dataframe.")
        print(f'\n  No. of rows    : {df.shape[0]}\n  No. of columns : {df.shape[1]}')
        end_time = time.time() - start_time
        print("\n  time:",round(end_time,2), 's')
        return 'Process Completed.'
    
    except Exception as e:
        print("An error occured while uploading the table to DB.\n")
        return e
    
    
    
connection = pymssql.connect(server='44.228.162.177',user='tool',password='DR2sT&',database ='EDDIEXTRACTION',port='3630')


cursor = connection.cursor()

UploadTable_toDB(akk, connection, 'set15_category_prediction_correct_test', True)
    
    