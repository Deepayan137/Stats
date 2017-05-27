import os
import sys
import pandas as pd
import numpy as np
months = ['Jan','Feb','Mar','Apr']
languages = ['Eng','Hin','Sans']
def cost(path):
    for month in months:
        if os.path.exists(os.path.join(os.path.join(path, month),'cost_details.txt')):
            os.remove(os.path.join(os.path.join(path,month),'cost_details.txt'))
        cost_text = open(os.path.join(os.path.join(path,month),'cost_details.txt'),'a')
        df = pd.read_csv(os.path.join(os.path.join(path,month),'date_wise.csv'))
        df.fillna(0, inplace=True)
        for language in languages:
            error_words =[]
            accuracy=[]
            time =[]
            Days = len(df[language])
            Pages = np.sum(np.array(df[language]))
            for i in range(Days):
                if df[language+'_error_words'].iloc[i] != 0.0:
                    error_words.append(df[language+'_error_words'].iloc[i])


            #print df[language+'_error_words'].tolist()4
                if df[language + '_Accuracy'].iloc[i] != 0.0:
                    accuracy.append(df[language + '_Accuracy'].iloc[i]*100)
                    #print accuracy

                if df['Time_'+language].iloc[i] != 0.0:
                    time.append(df['Time_'+language].loc[i])


            #cost_per_word = 3000 / (Pages * mean_error_words)
            mean_error_words = np.mean(error_words)
            mean_accuracy = np.mean(accuracy)
            mean_time = np.mean(time)
            text = 'Days worked:'+str(Days)+'\n'+language+'\n'+'Pages:'+str(Pages)+'\n'+'Mean_words:'+str(np.ceil(mean_error_words))+'\n'+\
                   'Mean_Accuracy:'+str(np.ceil(mean_accuracy))+'\n'+'Mean_Time:'+str(np.ceil(mean_time))+'\n'

            cost_text.write(text)
cost('/home/deepayan/CVIT_codes/Stats/')