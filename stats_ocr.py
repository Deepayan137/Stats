import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

editors = ['pampari vijaya', 'vijay krishna', 'Banaka preethi', 'sandeep', 'Ram sharma', 'K.Kanaka Durga', 'vinod yadav',
           'sonygiri','morthad rajinikanth','julijha','pampari.anusha','dandu.jyothi']
language = ['English', 'Hindi', 'Sanskrit', 'Telugu']

months = ['may']
def create_csv(path):
    for month in months:
        if not os.path.exists(os.path.join(path,month)):
            os.makedirs(os.path.join(path,month))
        print (os.path.join(path, month) + '.csv')
        df = pd.read_csv(os.path.join(path,month)+'.csv')

    #df.columns = ['Editor', 'Language', 'Book','Page', 'ID', 'Time', 'Date','Total Words','Errors Detected','Accuracy']
        df.set_index(['Date'], inplace=True)

        for i in range(len(editors)):
            df.loc[df['Editor'] == editors[i]].to_csv(os.path.join(month,editors[i]) + '.csv')


create_csv('/home/deepayan/CVIT_codes/Stats/')

# df = df = pd.read_csv('s suhana.csv')
# print df.head()
def manage_stats(path):
    for month in months:
        for i in range(len(editors)):
            df = pd.read_csv(os.path.join(os.path.join(path,month),editors[i] + '.csv'))

            date_set = set(df['Date'])
            date_list = list(date_set)

            row = []
            for k in range(len(date_list)):

                page = 0.0
                total_time = 0.0
                eng = 0.0
                eng_time = 0.0
                hindi = 0.0
                hindi_time = 0.0
                sans = 0.0
                sans_time = 0.0
                tel = 0.0
                tel_time = 0.0
                df1 = df.loc[df['Date'] == date_list[k]]

                date = date_list[k]
                total_time = np.sum(np.array(df1['Time Taken (s)']))
                page = len(df1['Language'])
                total_time = total_time / (60 * page)

                for j in range(len(df1)):

                    if df1['Language'].iloc[j] == 'English':
                        eng += 1
                        eng_time += df1['Time Taken (s)'].iloc[j]
                    elif df1['Language'].iloc[j] == 'Sanskrit':
                        sans += 1
                        sans_time += df1['Time Taken (s)'].iloc[j]
                    elif df1['Language'].iloc[j] == 'Hindi':
                        hindi += 1
                        hindi_time += df1['Time Taken (s)'].iloc[j]
                    elif df1['Language'].iloc[j] == 'Telugu':
                        tel += 1
                        tel_time += df1['Time Taken (s)'].iloc[j]
                if tel != 0:
                    tel_time = tel_time / (60.0 * tel)
                if hindi != 0:
                    # print hindi_time
                    hindi_time = hindi_time / (60.0 * hindi)
                if sans != 0:
                    sans_time = sans_time / (60.0 * sans)
                if eng != 0:
                    eng_time = eng_time / (60.0 * eng)

                row.append([date, page, total_time, eng, eng_time, sans, sans_time
                               , hindi, hindi_time, tel, tel_time])
                # print row
            df2 = pd.DataFrame(row, columns=['Date', 'Pages', 'Total_Time', 'English', 'Time_Eng', 'Sanskrit', 'Time_Sans',
                                             'Hindi',
                                             'Time_Hindi', 'Telugu', 'Time_Telugu'])

            df2.to_csv(os.path.join(os.path.join(path,month),editors[i])+ '_date_wise.csv')


manage_stats('/home/deepayan/CVIT_codes/Stats/')
def gen_stats():
    df1 = pd.read_csv('may/date_wise.csv')
    #df1.columns = ['Date', 'Corrections']
    # df1.set_index(['Date'],inplace =True)

    date_set = set(df1['Date'])
    date_list = list(date_set)

    date_list = sorted(date_list, key=lambda d: map(int, d.split('-')))

    index = np.arange(len(date_list))
    bar_width = 0.35
    opacity = 0.4
    pages = df1['Pages'].tolist()
    plt.xticks(rotation=45)
    plt.bar(index, pages, bar_width,
            alpha=opacity,
            color='b',
            label='Pages')
    plt.plot(index, pages, '-o', color='m', label='Pages')
    plt.xticks(index + bar_width, date_list)
    plt.xlabel('Date')
    plt.ylabel('Corrections')
    plt.title('General stats')
    plt.legend()
    plt.savefig('/home/deepayan/CVIT_codes/Stats/general_stats.png')

    #plt.tight_layout()
    #plt.show()

#gen_stats()
def books():
    for i in range(len(editors)):
        df = pd.read_csv(editors[i] + '.csv')
        books_set = set(df['Book'])
        books_list = list(books_set)
        row = []
        for k in range(len(books_list)):
            df1 = df.loc[df['Book'] == books_list[k]]
            pages = len(df1['Book'])
            total_time = np.sum(np.array(df1['Time Taken (s)']))
            total_time = (total_time / (pages * 60.0))
            row.append([books_list[k], total_time, pages])

        df2 = pd.DataFrame(row, columns=['Book', 'Average Time', 'Pages'])

        df2.to_csv(editors[i] + '_book_wise.csv')	


#books()

def merge_books():
    row = []
    row2 = []
    df_merge = pd.DataFrame()
    for i in range(0, len(editors)):
        df = pd.read_csv(editors[i] + '_book_wise.csv')
        df.set_index('Book', inplace=True)
        row.append(df)
    df = pd.concat(row, axis=1)
    df.fillna(0, inplace=True)
    df.to_csv('all_books.csv')
    '''for j in range((len(editors)/2)+1,len(editors)):
        df = pd.read_csv(editors[i] + '_book_wise.csv')
        #df.set_index('Book', inplace=True)
        row2.append(df)
    df1 = pd.concat(row2,axis=2)


    df_merge=df.merge(df,df1,on='Book')

    #df1.set_index('Book',inplace=True)
    #df1.drop(df1.columns[[1,5,9,13,17]], axis=1,inplace=True)
    df_merge.to_csv('all_books.csv')'''


#merge_books()

def plot_stats():


    df = pd.read_csv('may/date_wise.csv')
    date_set = set(df['Date'])
    date_list = list(date_set)

    date_list = sorted(date_list, key=lambda d: map(int, d.split('-')))

    index = np.arange(len(date_list))

    bar_width = 0.15

    opacity = 0.4

    english = (df['Eng'].tolist())
    time = df.Total_Time.tolist()
    sanskrit = df['Sans'].tolist()

    hindi = df['Hin'].tolist()

    #telugu = df.Telugu.tolist()

    mean_time = np.mean(np.array(time))
    max_time = np.max(np.array(time))
    min_time = np.min(np.array(time))

    txt = 'Avearge Time: %d' % mean_time + '\n' + 'Maximum Time: %d' % max_time + '\n' + 'Minimum Time: %d' % min_time
    # fig = plt.figure()
    print txt
    #ax1 = plt.subplot2grid((1, 1), (0, 0))




    f = plt.figure(1)
    plt.xticks(rotation=45)

    f1 = plt.bar(index, english, bar_width, color='y')
    f2 = plt.bar(index, hindi, bar_width, color='b', bottom=english)
    f3 = plt.bar(index, sanskrit, bar_width, color='r', bottom=[i + j for i, j in zip(english, hindi)])
    f4 = plt.plot(index, time, '-o', color='m', label='Average Time')
    # p4 = plt.bar(index, telugu , width, color='r')
    plt.ylabel('Pages')
    plt.title('Language Wise Stats')
    plt.xticks(index - bar_width / 2., date_list)
    plt.yticks(np.arange(0, 100, 10))
    plt.legend((f1[0], f2[0], f3[0], f4[0]), ('English', 'Hindi', 'Sanskrit','Time'))
    '''ax1.bar(index + 3 * bar_width, telugu, bar_width,
            alpha=opacity,
            color='',
            label='Telugu')'''


    font_dict = {'family': 'Times New Roman',
                 'color': 'darkred',
                 'size': 12}
    plt.annotate(txt, xy=(0, 1), xycoords='axes fraction', fontsize=12,
                 horizontalalignment='left', verticalalignment='top')
    plt.savefig('/home/deepayan/CVIT_codes/Stats/bylanguage.png')



    # plt.tight_layout()
    # plt.show()


#plot_stats()
def stacked_date(path):
    for month in months:
        df = pd.read_csv(os.path.join(path,month)+'.csv')
        print df.head()
        df.columns = ['Editor', 'Language', 'Book','Page', 'ID', 'Time', 'Date','Total Words','Errors Detected','Accuracy','Image URL']
        date_set = set(df['Date'])

        date_list = list(date_set)

        new_date_list = []
        for date in date_list:

            split_date = date.split('-')
            year = int(split_date[0])
            mon = int(split_date[1])

            day = int(split_date[2])
            #if month == 4 or month == 5:
            new_date_list.append(date)
        new_date_list = sorted(new_date_list, key=lambda d: map(int, d.split('-')))

        row = []
        new_row = []


        for k in range(len(new_date_list)):
            df1 = df.loc[df['Date']== new_date_list[k]]
            #df2 = df.loc[df['Editor'] == editors[k]]
            date = new_date_list[k]
            #editors = ['s suhana', 'sonygiri', 'Heerekar Anuradha', 'nagajyothi', 'P Nagamani', 'T Mamatha', 'Dt Mounika',
              #         'snigdha vempalla']
            total_time = (np.sum(np.array(df1['Time'])))

            eng = []
            eng_time = 0.0
            hindi = []
            hindi_time = 0.0
            sans = []
            sans_time = 0.0
            sans_acc = []
            hindi_acc = []
            eng_acc = []
            telegu_acc = []
            eng_words =[]
            sans_words =[]
            hindi_words = []
            telugu_words = []
            tel = 0.0
            tel_time = 0.0
            pampri = 0.0
            pampri_time =0.0
            pampri_words=0.0
            vijay = 0.0
            vijay_time = 0.0
            vijay_words = 0.0
            banaka = 0.0
            banaka_time =0.0
            banaka_words = 0.0
            sandeep = 0.0
            sandeep_time = 0.0
            sandeep_words =0.0
            ram = 0.0
            ram_time = 0.0
            ram_words=0.0
            durga = 0.0
            durga_time = 0.0
            durga_words = 0.0
            vinod = 0.0
            vinod_time = 0.0
            vinod_words = 0.0
            juli = 0.0
            juli_time = 0.0
            juli_words=0.0
            anu = 0.0
            anu_time=0.0
            anu_words=0.0
            jyoti = 0.0
            jyoti_time = 0.0
            jyoti_words = 0.0
            rajni=0.0
            rajni_time = 0.0
            rajni_words=0.0
            editors_per_day= []

            for j in range(len(df1)):

                if df1['Editor'].iloc[j] == 'pampari vijaya':
                    pampri+= 1
                    pampri_time+= df1['Time'].iloc[j]
                    pampri_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'vijay krishna':
                    vijay += 1
                    vijay_time += df1['Time'].iloc[j]
                    vijay_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'Banaka preethi':
                    banaka += 1
                    banaka_time += df1['Time'].iloc[j]
                    banaka_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'sandeep':
                    sandeep += 1
                    sandeep_time += df1['Time'].iloc[j]
                    sandeep_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'Ram sharma':
                    ram += 1
                    ram_time += df1['Time'].iloc[j]
                    ram_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'K.Kanaka Durga':
                    durga += 1
                    durga_time += df1['Time'].iloc[j]
                    durga_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'vinod yadav':
                    vinod+= 1
                    vinod_time += df1['Time'].iloc[j]
                    vinod_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'julijha':
                    juli += 1
                    juli_time += df1['Time'].iloc[j]
                    juli_words+= df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'pampari.anusha':
                    anu += 1
                    anu_time += df1['Time'].iloc[j]
                    anu_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'dandu.jyothi':
                    jyoti += 1
                    jyoti_time= df1['Time'].iloc[j]
                    jyoti_words += df1['Total Words'].iloc[j]
                elif df1['Editor'].iloc[j] == 'morthad rajinikanth':
                    rajni+= 1
                    rajni_time +=df1['Time'].iloc[j]
                    rajni_words += df1['Total Words'].iloc[j]



                if df1['Language'].iloc[j] == 'English':
                    eng.append(1)
                    eng_words.append(df1['Total Words'].iloc[j])
                    #print date,df1['Time'].iloc[j]
                    eng_time += df1['Time'].iloc[j]

                    if df1['Accuracy'].iloc[j] != 100:
                        eng_acc.append(df1['Accuracy'].iloc[j])
                elif df1['Language'].iloc[j] == 'Sanskrit':
                    sans.append(1)
                    sans_words.append(df1['Total Words'].iloc[j])
                    sans_time += df1['Time'].iloc[j]
                    if df1['Accuracy'].iloc[j] != 100:
                        sans_acc.append(df1['Accuracy'].iloc[j])
                elif df1['Language'].iloc[j] == 'Hindi':
                    hindi.append(1)
                    hindi_words.append(df1['Total Words'].iloc[j])
                    hindi_time += df1['Time'].iloc[j]
                    if df1['Accuracy'].iloc[j] != 100:
                        hindi_acc.append(df1['Accuracy'].iloc[j])
                elif df1['Language'].iloc[j] == 'Telugu':
                    tel += 1
                    telugu_words.append(df1['Total Words'].iloc[j])
                    tel_time += df1['Time'].iloc[j]
                    if df1['Accuracy'].iloc[j] != 100:
                        telegu_acc.append(df1['Accuracy'].iloc[j])
            hindi = np.sum(np.array(hindi))
            sans = np.sum(np.array(sans))
            eng = np.sum(np.array(eng))





            if tel!= 0:
                tel_time = np.ceil(tel_time/(np.sum(np.array(telugu_words))))

            if pampri != 0:
                editors_per_day.append(1)
                pampri_time = np.ceil(pampri_time / (pampri*60))
                pampri_words = np.ceil(pampri_words / (pampri))
            if juli != 0:
                # print hindi_time
                editors_per_day.append(1)
                juli_time = np.ceil(juli_time / (juli*60))
                juli_words = np.ceil(juli_words/juli)
            if anu != 0:
                # print hindi_time
                editors_per_day.append(1)
                anu_time = np.ceil(anu_time / (anu*60))
                anu_words = np.ceil(anu_words/anu)
            if  vijay!= 0:
                editors_per_day.append(1)
                vijay_time = np.ceil(vijay_time / (vijay*60))
                vijay_words = np.ceil(vijay_words/ (vijay))
            if banaka != 0:
                editors_per_day.append(1)
                banaka_time = np.ceil(banaka_time / (banaka*60))
                banaka_words = np.ceil(banaka_words/banaka)
            if sandeep != 0:
                editors_per_day.append(1)
                sandeep_time = np.ceil(sandeep_time / (sandeep*60))
                sandeep_words = np.ceil(sandeep_words/sandeep)
            if ram != 0:
                # print hindi_time
               editors_per_day.append(1)
               ram_time = np.ceil(ram_time / (ram*60))
               ram_words =np.ceil(ram_words/ram)
            if durga != 0:
                editors_per_day.append(1)
                durga_time = np.ceil(durga_time / (durga *60))
                durga_words = np.ceil(durga_words/(durga))
            if vinod != 0:
                editors_per_day.append(1)
                vinod_time = np.ceil(vinod_time / (vinod*60))
                vinod_words = np.ceil(vinod_words/vinod)
            if jyoti != 0:
                editors_per_day.append(1)
                jyoti_time = np.ceil(jyoti_time / (jyoti*60))
                jyoti_words = np.ceil(jyoti_words/jyoti)
            if rajni != 0:
                editors_per_day.append(1)
                rajni_time = np.ceil(rajni_time / (rajni * 60))
                rajni_words = np.ceil(rajni_words/rajni)
            try:
                mean_eng_accuracy = np.mean(np.array(eng_acc))/100
                mean_eng_words = np.ceil(np.mean(np.array(eng_words)))
                mean_hin_accuracy = np.mean(np.array(hindi_acc))/100
                mean_hindi_words = np.ceil(np.mean(np.array(hindi_words)))
                mean_sanskrit_accuracy = np.mean(np.array(sans_acc))/100
                mean_sanskrit_words = np.ceil(np.mean(np.array(sans_words)))
                mean_telugu_accuracy = np.mean(np.array(telegu_acc))/100
                mean_telegu_words = np.mean(np.array(telugu_words))
                page = pampri + vijay + juli + anu + sandeep + banaka + vinod + durga + ram+jyoti+rajni
                # print page
                total_words = pampri_words + vijay_words + juli_words + anu_words +jyoti_words+ sandeep_words + banaka_words + rajni_words + vinod_words + durga_words + ram_words

                editors = np.sum(np.array(editors_per_day))
                mean_words = np.ceil(total_words/11*3)
                error_eng_words = np.ceil((1-mean_eng_accuracy)*mean_eng_words)
                error_hindi_words = np.ceil((1-mean_hin_accuracy)*mean_hindi_words)
                error_sanskrit_words= np.ceil((1-mean_sanskrit_accuracy)*mean_sanskrit_words)

                error_telugu_words = np.ceil((1-mean_telegu_words)*mean_telegu_words)
                if eng != 0:
                    eng_time = np.ceil(eng_time / (eng*error_eng_words))

                if hindi != 0:
                    hindi_time = np.ceil(hindi_time / (hindi*error_hindi_words))
                if sans != 0:
                    sans_time = np.ceil(sans_time / (sans*error_sanskrit_words))



                row.append([date, page, np.ceil(total_time/3600.0), eng,mean_eng_words, eng_time,mean_eng_accuracy,error_eng_words,
                      sans,mean_sanskrit_words,sans_time,
                               mean_sanskrit_accuracy,error_sanskrit_words, hindi,mean_hindi_words,hindi_time,mean_hin_accuracy,
                     error_hindi_words, tel,mean_telegu_words, tel_time,mean_telugu_accuracy,
                                error_telugu_words])


                new_row.append([date, editors,pampri,pampri_time,pampri_words,vijay,vijay_time,vijay_words,banaka,banaka_time,banaka_words,
                               sandeep,sandeep_time,sandeep_words,ram,ram_time,ram_words,durga,durga_time,durga_words,vinod,vinod_time,vinod_words,
                                juli,juli_time,juli_words,anu,anu_time,anu_words,rajni,rajni_time,rajni_words,jyoti,jyoti_time,jyoti_words])
            except Exception as e:
                print e
        df2 = pd.DataFrame(row,columns=['Date', 'Pages','Total_Time', 'Eng','Words','Time_Eng','Eng_Accuracy',
                                        'Eng_error_words','Sans','Words','Time_Sans','Sans_Accuracy','Sans_error_words','Hin',
                                            'Words','Time_Hin','Hin_Accuracy','Hin_error_words', 'Telugu', 'Words','Time_Telugu','Telegu_Accuracy','Tel_eror_words'])
        #print "hii"
        df2.fillna(0, inplace=True)
        df2.to_csv(os.path.join(os.path.join(path,month),'date_wise.csv'))
        df3 = pd.DataFrame(new_row,columns=['Date','no.Editors','pampari vijaya','pampri_time','pampri_words','vijay krishna',
                                            'Vijay_time','Vijay_words','Banaka preethi','Banaka_time','Banaka_words','sandeep','Sandeep_time','Sandeep_words',
                                            'Ram sharma','Ram_Sharma_time','Ram_words','K.Kanaka Durga','Durga_time','Durga_words','vinod yadav','Vinod_time','Vinod_words',
                                                'Julijha','Julijha_time','Julijah_words','Anusha','Anusha_time','Anusha_words','morthad rajinikanth','Rajni_time','Rajni_words','Jyothi',
                                            'Jyothi_time','Jyothi_words'])
        df3.fillna(0, inplace=True)
        df3.to_csv(os.path.join(os.path.join(path,month),'editor_wise.csv'))


stacked_date('/home/deepayan/CVIT_codes/Stats/')

def plot_stacked_date():
    for month in months:
        df = pd.read_csv(os.path.join(month,'date_wise.csv'))

        date_list = list(set(df['Date']))
        date_list = sorted(date_list, key=lambda d: map(int, d.split('-')))
        index = np.arange(len(date_list))
        plt.xticks(rotation=45)
        width = 0.2

        #print date_list
        english = (df['Eng'].tolist())

        sanskrit = df.Sans.tolist()
        hindi = df.Hin.tolist()
        f = plt.figure(1)
        plt.xticks(rotation=45)

        f1 = plt.bar(index, english, width, color='y')
        f2 = plt.bar(index, hindi , width, color='b',bottom=english)
        f3 = plt.bar(index, sanskrit , width, color='r',bottom=[i+j for i,j in zip(english,hindi)])
        #p4 = plt.bar(index, telugu , width, color='r')
        plt.ylabel('Pages')
        plt.title('Language Wise Stats')
        plt.xticks(index-width/2. , date_list)
        plt.yticks(np.arange(0, 200, 10))
        plt.legend((f1[0], f2[0], f3[0]), ('English', 'Hindi', 'Sanskrit'))
        plt.savefig('/home/deepayan/CVIT_codes/Stats/'+month+'_Language.png')
        #f.show()

        '''p = plt.figure(2)
        df1 = pd.read_csv(os.path.join(month,'editor_wise.csv'))
        df1['pampari vijaya']=1
        #p.use_style('ggplot')
        width = 0.35

        opacity = 0.4
        plt.xticks(rotation=45)
        #print df['Suhana']
        p1 = plt.bar(index, df1['pampari vijaya'], width, color='y')
        p2 = plt.bar(index, df1['vijay krishna'], width, color='g',bottom=df['pampari vijaya'])
        p3 = plt.bar(index, df1['Banaka preethi'], width, color='m',bottom=[i+j for i,j in zip(df['pampari vijaya'],df['vijay krishna'])])
        #p4 = plt.bar(index, df['NagaJyothi'], width, color='k',bottom=df['Anuradha'])
        p4 = plt.bar(index, df1['sandeep'], width, color='b',bottom=[i+j+k for i,j,k in zip(df['pampari vijaya'],df['vijay krishna'],df['Banaka preethi'])])
        p5 = plt.bar(index, df1['Ram sharma'], width, color='c',bottom=[i+j+k+l for i,j,k,l in zip(df['pampari vijaya'],df['vijay krishna'],df['Banaka preethi'],df['sandeep'])])
        p6 = plt.bar(index, df1['K.Kanaka Durga'], width, color='k',bottom=[i+j+k+l+m for i,j,k,l,m in zip(df['pampari vijaya'],df['vijay krishna'],df['Banaka preethi'],df['sandeep'],df['Ram sharma'])])
        #p8 = plt.bar(index, df['Snigdha'], width, color='w',bottom=df['Mounika'])
        p7 = plt.bar(index, df1['vinod yadav'], width, color='darkgreen',bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(df['pampari vijaya'],df['vijay krishna'],df['Banaka preethi'] ,df['sandeep'],df['Ram sharma'],df['K.Kanaka Durga'])])

        p8 = plt.bar(index, df1['sonygiri'], width, color='y', bottom=[i + j + k + l + m + n + o for i, j, k, l, m,n,o in zip(df['pampari vijaya'], df['vijay krishna'], df['Banaka preethi']
                                                                             ,df['sandeep'], df['Ram sharma'],
                                                                             df['K.Kanaka Durga'],df['vinod yadav'])])

        p9 = plt.bar(index, df['morthad rajinikanth'], width, color='seagreen', bottom=[i + j + k + l + m+n+o+p for i, j, k, l, m,n,o,p in
                                                                       zip(df['pampari vijaya'], df['vijay krishna'],
                                                                           df['Banaka preethi']
                                                                           , df['sandeep'], df['Ram sharma'],
                                                                           df['K.Kanaka Durga'], df['vinod yadav'],df['sonygiri'])])

        plt.ylabel('Pages')
        plt.title('Editor wise Stats')
        plt.xticks(index - width / 2., date_list)
        plt.yticks(np.arange(0, 200, 10))
        plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0],p6[0],p7[0],p8[0],p9[0]), ('pampari vijaya', 'vijay krishna', 'Banaka preethi', 'sandeep', 'Ram sharma', 'K.Kanaka Durga', 'vinod yadav','Soni Giri','rajnikanth'))
        plt.savefig('/home/deepayan/CVIT_codes/Stats/'+month+'_Editors.png')
        #p.show()
        raw_input()'''
plot_stacked_date()

editors = ['pampari vijaya', 'vijay krishna', 'Banaka preethi', 'sandeep', 'Ram sharma', 'K.Kanaka Durga', 'vinod yadav',
               'sonygiri','morthad rajinikanth','dandu.jyothi']
