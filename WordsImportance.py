import pandas as pd
import matplotlib.pyplot as plt
from main import df0_NB,df1_NB,df2_NB,df0_SVM,df1_SVM,df2_SVM,df0_LR,df1_LR,df2_LR,df0_DT,df1_DT,df2_DT,df0_GB,df1_GB,df2_GB




#Funktion für Anzahl anzeige im Balken
def add_value_label(x_list,y_list,list_count):
    for i in range(1, len(x_list)+1):
        plt.text(i-1,y_list[i-1]/2,list_count[i-1], ha="center")

#Funktion erstellen Plot
def make_importantWordsPlot (classifier, Anzahl, label):

    if classifier == 'NB':
        if label==0:
            df = df0_NB
        elif label==1:
            df = df1_NB
        elif label==2:
            df = df2_NB

    if classifier == 'SVM':
        if label==0:
            df = df0_SVM
        elif label==1:
            df = df1_SVM
        elif label==2:
            df = df2_SVM

    if classifier == 'LR':
        if label==0:
            df = df0_LR
        elif label==1:
            df = df1_LR
        elif label==2:
            df = df2_LR

    if classifier == 'DT':
        if label==0:
            df = df0_DT
        elif label==1:
            df = df1_DT
        elif label==2:
            df = df2_DT

    if classifier == 'GB':
        if label==0:
            df = df0_GB
        elif label==1:
            df = df1_GB
        elif label==2:
            df = df2_GB



    words = df['word'][:Anzahl].tolist()
    impact = df['mean'][:Anzahl].tolist()
    count = df['count'][:Anzahl].tolist()


    plt.bar(words,impact)
    plt.xticks(rotation=45, ha='right')
    add_value_label(words,impact,count)


    plt.xlabel("words")
    plt.ylabel("impact")
    plt.show()





