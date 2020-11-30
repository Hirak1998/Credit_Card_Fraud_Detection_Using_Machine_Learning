from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from IPython import display
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.metrics import mean_squared_error, r2_score
import sys

sys.setrecursionlimit(10**9)


main = tkinter.Tk()
main.title("Credit Card Fraud Detection") #designing main screen
main.geometry("1300x800")

global filename
global cls
global X, Y, X_train, X_test, y_train, y_test
global random_acc # all global variables names define in above lines
global clean
global attack
global total


def traintest(train):     #method to generate test and train data from dataset
    X = train.values[:, 0:29] 
    Y = train.values[:, 30]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all five features data
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    try:
        train = pd.read_csv(filename)
    except NameError:
        messagebox.showwarning("warning","Please Upload the file first") 
    X, Y, X_train, X_test, y_train, y_test = traintest(train)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")



def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");



def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(50):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy



def random():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global random_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    try:
        cls = RandomForestClassifier(n_estimators=50,max_depth=2,random_state=0,class_weight='balanced')
        cls.fit(X_train, y_train) 
        prediction_data = prediction(X_test, cls) 
        
    except:
        messagebox.showwarning("warning","Please Train and Test the model")
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Accuracy')
    r2sco=r2_score(y_test, prediction_data)
    text.insert(END,"Coeffiecent of determination : "+str(r2sco)+"\n\n")
    meansqua=mean_squared_error(y_test,prediction_data)
    text.insert(END,"Mean squared Error : "+str(meansqua)+"\n\n")
    text.insert(END,"\n\nPrediction of Fraud And Clean Transactions\n\n")
    clean = 0;
    attack = 0;
    try:
        test = pd.read_csv(filename)
        test = test.values[:, 0:29]
        total = len(test)
    except NameError:
        messagebox.showwarning("warning","Please Upload the file first")
    y_pred = cls.predict(test)
    for i in range(len(test)):
        if str(y_pred[i]) == '1.0':
            attack = attack + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Contains Fraud Transaction Signature')+"\n\n")
        else:
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Transaction Contains Cleaned Signatures')+"\n\n")
    hei = [total,clean,attack]
    bars = ('Total Transactions','Normal Transaction','Fraud Transaction')
    plot2=plt.figure("Bar Graph")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, hei)
    plt.xticks(y_pos, bars)
    plt.show()

def GNB():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global random_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    try:
        cls = GaussianNB()
        cls.fit(X_train, y_train) 
        prediction_data = prediction(X_test, cls) 
        random_acc = cal_accuracy(y_test, prediction_data,"Gaussian NB accuracy")
    except:
        messagebox.showwarning("warning","Please Train and Test the model")
    r2sco=r2_score(y_test, prediction_data)
    text.insert(END,"Coeffiecent of determination : "+str(r2sco)+"\n\n")
    meansqua=mean_squared_error(y_test,prediction_data)
    text.insert(END,"Mean squared Error : "+str(meansqua)+"\n\n")
    text.insert(END,"\n\nPrediction of Fraud And Clean Transactions\n\n")
    clean = 0;
    attack = 0;
    try:
        test = pd.read_csv(filename)
        test = test.values[:, 0:29]
        total = len(test)
        text.insert(END,filename+" test file loaded\n");
    except NameError:
        messagebox.showwarning("warning","Please Upload the file first")
    y_pred = cls.predict(test)
    for i in range(len(test)):
        if str(y_pred[i]) == '1.0':
            attack = attack + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Contains Fraud Transaction Signature')+"\n\n")
        else:
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Transaction Contains Cleaned Signatures')+"\n\n")
    hei = [total,clean,attack]
    bars = ('Total Transactions','Normal Transaction','Fraud Transaction')
    plot2=plt.figure("Bar Graph")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, hei)
    plt.xticks(y_pos, bars)
    plt.show()


def SupportVector():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global random_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    try:
        cls = SVC(kernel='linear')
        cls.fit(X_train, y_train) 
        text.insert(END,"\n\nPrediction Results\n\n") 
        prediction_data = prediction(X_test, cls) 
        random_acc = cal_accuracy(y_test, prediction_data,"Support Vector Classifier")
    except:
        messagebox.showwarning("warning","Please Train and Test the model")
    r2sco=r2_score(y_test, prediction_data)
    text.insert(END,"Coeffiecent of determination : "+str(r2sco)+"\n\n")
    meansqua=mean_squared_error(y_test,prediction_data)
    text.insert(END,"Mean squared Error : "+str(meansqua)+"\n\n")
    text.insert(END,"\n\nPrediction of Fraud And Clean Transactions\n\n")
    clean = 0;
    attack = 0;
    try:
        test = pd.read_csv(filename)
        test = test.values[:, 0:29]
        total = len(test)
    except NameError:
        messagebox.showwarning("warning","Please Upload the file first")
    y_pred = cls.predict(test)
    for i in range(len(test)):
        if str(y_pred[i]) == '1.0':
            attack = attack + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Contains Fraud Transaction Signature')+"\n\n")
        else:
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Transaction Contains Cleaned Signatures')+"\n\n")
    hei = [total,clean,attack]
    bars = ('Total Transactions','Normal Transaction','Fraud Transaction')
    plot2=plt.figure("Bar Graph")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, hei)
    plt.xticks(y_pos, bars)
    plt.show()
    
                


font = ('times', 16, 'bold')
title = Label(main, text='Credit Card Fraud Detection Using Machine Larning Algorithms')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Credit Card Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Generate Train & Test Model", command=generateModel)
modelButton.place(x=350,y=550)
modelButton.config(font=font1) 

runrandomButton = Button(main, text="Run Support Vector classifier", command=SupportVector)
runrandomButton.place(x=650,y=550)
runrandomButton.config(font=font1)

runrandomButton = Button(main, text="Run Gaussian Navie Bayes", command=GNB)
runrandomButton.place(x=50,y=600)
runrandomButton.config(font=font1)

runrandomButton = Button(main, text="Run Random Forest classifier", command=random)
runrandomButton.place(x=350,y=600)
runrandomButton.config(font=font1) 


exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=650,y=600)
exitButton.config(font=font1) 

main.config(bg='LightSkyBlue')
main.mainloop()
