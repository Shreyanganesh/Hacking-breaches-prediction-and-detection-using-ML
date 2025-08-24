from django.shortcuts import render

from django.http import HttpResponseRedirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from .models import Cyber_Breaches

# Create your views here.
def index(request):
    return render(request,"index.html")

def register(request):
    if request.method=="POST":
        first=request.POST['fname']
        last=request.POST['lname']
        uname=request.POST['uname']
        em=request.POST['email']
        ps=request.POST['psw']
        ps1=request.POST['psw1']
        if ps==ps1:
            if User.objects.filter(username=uname).exists():
                messages.info(request,"Username Exists")
                return render(request,"register.html")
            elif User.objects.filter(email=em).exists():
                messages.info(request,"Email exists")
                return render(request,"register.html")
            else:
                user=User.objects.create_user(first_name=first,
            last_name=last,username=uname,email=em,password=ps)
                user.save()
                return HttpResponseRedirect("login")
        else:
            messages.info(request,"Password not Matching")
            return render(request,"register.html")

    return render(request,"register.html")

def login(request):
    if request.method=="POST":
        uname=request.POST['uname']
        ps=request.POST['psw']
        user=auth.authenticate(username=uname,password=ps)
        if user is not None:
            auth.login(request,user)
            return HttpResponseRedirect('data')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"login.html")
    return render(request,"login.html")

def adminlogin(request):
    if request.method=="POST":
        un=request.POST['uname']
        ps=request.POST['psw']
        user=auth.authenticate(username=un,password=ps)
        if user.is_superuser is not None:
            auth.login(request,user)
            return HttpResponseRedirect('adminhome')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"adminlogin.html")
    return render(request,"adminlogin.html")

def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/')

def data(request):
    if request.method=="POST":
        organisation=request.POST['org']
        records=int(request.POST['records'])
        info=request.POST['info']
        url=request.POST['url']
        lat=float(request.POST['lat'])
        long=float(request.POST['long'])
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        import numpy as np 
        df=pd.read_csv(r"static/dataset/CyberBreach.csv")
        print(df.head())
       
        print(df.isnull().sum())
        l=LabelEncoder()
        org=l.fit_transform(df["Type of organization"])
        information=l.fit_transform(df["Information Source"])
        organisation_1=l.fit_transform([organisation])
        info_1=l.fit_transform([info])
        df["Organisation"]=org
        df["Information_Source"]=information 
        df["Total Records"]=pd.to_numeric(df["Total Records"],errors = 'coerce')
        df=df.drop(["State","Date Made Public","Company","City","Description of incident","Information Source",
        "Type of organization","Source URL","Year of Breach","Total Records"],axis=1)
        print(df.head())
        
        print(df.isnull().sum())
        X=df.drop("Type of breach",axis=1)
        y=df["Type of breach"]
        print(X[0:2])
        print(y[0:2])
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression()
        log.fit(X_train,y_train)
        pred_log=log.predict(X_test)
        from sklearn.neighbors import KNeighborsClassifier
        knn=KNeighborsClassifier()
        knn.fit(X_train,y_train)
        pred_knn=knn.predict(X_test)
        from sklearn.naive_bayes import GaussianNB
        gn=GaussianNB()
        gn.fit(X_train,y_train)
        pred_gn=gn.predict(X_test)
        from sklearn.ensemble import RandomForestClassifier
        rf=RandomForestClassifier()
        rf.fit(X_train,y_train)
        pred_rf=rf.predict(X_test)
        print(X_test[0:10])
        print(pred_rf[0:10])
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix
        def plot_confusion_matrix(cm, title='CONFUSION MATRIX', cmap=plt.cm.Reds):
            target_names=['CARD', 'DISC', 'HACK', 'INSD', 'PHYS', 'PORT', 'STAT', 'UNKN']
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        confusionMatrix = confusion_matrix(y_test, pred_log)
        print('Confusion matrix,Random Forest')
        print(confusionMatrix)
        plot_confusion_matrix(confusionMatrix)
        plt.show()
        print("Accuracy Score")
        print("Logistic Regression: ",accuracy_score(pred_log,y_test))
        print("KNN : ",accuracy_score(pred_knn,y_test))
        print("Random Forest: ",accuracy_score(pred_rf,y_test))
        print("Naive Bayes: ",accuracy_score(pred_gn,y_test))
        from sklearn.ensemble import RandomForestClassifier
        rf=RandomForestClassifier()
        rf.fit(X,y)
        pred_data=np.array([[organisation_1,info_1,lat,long]],dtype=object)
        prediction=rf.predict(pred_data)
        cb=Cyber_Breaches.objects.create(Organisation=organisation,Records=records,
        Information_Source=info,Source_URL=url,Latitude=lat,Longitude=long,Type_Of_Breaches=prediction)
        cb.save()
        return render(request,"predict.html",{"url":url,"organisation":organisation,
        "info":info,"records":records,"lat":lat,"long":long,"prediction":prediction})
    return render(request,"data.html")


def predict(request):
    return render(request,"predict.html")

def adminhome(request):
    cb=Cyber_Breaches.objects.all()
    return render(request,"adminhome.html",{"cb":cb})

