from django.shortcuts import render

from django.conf import settings
from django.core.files.storage import FileSystemStorage

nn="media/creditcard.csv"
def preloader(request):
    return render(request,'preloader.html',{})

def pre1(request):
    return render(request,'preloader1.html',{})

def loader2(request):
    return render(request,'preloader2.html',{})

def preload3(request):
    return render(request,'preloader3.html',{})

def handler404(request):
    return render(request, '404.html', status=404)

def handler500(request):
    return render(request, '500.html', status=500)

def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        global nn
        nn = uploaded_file_url
        return render(request, 'home.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'home.html')

def startTraining(request):
            
           #!/usr/bin/env python
            import pandas as pd
            import numpy as np
            import keras

            np.random.seed(2)
            data = pd.read_csv(nn)
            
            count=data.shape
            
            mv=data.isnull().values.any()
            
            frauds = data[data.Class == 1]
            normal = data[data.Class == 0]

            fd=frauds.shape
            nm=normal.shape

            data.head()

            from sklearn.preprocessing import StandardScaler
            data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
            data = data.drop(['Amount'],axis=1)

            data.head()

            data = data.drop(['Time'],axis=1)
            data.head()

            X = data.iloc[:, data.columns != 'Class']
            y = data.iloc[:, data.columns == 'Class']

            y.head()

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

            X_train.shape

            X_test.shape

            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import confusion_matrix
            decision_tree = DecisionTreeClassifier()

            decision_tree.fit(X_train,y_train.values.ravel())
            y_pred = decision_tree.predict(X)
            p = decision_tree.score(X_test,y_test)
            p=p*100
            y_expected = pd.DataFrame(y)
            cm = confusion_matrix(y_expected,y_pred.round())
            classes=[0,1]
            normalize=False
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
            
            d=cm[0]
            q=cm[1]
            st="Decision Tree"

            return render(request, 'new.html',{'d2':d[0],'y':d[1],'q':q[0],'r':q[1],'p1':p,'str':st,
                                               'count1':count[0],'mvs':mv,'fd1':fd[0],'nm1':nm[0], 'nn':nn, 
                                              })

def startAnalysing(request):
    import pandas as pd
    import numpy as np
    import keras

    np.random.seed(2)
    data = pd.read_csv(nn)
    count=data.shape
            
    mv=data.isnull().values.any()
            
    frauds = data[data.Class == 1]
    normal = data[data.Class == 0]

    fd=frauds.shape
    nm=normal.shape
    data.head()
    from sklearn.preprocessing import StandardScaler
    data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
    data = data.drop(['Amount'],axis=1)
    data = data.drop(['Time'],axis=1)
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']
    y.head()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
    X_train.shape
    X_test.shape
    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train,y_train.values.ravel())
    y_pred = random_forest.predict(X_test)
    p=random_forest.score(X_test,y_test)
    p=p*100
    import matplotlib.pyplot as plt
    import itertools

    from sklearn import svm, datasets
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(cm, classes,
                            normalize=False):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        return cm
        
    cnf_matrix = confusion_matrix(y_test,y_pred)
    cm1=plot_confusion_matrix(cnf_matrix,classes=[0,1])
    y_pred = random_forest.predict(X)
    y_pred
    cnf_matrix = confusion_matrix(y,y_pred.round())
    c=plot_confusion_matrix(cnf_matrix,classes=[0,1])
    d=c[0]
    q=c[1]
    st="Random Forest"
    return render(request, 'new.html',{'d2':d[0],'y':d[1],'q':q[0],'r':q[1],'p1':p,'str':st,
                                       'count1':count[0],'mvs':mv,'fd1':fd[0],'nm1':nm[0], 'nn':nn, 
                                      })


def startLearning(request):
    import pandas as pd
    import numpy as np
    import keras

    np.random.seed(2)
    data = pd.read_csv(nn)
    count=data.shape
            
    mv=data.isnull().values.any()
            
    frauds = data[data.Class == 1]
    normal = data[data.Class == 0]

    fd=frauds.shape
    nm=normal.shape
    from sklearn.preprocessing import StandardScaler
    data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
    data = data.drop(['Amount'],axis=1)
    data = data.drop(['Time'],axis=1)
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
    X_train.shape
    X_test.shape
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    '''Deep neural network'''
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),])
    model.summary()
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=15,epochs=5)
    score = model.evaluate(X_test, y_test)
    p=score[1]
    p=p*100
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(cm, classes,
                            normalize=False):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        return cm
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred.round())
    print(cnf_matrix)
    cm1=plot_confusion_matrix(cnf_matrix, classes=[0,1])
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred.round())
    print(cnf_matrix)
    cm2=plot_confusion_matrix(cnf_matrix, classes=[0,1])


    d=cm2[0]
    q=cm2[1]
    st="Deep Learning"
    return render(request, 'new.html',{'d2':d[0],'y':d[1],'q':q[0],'r':q[1],'p1':p,'str':st,
                                       'count1':count[0],'mvs':mv,'fd1':fd[0],'nm1':nm[0], 'nn':nn, 
                                      })



def startSampling(request):
    import pandas as pd
    import numpy as np
    import keras

    np.random.seed(2)
    data = pd.read_csv(nn)
    from sklearn.preprocessing import StandardScaler
    data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
    data = data.drop(['Amount'],axis=1)
    data = data.drop(['Time'],axis=1)
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
    X_train.shape
    X_test.shape
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    '''Deep neural network'''
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    
    model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),])
    model.summary()
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=15,epochs=5)

    score = model.evaluate(X_test, y_test)
    print(score)
    
    from sklearn.metrics import confusion_matrix
    def plot_confusion_matrix(cm, classes,
                            normalize=False):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        return cm
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred.round())
    print(cnf_matrix)
    cm1=plot_confusion_matrix(cnf_matrix, classes=[0,1])
    
    
    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    print(cnf_matrix)
    cm2=plot_confusion_matrix(cnf_matrix, classes=[0,1])
    
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred.round())
    print(cnf_matrix)
    cm3=plot_confusion_matrix(cnf_matrix, classes=[0,1])
    
    
    #underSampling
    fraud_indices = np.array(data[data.Class == 1].index)
    number_records_fraud = len(fraud_indices)
    print(number_records_fraud)

    normal_indices = data[data.Class == 0].index

    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    print(len(random_normal_indices))

    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
    print(len(under_sample_indices))

    under_sample_data = data.iloc[under_sample_indices,:]

    X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

    X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model.summary()

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=15,epochs=5)

    y_pred = model.predict(X_test)
    y_expected = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm4=plot_confusion_matrix(cnf_matrix, classes=[0,1])


    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm5=plot_confusion_matrix(cnf_matrix, classes=[0,1])


    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm6=plot_confusion_matrix(cnf_matrix, classes=[0,1])

    import os
    cmd = 'pip install --user imbalanced-learn'
    os.system(cmd)
    from imblearn.over_sampling import SMOTE

    X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())

    y_resample = pd.DataFrame(y_resample)
    X_resample = pd.DataFrame(X_resample)

    X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=15,epochs=5)

    y_pred = model.predict(X_test)
    y_expected = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm7=plot_confusion_matrix(cnf_matrix, classes=[0,1])

    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm8=plot_confusion_matrix(cnf_matrix, classes=[0,1])

    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm9=plot_confusion_matrix(cnf_matrix, classes=[0,1])

    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm11=plot_confusion_matrix(cnf_matrix, classes=[0,1])

    y_pred = model.predict(X)
    y_expected = pd.DataFrame(y)
    cnf_matrix = confusion_matrix(y_expected, y_pred.round())
    cm12=plot_confusion_matrix(cnf_matrix, classes=[0,1])

    

    d=cm12[0]
    q=cm12[1]
    st="Deep Learning with sampling"
    return render(request, 'new.html',{'d2':d[0],'str':st,'y':d[1],'q':q[0],'r':q[1]})

        

    





        


