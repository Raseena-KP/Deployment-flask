#Iris Classification Prediction Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

#reading the dataset to python 
data=pd.read_csv("framingham.csv") 
# Handling Missing values
data.isna().sum()
data['education'].fillna(data['education'].mode()[0],inplace=True)
data['cigsPerDay'].fillna(20,inplace=True)
BP_nullindex = data[data['BPMeds'].isnull()].index
for i in BP_nullindex:
    if (data['sysBP'][i] > 140 or data['diaBP'][i] > 90):
        data.loc[i,'BPMeds'] = 1.0  
    else:
        data.loc[i,'BPMeds'] = 0.0
def impute_glucose(cols):
    dia=cols[0]
    glu=cols[1]
    if pd.isnull(glu):
        if dia == 0:
            return 78
        else:
            return 145
    else:
        return glu
data['glucose'] = data[['diabetes','glucose']].apply(impute_glucose,axis=1)
data['BMI'].fillna(data['BMI'].median(),inplace=True)
data['totChol'].fillna(data['totChol'].median(),inplace=True)
data['heartRate'].fillna(data["heartRate"].mode()[0],inplace=True)

#Handling outliers in dataset
Q1 = data['cigsPerDay'].quantile(0.25)
Q3 = data['cigsPerDay'].quantile(0.75)
IQR = Q3-Q1
LL=Q1-1.5*IQR #LL - Lower Limit
UL=Q3+1.5*IQR #UL - Upper Limit
data['cigsPerDay'] = np.where(data['cigsPerDay']>UL,UL,np.where(data['cigsPerDay']<LL,LL,data['cigsPerDay']))

#Feature Engineering
data['MAP'] = (data['sysBP'] + (2* data['diaBP']))/3
def Weight(a):
    if a<18.5:
        return "Underweight"
    if 18.5<=a<24.9:
        return "Normal"
    elif 24.9<=a<29.9:
        return "Overweight"
    elif a>29.9:
        return "Obese"
data["Weight"]=data["BMI"].apply(lambda x: Weight(x))


# Label encoding the Classification column
data['Weight'] = data['Weight'].map({'Underweight': 1, 'Normal': 2,'Overweight':3,'Obese':4})

# Feature Reduction
data.drop(['currentSmoker','sysBP','diaBP'],axis=1,inplace=True)

#Statndardisation
df_con=['cigsPerDay', 'totChol','BMI','heartRate', 'glucose', 'MAP','age']

# Split the data into features & target variable
X = data.drop(['TenYearCHD'],axis=1)
y = data['TenYearCHD']

# Splitting the data into train & Test set
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=45,test_size=0.2)

# train the model using train data set
LR = LogisticRegression()
LR.fit(X_train,y_train)

# Saving the model to disk
pickle.dump(LR, open("model.pkl","wb"))
