import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import seaborn as sns

data = pd.read_csv('train (1).csv')
df = data.copy()

# CLeaning Data
def cleaner(dataframe):
    for i in dataframe.columns:
        if (dataframe[i].isnull().sum()/len(dataframe) * 100) > 30:
            dataframe.drop(i, inplace = True, axis = 1)

        elif dataframe[i].dtypes != 'O':
            dataframe[i].fillna(dataframe[i].median(), inplace = True)

        else:
            dataframe[i].fillna(dataframe[i].mode()[0], inplace = True)

    print(dataframe.isnull().sum().sort_values(ascending = False).head())
    print(f'\n\t\t\t\t\t\t\t\tDATAFRAME IS CLEANED')
    return dataframe

cleaner(df)

# Transformer Data
def transformer(dataframe):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    encoder = LabelEncoder()

    for i in dataframe.columns:
        if dataframe[i].dtypes != 'O':
            dataframe[i] = scaler.fit_transform(dataframe[[i]])
        else:
            dataframe[i] = encoder.fit_transform(dataframe[i])
    return dataframe

df = transformer(df.drop('SalePrice', axis = 1))
df.head()

# 
sel_cols = ['LotArea', 'LotFrontage', 'MSSubClass', 'BsmtUnfSF', 'GrLivArea', 'GarageArea', 'BsmtFinSF1']
new_data = df[sel_cols]
new_data = pd.concat([new_data, data['SalePrice']], axis = 1)
print(new_data.head())

# split into train and test
x = new_data.drop('SalePrice', axis = 1)
y = new_data.SalePrice

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 82)
print('All is Well\n\n')

# Modelling
model = RandomForestRegressor()
model.fit(xtrain, ytrain)
print(f'\nRandom Forest Regressor Model has been Created \n')

model = pickle.load(open('housePriceModel.pkl', "rb"))

st.markdown("<h1 style = 'color: #0A2647; text-align: center; font-family: helvetica '>HOUSE PRICE PREDICTION</h1>", unsafe_allow_html = True)

st.markdown("<h4 style = 'margin: -30px; color: #FF87CA; text-align: center; font-family: cursive '>Built By Dullar Tech</h4>", unsafe_allow_html = True)
st.image('pngwng.com.png')

password = ['one', 'two', 'three']
username = st.text_input('Pls enter your username')
passes = st.text_input('Pls input password')

if passes in password:
    st.toast('Registered User')
    print(f'Welcome {username}, Pls enjoy your usage as a registered user')
else:
    st.error('You are not a registered user. But you have three trials')

st.sidebar.image('pngwing.com (1).png', caption = f'Welcome {username}')

dx = data[['LotArea', 'LotFrontage', 'MSSubClass', 'BsmtUnfSF', 'GrLivArea', 'GarageArea', 'BsmtFinSF1']]
st.write(dx.head())

st.markdown('<br><br>', unsafe_allow_html= True)
# INPUT FEATURES
input_type = st.sidebar.radio("Select Your Preferred Input Style", ["Slider", "Number Input"])
st.markdown('<br><br>', unsafe_allow_html= True)

if input_type == 'Slider':
    ltArea = st.sidebar.slider('Lot Size', dx['LotArea'].min(), dx['LotArea'].max())
    ltFront = st.sidebar.slider('Lot Frontage', dx['LotFrontage'].min(), dx['LotFrontage'].max())
    mssubClass = st.sidebar.slider('Building Class', dx['MSSubClass'].min(), dx['MSSubClass'].max())
    bsmUnf = st.sidebar.slider('Unfinished Square Feet of Basement Area', dx['BsmtUnfSF'].min(), dx['BsmtUnfSF'].max())
    mssubClass = st.sidebar.slider('Above Grade (ground) Living Area Square Feet', dx['GrLivArea'].min(), dx['GrLivArea'].max())
    garage = st.sidebar.slider('Garage Size', dx['GarageArea'].min(), dx['GarageArea'].max())
    squareFeet = st.sidebar.slider('Finished Square Feet', dx['BsmtFinSF1'].min(), dx['BsmtFinSF1'].max())
else:
    ltArea = st.sidebar.number_input('Lot Size', dx['LotArea'].min(), dx['LotArea'].max())
    ltFront = st.sidebar.number_input('Lot Frontage', dx['LotFrontage'].min(), dx['LotFrontage'].max())
    mssubClass = st.sidebar.number_input('Building Class', dx['MSSubClass'].min(), dx['MSSubClass'].max())
    bsmUnf = st.sidebar.number_input('Unfinished square feet of basement area', dx['BsmtUnfSF'].min(), dx['BsmtUnfSF'].max())
    mssubClass = st.sidebar.number_input('Above grade (ground) living area square feet', dx['GrLivArea'].min(), dx['GrLivArea'].max())
    garage = st.sidebar.number_input('Garage Size', dx['GarageArea'].min(), dx['GarageArea'].max())
    squareFeet = st.sidebar.number_input('finished square feet', dx['BsmtFinSF1'].min(), dx['BsmtFinSF1'].max())

st.header('Input Values')
# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'LotArea':ltArea, 'LotFrontage': ltFront, 'MSSubClass': mssubClass, 'BsmtUnfSF':bsmUnf, 'GrLivArea':garage, 'GarageArea': garage, 'BsmtFinSF1':squareFeet}])

st.write(input_variable)

# Standard Scale the Input Variable.
for i in input_variable.columns:
    input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('Profitability Predicted')
    st.image('pngwing2.com.png', width = 200)
    st.success('Predicted. Pls check the Interpretation Tab for interpretation')

    st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Model Interpretation</h2>", unsafe_allow_html = True)

