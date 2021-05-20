#Importing Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#1. Display Title
st.header('Heart Disease Prediction')

#2. Image selection
image=Image.open('heart.jpg')
st.image(image, caption='Heart Disease Detection using ML', use_column_width=True)

#3. Get Data
df=pd.read_csv('heart-20000.csv')

del df['id']
df.drop(df[df['ap_hi']>250].index, inplace = True)
df.drop(df[df['ap_hi']<60].index, inplace = True)
df.drop(df[df['ap_lo']>180].index, inplace = True)
df.drop(df[df['ap_lo']<50].index, inplace = True)

#4. Set a subheader
st.subheader('Data Information:')
#Show data as a table
st.dataframe(df)
#Give description on the data
st.write(df.describe())
#Show as chart
chart=st.line_chart(df)

#5. X independent and y dependent
X=df.iloc[:, 0:11].values
Y=df.iloc[:, -1].values

#6. Split data into 75% training and 25% testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.005, random_state=0)

#7. Feature input from user
def get_user_input():
    Age_in_days = st.sidebar.slider('age', 2000, 30000, 10000)
    Gender = st.sidebar.slider('gender', 1, 2, 1)
    Height = st.sidebar.slider('height', 120, 210, 175)
    Weight = st.sidebar.slider('weight', 50, 200, 80)
    AP_hi = st.sidebar.slider('ap_hi', 100, 250, 120)
    AP_lo = st.sidebar.slider('ap_lo', 50, 110, 90)
    Cholestrol = st.sidebar.slider('cholestrol', 1, 3, 2)
    Glucose = st.sidebar.slider('gluc', 1, 3, 2)
    Smoking = st.sidebar.slider('smoke', 0, 1, 0)
    Alcohol = st.sidebar.slider('alco', 0, 1, 0)
    Active = st.sidebar.slider('active', 0, 1, 1)

    #Store a dictionary into a variable
    user_data = {
        'Age (in days)':Age_in_days,
        'Gender':Gender,
        'Height':Height,
        'Weight':Weight,
        'AP_hi':AP_hi,
        'AP_lo':AP_lo,
        'Cholestrol':Cholestrol,
        'Glucose':Glucose,
        'Smoking intake':Smoking,
        'Alcohol intake':Alcohol,
        'Active':Active
    }
    #Reansforming into dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

#8. Store user input into a variable
user_input = get_user_input()

#9. Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#10. Create and train the model
classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)
#11. Show the models metrics
st.subheader('Model Test Accuracy Score:')
score = accuracy_score(Y_test, prediction) * 100
st.write(score, '%')

#12. Store the models predictions in a variable
prediction1 = classifier.predict(user_input)

#13. Set a subheader and display
st.subheader('Classification: ')
st.write(prediction1)
