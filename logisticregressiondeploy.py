import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("Titanic Survival Prediction: Logistic Regression Model")

st.sidebar.header("User Input Parameters")

def user_input_features():
    Pclass = st.sidebar.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", (1, 2, 3))
    Age = st.sidebar.number_input("Age", 0.0, 100.0)
    SibSp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
    Parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    Fare = st.sidebar.number_input("Fare Paid", 0.0, 500.0, 0.0)
    Sex = st.sidebar.selectbox("Sex", ["male", "female"])
    Embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])
    data = {
        "Pclass": Pclass,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Sex": Sex,
        "Embarked": Embarked
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader("User Input Parameters")
st.write(df)

titanic = pd.read_csv(r"C:\Users\alekhya\Downloads\Titanic_train.csv")

titanic = titanic.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
titanic["Embarked"].fillna(titanic["Embarked"].mode()[0], inplace=True)
titanic = pd.get_dummies(titanic, columns=["Sex", "Embarked"], drop_first=True)

X = titanic.drop("Survived", axis=1)
Y = titanic["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_scaled, Y_train)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
df = df.reindex(columns=X.columns, fill_value=0)
df_scaled = scaler.transform(df)

prediction = clf.predict(df_scaled)
prediction_proba = clf.predict_proba(df_scaled)

st.subheader("Predicted Result")
st.write("Survived" if prediction[0] == 1 else "Not Survived")

st.subheader("Prediction Probability")
st.write(f"Probability of Survival: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of Not Surviving: {prediction_proba[0][0]:.2f}")