import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def credit():

    # Wczytywanie danych
    data = pandas.read_csv(
        "kredyt_dane_2.csv", keep_default_na=True, sep=';')
    # Tworzenie listy nazw kolumn
    columns_names = data.columns.values.tolist()

    print("Czy NA? \n", data.isna().sum(), "\n")  # informacje o brakach danych
    # uzupełnianie pustych miesc średnią w zmiennych ilościowych
    data = data.fillna(data.mean())
    # informacje o brakach danych
    print("Czy NA? \n", data.isna().sum(), "\n")

    # zmienne Gender, Married, Self_Employed i Credit_History to zmienne kategoryczne. Braki w danych uzupełniono dominantą.

    data = data.fillna(data.mode().iloc[0])
    print("Is NA? \n", data.isna().sum(), "\n")

    # Zmienną “Loan_ID” usunuęto, gdyż nie wnosi ona żadnej istotnej informacji.

    data.drop('Loan_ID', axis=1, inplace=True)

   # Zamiana zmiennych jakościowych na wartości liczbowe
    data.loc[data["Gender"] == "Male", "Gender"] = 1
    data.loc[data["Gender"] == "Female", "Gender"] = 0
    data.loc[data["Married"] == "Yes", "Married"] = 1
    data.loc[data["Married"] == "No", "Married"] = 0
    data.loc[data["Education"] == "Graduate", "Education"] = 1
    data.loc[data["Education"] == "Not Graduate", "Education"] = 0
    data.loc[data["Self_Employed"] == "Yes", "Self_Employed"] = 1
    data.loc[data["Self_Employed"] == "No", "Self_Employed"] = 0
    data.loc[data["Credit_History"] == "Y", "Credit_History"] = 1
    data.loc[data["Credit_History"] == "N", "Credit_History"] = 0
    data.loc[data["Loan_Status"] == "Y", "Loan_Status"] = 1
    data.loc[data["Loan_Status"] == "N", "Loan_Status"] = 0
    data.loc[data["Dependents"] == "3", "Dependents"] = 3

    Y = data['Loan_Status']

    data.drop('Loan_Status', axis=1, inplace=True)
    X = data.values

    # Tworzenie zbioru testowego i uczącego
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, random_state=1)
    print('xtrain', X_train)
    print('xtest', X_test)
    print('ytrain', Y_train)
    print('ytest', Y_test)

    Y_train = np.array(Y_train).astype('int')
    Y_test = np.array(Y_test).astype('int')
    print('ytrain', type(Y_train))
    print('ytest', type(Y_test))
    # Stworzenie macierzy zer, w której zapisywane będą wyniki modeli
    wyniki = [0] * 5


# REGRESJA LOGISTYCZNA


    klasyfikator = LogisticRegression(solver='liblinear')
    klasyfikator.fit(X_train, Y_train)
    Y_pred = klasyfikator.predict(X_train)
    print("REGRESJA LOGISTYCZNA")
    print("Dokladnosc dla treningowych: ", accuracy_score(Y_train, Y_pred))
    Y_pred = klasyfikator.predict(X_test)
    print("Dokladnosc dla uczacych: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    wyniki[0] = f1_score(Y_test, Y_pred, average='macro')


# K-NAJBLIZSZYCH SASIADOW



    klasyfikator = KNeighborsClassifier(n_neighbors=10)
    klasyfikator.fit(X_train, Y_train)
    Y_pred = klasyfikator.predict(X_train)
    print("K-NAJBLIZSZYCH SASIADOW")
    print("Dokladnosc dla treningowych: ", accuracy_score(Y_train, Y_pred))
    Y_pred = klasyfikator.predict(X_test)
    print("Dokladnosc dla uczacych: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    wyniki[1] = f1_score(Y_test, Y_pred, average='macro')

# DRZEWA DECYZYJNE


    klasyfikator = tree.DecisionTreeClassifier()
    klasyfikator.fit(X_train, Y_train)
    Y_pred = klasyfikator.predict(X_train)
    print("DRZEWA KLASYFIKACYJNE")
    print("Dokladnosc dla treningowych: ", accuracy_score(Y_train, Y_pred))
    Y_pred = klasyfikator.predict(X_test)
    print("Dokladnosc dla uczacych: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    wyniki[2] = f1_score(Y_test, Y_pred, average='macro')

# LAS LOSOWY


    klasyfikator = RandomForestClassifier()
    klasyfikator.fit(X_train, Y_train)
    Y_pred = klasyfikator.predict(X_train)
    print("LAS LOSOWY")
    print("Dokladnosc dla treningowych: ", accuracy_score(Y_train, Y_pred))
    Y_pred = klasyfikator.predict(X_test)
    print("Dokladnosc dla uczacych: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    wyniki[3] = f1_score(Y_test, Y_pred, average='macro')

# MLP


    klasyfikator = MLPClassifier(random_state=0)
    klasyfikator.fit(X_train, Y_train)
    Y_pred = klasyfikator.predict(X_train)
    print("MLP")
    print("Dokladnosc dla treningowych: ", accuracy_score(Y_train, Y_pred))
    Y_pred = klasyfikator.predict(X_test)
    print("Dokladnosc dla uczacych: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    wyniki[4] = f1_score(Y_test, Y_pred, average='macro')

    wyniki_id = {"Regresja logistyczna": wyniki[0], "KNN": wyniki[1],
                 "Drzewa decyzyjne": wyniki[2], "Random Forest": wyniki[3], " MLP": wyniki[4]}

    for x, y in wyniki_id.items():
        print(x, y)

    print("\n Najbardziej dopasowany model to: ",
          max(wyniki_id, key=wyniki_id.get))