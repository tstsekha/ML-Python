import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузить данные, взяли https://www.kaggle.com/datasets/brendan45774/test-file/data
df = pd.read_csv("Lab1/titanic.csv")                                    # Чтение CSV файла, скаченного с сайта
print(df.info())                                                        # Вывод информации о столбцах датасета
print(df)                                                               # Вывод датасета (начало и конец)

# 2. Заполнить нулевые значения
df["Age"] = df["Age"].fillna(df["Age"].median())                        # Заполнение нулевых Age медианой
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())                       # Заполнение нулевых Fare средним значением
df.drop(columns=["Cabin"], inplace = True)                              # Удаляем столбец Cabin, так как данных мало

# 3. Провести нормализацию данных
scaler = MinMaxScaler()                                                 # Нормализация с помощью Min-Max Scaling
df[["Age", "Fare", "SibSp", "Parch"]] = scaler.fit_transform(df[["Age", "Fare", "SibSp", "Parch"]])

# 4. Преобразовать категориальные данные в численные
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)   # One-hot encoding для категориальных данных
print(df.info())
print(df)