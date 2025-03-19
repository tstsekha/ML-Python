import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 0. Загрузить данные и разделить их, взяли https://www.kaggle.com/datasets/himanshunakrani/iris-dataset/data
df = pd.read_csv("Lab2/iris.csv")
print(df.info())
print(df)

label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])              # Преобразуем категориальный столбец 'species' в числовой

X = df.drop('species', axis = 1)                                        # Разделим данные на признаки и целевую переменную
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)   # Разделим на обучающую и тестовую выборки

# 1. а) Классификация объектов по категориям на основе логистической регрессии
logreg = LogisticRegression()                                           # Логистическая регрессия
logreg.fit(X_train, y_train)                                            # Её обучение
y_pred_logreg = logreg.predict(X_test)                                  # Предсказания

# 1. б) Классификация объектов по категориям на основе линейной регрессии
linreg = LinearRegression()                                             # Линейная регрессия
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
y_pred_linreg = y_pred_linreg.round().astype(int)                       # Округляем предсказания для линейной регрессии до целых чисел

# 2. Оценка результатов моеделей и построение матриц ошибок
print("Логистическая регрессия:")
print(classification_report(y_test, y_pred_logreg))
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
print("Матрица ошибок для логистической регрессии:")
print(cm_logreg)

print("Линейная регрессия:")
print(classification_report(y_test, y_pred_linreg))
cm_linreg = confusion_matrix(y_test, y_pred_linreg)
print("Матрица ошибок для линейной регрессии:")
print(cm_linreg)