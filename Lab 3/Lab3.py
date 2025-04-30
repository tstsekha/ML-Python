import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt

# 1. Использовать датасет из лабораторной работы №1
df = pd.read_csv('Lab 3/titanic.csv')

# Обработка данных
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df.drop(columns=["Cabin"], inplace=True)

# Создание новых признаков
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Масштабирование числовых признаков
scaler = MinMaxScaler()
df[["Age", "Fare", "SibSp", "Parch"]] = scaler.fit_transform(df[["Age", "Fare", "SibSp", "Parch"]])

# Преобразование категориальных признаков
df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Выбор признаков для моделей
features = ['Pclass', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S']
title_features = [col for col in df.columns if col.startswith('Title_')]
features.extend(title_features)

# 2. Задача регрессии (предсказание возраста) и оценка работоспособности модели
X_reg = df[features]
y_reg = df['Age']

# Разделение данных на обучающую и тестовую выборки
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Обучение модели регрессии
reg_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
reg_model.fit(X_reg_train, y_reg_train)

# Оценка модели регрессии
y_reg_pred = reg_model.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("Результаты регрессии:")
print(f"Среднеквадратичная ошибка: {mse:.2f}")
print(f"R2 score: {r2:.2f}")

# 3. Задача классификации (предсказание выживания) и оценка работоспособности модели
X_clf = df[features]
y_clf = df['Survived']

# Разделение данных на обучающую и тестовую выборки
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Обучение модели классификации
clf_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
clf_model.fit(X_clf_train, y_clf_train)

# Оценка модели классификации с помощью ROC-кривой
y_clf_pred_proba = clf_model.predict_proba(X_clf_test)[:, 1]
fpr, tpr, _ = roc_curve(y_clf_test, y_clf_pred_proba)
roc_auc = auc(fpr, tpr)

# Построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("\nРезультаты классификации:")
print(f"Площадь под ROC-кривой (AUC): {roc_auc:.2f}")
