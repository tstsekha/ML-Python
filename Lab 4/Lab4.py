import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 1. Использовать датасет из лабораторной работы №2
data = pd.read_csv('Lab 4/iris.csv')
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Построить модели классификации на основе случайных лесов и градиентного бустинга
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Градиентный бустинг
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# 3. Оценить качество работы моделей с помощью метрик accuracy, precision, recall, f1-score
print("\nМетрики для Случайного леса:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1-Score:", f1_score(y_test, rf_pred, average='weighted'))
print("Precision:", precision_score(y_test, rf_pred, average='weighted'))
print("Recall:", recall_score(y_test, rf_pred, average='weighted'))

print("\nМетрики для Градиентного бустинга:")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("F1-Score:", f1_score(y_test, gb_pred, average='weighted'))
print("Precision:", precision_score(y_test, gb_pred, average='weighted'))
print("Recall:", recall_score(y_test, gb_pred, average='weighted'))

# Перекрестная проверка
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='accuracy')

print("\nРезультаты перекрестной проверки:")
print("Случайный лес (средняя точность):", rf_cv.mean())
print("Градиентный бустинг (средняя точность):", gb_cv.mean())

# Визуализация результатов
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
rf_scores = [
    accuracy_score(y_test, rf_pred),
    f1_score(y_test, rf_pred, average='weighted'),
    precision_score(y_test, rf_pred, average='weighted'),
    recall_score(y_test, rf_pred, average='weighted')
]
gb_scores = [
    accuracy_score(y_test, gb_pred),
    f1_score(y_test, gb_pred, average='weighted'),
    precision_score(y_test, gb_pred, average='weighted'),
    recall_score(y_test, gb_pred, average='weighted')
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, rf_scores, width, label='Случайный лес')
rects2 = ax.bar(x + width/2, gb_scores, width, label='Градиентный бустинг')

ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение моделей')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()
