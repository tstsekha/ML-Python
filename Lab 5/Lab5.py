import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загрузка данных
X = np.loadtxt('Lab 5/dataIn.txt')
y = np.loadtxt('Lab 5/dataOut.txt')

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Learning rate schedule
initial_learning_rate = 0.001
decay_rate = 0.1
decay_steps = 1000

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate
)

# Создание модели согласно заданию
model = keras.Sequential([
    keras.layers.Dense(8, activation='sigmoid', input_shape=(12,)),  # Один скрытый слой с sigmoid
    keras.layers.Dense(2, activation='softmax')  # Выходной слой
])

# Компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Оценка модели
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Сохранение модели
model.save('Lab 5/election_model.h5') 