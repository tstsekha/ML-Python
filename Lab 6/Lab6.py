import numpy as np 
import tensorflow as tf 
import pandas as pd
import os
import random

# Загрузка данных
train_data = pd.read_csv('Lab 6/sign_mnist_train.csv')
test_data = pd.read_csv('Lab 6/sign_mnist_test.csv')

# Подготовка данных
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(train_data.iloc[:, 0].values)

X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test = tf.keras.utils.to_categorical(test_data.iloc[:, 0].values)

# Проверка наличия сохраненной модели
model_path = 'Lab 6/sign_language_model.h5'
if os.path.exists(model_path):
    use_saved = input("Найдена сохраненная модель. Использовать её? (y/n): ").lower() == 'y'
    if use_saved:
        model = tf.keras.models.load_model(model_path)
        print("Модель загружена из файла")
    else:
        print("Начинаем обучение новой модели")
        use_saved = False
else:
    use_saved = False

if not use_saved:
    # Улучшенная архитектура модели
    model = tf.keras.models.Sequential([
        # Первый сверточный блок
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        
        # Второй сверточный блок
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        
        # Полносвязные слои
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(25, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Обучение модели
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=5,
                        batch_size=64)

    # Сохранение модели
    model.save(model_path)
    print("Модель сохранена в файл")

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Проверка на случайных примерах
print("\nПроверка на случайных примерах:")
indices = random.sample(range(len(X_test)), 10)
for idx in indices:
    true_label = np.argmax(y_test[idx])
    pred = model.predict(X_test[idx:idx+1])
    pred_label = np.argmax(pred)
    print(f"Ожидаемый класс: {true_label}, Предсказанный класс: {pred_label}, {'✓' if true_label == pred_label else '✗'}")

# Функция для предсказания жеста
def predict_gesture(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, 
                                              target_size=(28, 28),
                                              color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    
    prediction = model.predict(img_array)
    return np.argmax(prediction)