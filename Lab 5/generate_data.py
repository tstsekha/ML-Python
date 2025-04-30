import numpy as np

# Генерация тестовых данных
np.random.seed(42)

# Генерируем 5000 примеров
n_samples = 5000
X = np.random.randint(0, 2, size=(n_samples, 12))

# Создаем зависимость: если большинство первых 6 признаков = 1, то победит первый кандидат
# если большинство вторых 6 признаков = 1, то победит второй кандидат
Y = np.zeros((n_samples, 2))

for i in range(n_samples):
    first_half = np.sum(X[i, :6])  # сумма первых 6 признаков
    second_half = np.sum(X[i, 6:]) # сумма вторых 6 признаков
    
    if first_half > second_half:
        Y[i] = [1, 0]  # победа первого кандидата
    else:
        Y[i] = [0, 1]  # победа второго кандидата

# Сохранение данных в файлы
np.savetxt('Lab 5/dataIn.txt', X, fmt='%d')
np.savetxt('Lab 5/dataOut.txt', Y, fmt='%d')

print("Данные успешно сгенерированы и сохранены в файлы dataIn.txt и dataOut.txt") 