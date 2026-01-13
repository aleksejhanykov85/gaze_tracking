import kagglehub
import os

# Укажите путь, куда вы хотите скачать данные внутри вашего проекта
target_dir = os.path.join('data', 'mpiigaze')

# Создаём папку, если её нет
os.makedirs(target_dir, exist_ok=True)

print(f'Скачивание датасета MPIIGaze в папку: {target_dir}')
# Скачиваем датасет
path = kagglehub.dataset_download("dhruv413/mpiigaze")
print(f'Датасет скачан по пути: {path}')

# (Опционально) Скопируйте или переместите файлы в нужную вам структуру
# Это может потребовать дополнительного кода в зависимости от структуры архива