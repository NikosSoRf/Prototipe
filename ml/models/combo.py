import pandas as pd
import os
import numpy as np
from tqdm import tqdm  #прогресс-бар

# Конфигурация данных
DATA_FOLDERS = {
    1: 'data/1_Dangerous_VFL_VF/full',
    2: 'data/2_Special_Form_VTTdP/full',
    3: 'data/3_Threatening_VT/full',
    4: 'data/4_Potential_Dangerous/full',
    5: 'data/5_Supraventricular/full',
    6: 'data/6_Sinus_rhythm/full'
}

CLASS_NAMES = {
    1: "Dangerous_VFL_VF",
    2: "Special_Form_VTTdP",
    3: "Threatening_VT",
    4: "Potential_Dangerous",
    5: "Supraventricular",
    6: "Sinus_rhythm"
}

def process_file(file_path, class_label):
    """Обработка одного файла с данными ЭКГ"""
    try:
        # Читаем данные (предполагаем, что это CSV без заголовков)
        ecg_data = pd.read_csv(file_path, header=None)
        
        # Преобразуем в 1D массив и нормализуем
        ecg_flattened = ecg_data.values.astype(np.float32).flatten()
        
        # Добавляем метку класса
        return [class_label, *ecg_flattened]
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return None

def combine_ecg_data(output_file='combined_data.csv'):
    """Объединение данных ЭКГ из всех папок"""
    all_data = []
    
    for class_label, folder_path in DATA_FOLDERS.items():
        print(f"\nОбработка класса {class_label} ({CLASS_NAMES[class_label]})...")
        
        # Проверяем существование папки
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует!")
            continue
            
        # Получаем список файлов
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        # Обрабатываем файлы с прогресс-баром
        for file in tqdm(files, desc=f"Class {class_label}"):
            file_path = os.path.join(folder_path, file)
            processed = process_file(file_path, class_label)
            if processed:
                all_data.append(processed)
    
    # Создаем DataFrame
    if not all_data:
        print("Нет данных для обработки!")
        return
    
    # Определяем количество признаков (длина самой длинной записи)
    max_length = max(len(row) for row in all_data)
    columns = ['class'] + [f'ecg_{i}' for i in range(max_length-1)]
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Сохраняем результат
    df.to_csv(output_file, index=False)
    print(f"\nДанные успешно сохранены в {output_file}")
    print(f"Общее количество записей: {len(df)}")
    print("Распределение по классам:")
    print(df['class'].value_counts().sort_index().rename(CLASS_NAMES))
    
    return df

if __name__ == "__main__":
    combined_data = combine_ecg_data()