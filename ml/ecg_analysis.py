import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from scipy.signal import find_peaks
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,  # Уровень логирования: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ecg_analysis.log"),  # Лог в файл
        logging.StreamHandler()  # Лог в консоль
    ]
)

logger = logging.getLogger(__name__)

def load_ecg_data(filepath):
    """Загрузка и предварительная обработка данных ЭКГ"""
    try:
        logger.info(f"Загрузка данных из файла: {filepath}")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, header=None)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath, header=None)
        else:
            raise ValueError("Unsupported file format")
        
        # Преобразование всех данных в числовой формат
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        data = df.values.flatten()[:1000]
        logger.info(f"Загружено {len(data)} точек данных")
        return data
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise

def extract_features(ecg_signal):
    """Извлечение признаков из сигнала ЭКГ"""
    try:
        logger.info("Начало извлечения признаков")
        features = [
            np.mean(ecg_signal), np.std(ecg_signal), np.min(ecg_signal), np.max(ecg_signal),
            np.median(ecg_signal), np.percentile(ecg_signal, 25), np.percentile(ecg_signal, 75),
            np.sum(np.abs(ecg_signal)), np.sum(np.square(ecg_signal)),
            len(find_peaks(ecg_signal)[0]), len(find_peaks(-ecg_signal)[0]),
            # Добавьте другие признаки по необходимости
        ]
        
        fft_vals = np.abs(np.fft.fft(ecg_signal)[:50])
        features.extend(fft_vals.tolist())
        
        # Вейвлет-признаки (заглушка)
        features.extend([0]*50)
        
        window_size = 50
        for i in range(0, len(ecg_signal)-window_size, window_size//2):
            window = ecg_signal[i:i+window_size]
            features.extend([np.mean(window), np.std(window), np.max(window)-np.min(window)])
        
        if len(features) < 361:
            features.extend([0]*(361 - len(features)))
        
        logger.info("Признаки успешно извлечены")
        return np.array(features[:361])
    
    except Exception as e:
        logger.error(f"Ошибка при извлечении признаков: {e}")
        raise

def analyze_ecg(input_path, output_path):
    try:
        logger.info(f"Начинается анализ файла: {input_path}")
        
        # Загрузка данных
        ecg_signal = load_ecg_data(input_path)
        
        # Извлечение признаков
        features = extract_features(ecg_signal)
        
        # Загрузка моделей
        model_dir = os.path.join(os.path.dirname(__file__), 'models/models')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        model_path = os.path.join(model_dir, 'ensemble_model.joblib')
        
        scaler = load(scaler_path)
        model = load(model_path)
        
        # Масштабирование и предсказание
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        probas = model.predict_proba(features_scaled)[0]
        
        class_names = {
            0: "Dangerous_VFL_VF",
            1: "Special_Form_VTTdP",
            2: "Threatening_VT",
            3: "Potential_Dangerous",
            4: "Supraventricular",
            5: "Sinus_rhythm"
        }
        
        result = {
            "main_diagnosis": class_names[np.argmax(probas)],
            "confidence": float(np.max(probas)),
            "predictions": {class_names[i]: float(probas[i]) for i in range(len(probas))},
            "ecg_image": os.path.basename(input_path).split('.')[0] + "_ecg.png"
        }
        
        # Визуализация сигнала ЭКГ
        plot_ecg(ecg_signal, os.path.join(os.path.dirname(output_path), result["ecg_image"]))
        
        # Сохранение результата в JSON файл
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
            
        logger.info(f"Анализ завершен успешно для файла: {input_path}")
        
    except Exception as e:
        logger.exception(f"Ошибка анализа файла {input_path}: {e}")
        

def plot_ecg(signal, save_path):
    try:
        plt.figure(figsize=(12, 4))
        plt.plot(signal)
        plt.title('ECG Signal')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close('all')
    except Exception as e:
         logger.error(f"Ошибка при сохранении графика ECG: {e}")
    finally:
         plt.close()

if __name__ == "__main__":
    #input_file = Path(sys.argv[1]).resolve()  # Преобразует в абсолютный путь
    #output_file = Path(sys.argv[2]).resolve()

    def to_raw_string(s):
        return repr(s)[1:-1]  # repr() даёт 'текст', убираем кавычки

    input_file = to_raw_string(sys.argv[1])
    output_file = to_raw_string(sys.argv[2])

    #input_file = "uploads\419_C_VFL_446_3s_full.csv"
    #output_file = "C:\Users\Acer\Desktop\prot2\public\output\419_C_VFL_446_3s_full.csv_result.json"
    
    logger.info(f"Запуск анализа для файла: {input_file}")
    sys.stdout.flush()
    
    analyze_ecg(input_file, output_file)

    sys.stdout.flush()
    sys.exit(0)