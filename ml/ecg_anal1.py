import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from scipy.signal import find_peaks

def load_ecg_data(filepath):
    """Загрузка и предварительная обработка данных ЭКГ"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, header=None)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath, header=None)
        else:
            raise ValueError("Unsupported file format")
        
        # Преобразование всех данных в числовой формат
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.values.flatten()[:1000]  # Берем первые 1000 точек для единообразия
    
    except Exception as e:
        raise ValueError(f"Error loading ECG data: {str(e)}")

def extract_features(ecg_signal):
    """Извлечение 361 признака из сигнала ЭКГ"""
    try:
        # 1. Базовые статистические признаки (20)
        features = [
            np.mean(ecg_signal), np.std(ecg_signal), np.min(ecg_signal), np.max(ecg_signal),
            np.median(ecg_signal), np.percentile(ecg_signal, 25), np.percentile(ecg_signal, 75),
            np.sum(np.abs(ecg_signal)), np.sum(np.square(ecg_signal)),
            len(find_peaks(ecg_signal)[0]), len(find_peaks(-ecg_signal)[0]),
            # Добавьте другие статистические признаки по необходимости
        ]
        
        # 2. Признаки из Фурье-преобразования (50)
        fft_vals = np.abs(np.fft.fft(ecg_signal)[:50])
        features.extend(fft_vals.tolist())
        
        # 3. Вейвлет-признаки (50)
        # Здесь должна быть реализация вейвлет-преобразования
        # Временно заполняем нулями
        features.extend([0]*50)
        
        # 4. Скользящие статистики (241)
        window_size = 50
        for i in range(0, len(ecg_signal)-window_size, window_size//2):
            window = ecg_signal[i:i+window_size]
            features.extend([np.mean(window), np.std(window), np.max(window)-np.min(window)])
        
        # Дополняем до 361 признака нулями если необходимо
        if len(features) < 361:
            features.extend([0]*(361 - len(features)))
            
        return np.array(features[:361])  # Точно 361 признак
    
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")

def analyze_ecg(input_path, output_path):
    try:
        # 1. Загрузка данных
        ecg_signal = load_ecg_data(input_path)
        print(f"Loaded ECG signal with {len(ecg_signal)} samples")
        
        # 2. Извлечение признаков
        features = extract_features(ecg_signal)
        print(f"Extracted {len(features)} features")
        
        # 3. Загрузка моделей
        model_dir = os.path.join(os.path.dirname(__file__), 'models/models')
        scaler = load(os.path.join(model_dir, 'scaler.joblib'))
        model = load(os.path.join(model_dir, 'ensemble_model.joblib'))
        
        # 4. Масштабирование и предсказание
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        probas = model.predict_proba(features_scaled)[0]
        
        # 5. Формирование результата
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
        
        # 6. Визуализация
        plot_ecg(ecg_signal, os.path.join(os.path.dirname(output_path), result["ecg_image"]))
        
        # 7. Сохранение
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
            
        return result
    
    except Exception as e:
        return {"error": str(e), "details": "ECG analysis failed"}

def plot_ecg(signal, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title('ECG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    testInput = "ml\\models\\data\\1_Dangerous_VFL_VF\\full\\418_C_VFL_492s_full.csv"
    testOutput = "testJson.json"
    """if len(sys.argv) < 3:
        print("Usage: python ecg_analysis.py <input_file> <output_json_file>")
        sys.exit(1)"""
    
    #result = analyze_ecg(sys.argv[1], sys.argv[2])
    result = analyze_ecg(testInput, testOutput)
    print(json.dumps(result, indent=2))