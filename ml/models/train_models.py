import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Создаем папку для моделей, если ее нет
os.makedirs('models', exist_ok=True)

# Соответствие меток классов и их названий
CLASS_NAMES = {
    1: "Dangerous_VFL_VF",
    2: "Special_Form_VTTdP",
    3: "Threatening_VT",
    4: "Potential_Dangerous",
    5: "Supraventricular",
    6: "Sinus_rhythm"
}

def load_and_preprocess_data(filepath):
    """Загрузка и предобработка данных ЭКГ из CSV"""
    # Загрузка данных
    data = pd.read_csv(filepath)
    
    # Проверка данных
    print("Первые 5 строк данных:")
    print(data.head())
    print("\nИнформация о данных:")
    print(data.info())
    print("\nРаспределение классов:")
    print(data.iloc[:, 0].value_counts().sort_index().rename(CLASS_NAMES))
    
    # 1. Удаление пропущенных значений
    data = data.dropna()
    
    # 2. Разделение на признаки и целевую переменную
    # Первый столбец - метка класса, остальные - признаки ЭКГ
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    # 3. Проверка баланса классов
    plot_class_distribution(y)
    
    # 4. Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 5. Разделение на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler

def plot_class_distribution(y):
    """Визуализация распределения классов"""
    class_counts = pd.Series(y).value_counts().sort_index()
    class_counts.index = class_counts.index.map(CLASS_NAMES)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Распределение классов в наборе данных')
    plt.xlabel('Класс')
    plt.ylabel('Количество образцов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/class_distribution.png')
    plt.close()

def train_models(X_train, y_train):
    """Обучение отдельных моделей и ансамбля"""
    print("\nОбучение моделей...")
    
    # 1. Decision Tree
    print("Обучение Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    # 2. K-Nearest Neighbors
    print("Обучение KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # 3. Support Vector Machine
    print("Обучение SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_train, y_train)
    
    # 4. Ансамбль моделей
    print("Обучение ансамбля...")
    ensemble = VotingClassifier(estimators=[
        ('dt', dt),
        ('knn', knn),
        ('svm', svm)
    ], voting='soft')
    ensemble.fit(X_train, y_train)
    
    return dt, knn, svm, ensemble

def evaluate_models(models, X_test, y_test):
    """Оценка качества моделей"""
    print("\nОценка моделей на тестовых данных:")
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'accuracy': acc, 'f1_score': f1}
        
        print(f"\n{name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES.values()))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=CLASS_NAMES.values(), 
                    yticklabels=CLASS_NAMES.values())
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'models/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def save_models(models, scaler):
    """Сохранение моделей и scaler"""
    print("\nСохранение моделей...")
    
    # Сохраняем модели с теми же ключами, которые использовались при создании
    joblib.dump(models['Decision Tree'], 'models/dt_model.joblib')
    joblib.dump(models['KNN'], 'models/knn_model.joblib')
    joblib.dump(models['SVM'], 'models/svm_model.joblib')
    joblib.dump(models['Ensemble'], 'models/ensemble_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Модели и scaler успешно сохранены в папку 'models'")
    
if __name__ == "__main__":
    # Укажите путь к вашему файлу с данными
    DATA_FILE = 'combined_data.csv'  # Используем объединенный файл
    
    # 1. Загрузка и предобработка данных
    print("Загрузка и предобработка данных...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATA_FILE)
    
    # 2. Обучение моделей
    dt, knn, svm, ensemble = train_models(X_train, y_train)
    
    # 3. Оценка моделей
    models = {
        'Decision Tree': dt,
        'KNN': knn,
        'SVM': svm,
        'Ensemble': ensemble
    }
    evaluation_results = evaluate_models(models, X_test, y_test)
    
    # 4. Сохранение моделей
    save_models(models, scaler)
    
    # Сохранение результатов оценки
    pd.DataFrame(evaluation_results).to_csv('models/evaluation_results.csv')
    print("\nОбучение и оценка моделей завершены успешно!")