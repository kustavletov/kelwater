import json
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_training_data() -> pd.DataFrame:
    logging.info("Создание набора данных для обучения моделей.")
    np.random.seed(42)
    num_samples = 1000
    data = {
        "avg_temp": np.random.normal(loc=5, scale=10, size=num_samples),
        "snowDepth": np.random.normal(loc=30, scale=15, size=num_samples),
        "groundWaterLevel": np.random.normal(loc=1.5, scale=0.5, size=num_samples),
        "soilMoisture": np.random.uniform(0.3, 0.9, size=num_samples),
        "total_rainfall_recent": np.random.normal(loc=50, scale=20, size=num_samples),
        "spring_melt_index_scaled": np.random.uniform(50, 100, size=num_samples),
    }
    df = pd.DataFrame(data)
    df['flood_occurred'] = (
        (df['avg_temp'] > 0) &
        (df['snowDepth'] > 20) &
        (df['groundWaterLevel'] > 1.0) &
        (df['soilMoisture'] > 0.6) &
        (df['total_rainfall_recent'] > 30) &
        (df['spring_melt_index_scaled'] > 60)
    ).astype(int)
    logging.info("Набор данных создан.")
    return df

def preprocess_training_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series, StandardScaler):
    logging.info("Начало предобработки данных.")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = StandardScaler()
    numerical_features = ['avg_temp', 'snowDepth', 'groundWaterLevel', 'soilMoisture', 'total_rainfall_recent', 'spring_melt_index_scaled']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    logging.info("Предобработка завершена.")
    return df[numerical_features], df['flood_occurred'], scaler

def train_and_save_models(X: pd.DataFrame, y: pd.Series):
    logging.info("Начало обучения моделей.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        logging.info(f"Обучение модели: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        logging.info(f"Модель {name} обучена. ROC AUC: {auc:.4f}")
    for name, model in trained_models.items():
        joblib.dump(model, f"{name}.joblib")
        logging.info(f"Модель {name} сохранена.")
    logging.info("Все модели обучены и сохранены.")
    return trained_models

def load_models(model_names: list) -> dict:
    logging.info("Загрузка моделей.")
    loaded_models = {}
    for name in model_names:
        model = joblib.load(f"{name}.joblib")
        loaded_models[name] = model
        logging.info(f"Модель {name} загружена.")
    return loaded_models

def load_input_data(json_input_path: str) -> pd.DataFrame:
    logging.info("Загрузка данных.")
    with open(json_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    logging.info("Данные успешно загружены.")
    return df

def preprocess_input_data(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    logging.info("Начало предобработки входных данных.")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    if 'averageTemperatures' in df.columns:
        df['avg_temp'] = df['averageTemperatures'].apply(lambda x: np.mean(x) if isinstance(x, list) else 0)
    else:
        df['avg_temp'] = 0
    if 'springMeltIndex' in df.columns:
        df['spring_melt_index_scaled'] = df['springMeltIndex'] * 100
    else:
        df['spring_melt_index_scaled'] = 0
    if 'rainfallHistory' in df.columns:
        df['total_rainfall_recent'] = df['rainfallHistory'].apply(lambda x: np.sum(x) if isinstance(x, list) else 0)
    else:
        df['total_rainfall_recent'] = 0
    numerical_features = ['avg_temp', 'snowDepth', 'groundWaterLevel', 'soilMoisture', 'total_rainfall_recent', 'spring_melt_index_scaled']
    df[numerical_features] = scaler.transform(df[numerical_features])
    logging.info("Предобработка входных данных завершена.")
    return df[numerical_features]

def ensemble_predictions(models: dict, X: pd.DataFrame) -> float:
    logging.info("Ансамблирование предсказаний.")
    predictions = [model.predict_proba(X)[:, 1] for model in models.values()]
    ensemble_prob = np.mean(predictions)
    logging.info(f"Ансамблированная вероятность: {ensemble_prob:.4f}")
    return ensemble_prob

def monte_carlo_simulation(input_features: np.ndarray, scaler: StandardScaler, n_simulations: int = 1000) -> float:
    logging.info("Имитация Монте-Карло.")
    simulations = np.random.normal(loc=input_features, scale=0.1, size=(n_simulations, len(input_features)))
    simulations = np.clip(simulations, -3, 3)
    simulated_probs = [sigmoid(sim.sum() / len(sim)) for sim in simulations]
    monte_carlo_prob = np.mean(np.array(simulated_probs) > 0.6)
    logging.info(f"Результат Монте-Карло: {monte_carlo_prob:.4f}")
    return monte_carlo_prob

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_final_probability(ensemble_prob: float, monte_carlo_prob: float) -> float:
    logging.info("Вычисление итоговой вероятности.")
    final_prob = (ensemble_prob + monte_carlo_prob) / 2
    final_prob_percent = final_prob * 100
    final_prob_percent = min(max(final_prob_percent, 0), 100)
    logging.info(f"Итоговая вероятность: {final_prob_percent:.2f}%")
    return final_prob_percent

def main_training_phase():
    df_train = load_training_data()
    X, y, scaler = preprocess_training_data(df_train)
    train_and_save_models(X, y)
    joblib.dump(scaler, "scaler.joblib")

def main_prediction_phase(json_input_path: str):
    model_names = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
    models = load_models(model_names)
    scaler = joblib.load("scaler.joblib")
    df_input = load_input_data(json_input_path)
    X_input = preprocess_input_data(df_input, scaler)
    ensemble_prob = ensemble_predictions(models, X_input)
    input_features = X_input.values[0]
    monte_carlo_prob = monte_carlo_simulation(input_features, scaler)
    final_probability = calculate_final_probability(ensemble_prob, monte_carlo_prob)
    print(f"Вероятность затопления территории: {final_probability:.2f}%")
    with open(json_input_path, 'r', encoding='utf-8') as f:
        original_json = json.load(f)
    print("\nВходной JSON-файл:")
    print(json.dumps(original_json, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 2:
        print("Использование:")
        print("  Для обучения моделей: python app.py train")
        print("  Для предсказания: python app.py predict <path_to_json_input>")
        sys.exit(1)
    mode = sys.argv[1].lower()
    if mode == "train":
        main_training_phase()
    elif mode == "predict":
        if len(sys.argv) != 3:
            print("Использование для предсказания:")
            print("  python app.py predict <path_to_json_input>")
            sys.exit(1)
        json_input_path = sys.argv[2]
        if not os.path.exists(json_input_path):
            print(f"Файл {json_input_path} не существует.")
            sys.exit(1)
        main_prediction_phase(json_input_path)
    else:
        print("Неверный режим. Используйте 'train' или 'predict'.")
        sys.exit(1)
