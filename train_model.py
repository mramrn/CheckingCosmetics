import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Пути к файлам
DATA_PATH = "data/cosmetics.csv"
MODEL_PATH = "best_cosmetic_model.pkl"


def load_data(file_path: str) -> pd.DataFrame:
    """Загружает и очищает датасет"""
    df = pd.read_csv(file_path, sep=";", encoding="ISO-8859-1")
    df = df.drop(columns=["Product_name"], errors="ignore")  # Убираем ненужный столбец
    df = df.dropna()  # Удаляем пропущенные значения
    return df


def preprocess_data(df: pd.DataFrame):
    """Разделяет данные на признаки и целевую переменную, масштабирует их"""
    X = df.drop(columns=["comedogenicity"])
    y = df["comedogenicity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Балансировка классов с помощью SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_balanced), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Обучает XGBoost с балансировкой классов"""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 2, 4],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1, 2, 5],
        "gamma": [0, 0.1, 0.2]
    }

    model = XGBClassifier(eval_metric='logloss')

    grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", cv=4, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Оценивает модель на тестовой выборке"""
    y_pred = model.predict_proba(X_test)[:, 1]

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC на тестовых данных: {roc_auc:.4f}")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC на тестовых данных: {pr_auc:.4f}")


def save_model(model, file_path: str):
    """Сохраняет обученную модель в файл"""
    joblib.dump(model, file_path)


if __name__ == "__main__":
    # Загрузка и подготовка данных
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Обучение модели
    best_model = train_model(X_train, y_train)

    # Оценка модели
    evaluate_model(best_model, X_test, y_test)

    # Сохранение модели
    save_model(best_model, MODEL_PATH)


