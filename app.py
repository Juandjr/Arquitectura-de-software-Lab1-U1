import math
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.model_selection import train_test_split

from filtros import (
    LoadDatasetFilter,
    ValidateDatasetFilter,
    PrepareDataFilter,
    TrainModelFilter,
    EvaluateModelFilter
)

# Configuración de Flask
app = Flask(
    __name__,
    static_folder="public",
    static_url_path="/public"
)

app.secret_key = "cambia-esta-clave-en-produccion"

# Configuración del modelo
MODEL_PATH = Path("models/weather_model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET = "rain"


def get_prediction_fields(saved):
    features = saved["features"]
    numeric_features = saved.get("numeric_features")
    categorical_options = saved.get("categorical_options", {})

    if numeric_features is None:
        numeric_features = []
        preprocessor = saved["model"].named_steps.get("preprocessor")
        if preprocessor is not None:
            for name, _, columns in preprocessor.transformers_:
                if name == "numeric":
                    numeric_features.extend(columns)

    numeric_features = set(numeric_features)
    return [
        {
            "name": feature,
            "numeric": feature in numeric_features,
            "options": categorical_options.get(feature, [])
        }
        for feature in features
    ]


def get_model_context():
    if not MODEL_PATH.exists():
        return None, None, None

    saved = joblib.load(MODEL_PATH)
    return saved, get_prediction_fields(saved), saved.get("results")


def render_page(active_tab, prediction_result=None):
    feature_fields = None
    results = None
    if MODEL_PATH.exists():
        try:
            _, feature_fields, results = get_model_context()
        except Exception as exc:
            app.logger.exception("Error al cargar el modelo guardado")
            flash(f"No se pudo cargar el modelo guardado: {exc}")

    return render_template(
        "index.html",
        active_tab=active_tab,
        results=results,
        feature_fields=feature_fields,
        prediction_result=prediction_result
    )


@app.route("/", methods=["GET"])
def index():
    active_tab = request.args.get("tab", "upload")
    if active_tab not in {"upload", "results", "predict"}:
        active_tab = "upload"
    return render_page(active_tab)

@app.route("/train", methods=["POST"])
def train():
    uploaded = request.files.get("dataset")

    if uploaded is None or not uploaded.filename:
        flash("Selecciona un archivo CSV.")
        return redirect(url_for("index", tab="upload"))

    if not uploaded.filename.lower().endswith(".csv"):
        flash("El archivo debe tener extensión .csv.")
        return redirect(url_for("index", tab="upload"))

    try:
        # Filtro 1: carga del dataset
        raw = LoadDatasetFilter().run(uploaded.stream)

        # Filtro 2: validación
        validated = ValidateDatasetFilter(TARGET).run(raw)

        # Filtro 3: preparación de los datos
        X, y, info = PrepareDataFilter(TARGET).run(validated)

        # Separación de datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Filtro 4: entrenamiento
        model = TrainModelFilter().run(
            X_train,
            y_train,
            info
        )

        # Filtro 5: evaluación
        metrics = EvaluateModelFilter().run(
            model,
            X_test,
            y_test
        )

        # Guardar modelo y configuración
        numeric_features = X.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        categorical_options = {
            feature: list(dict.fromkeys(
                str(value) for value in X[feature].dropna()
            ))
            for feature in X.columns
            if feature not in numeric_features
        }
        results = {
            "rows": len(validated),
            "features": len(X.columns),
            "target": info["target"],
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
            "metrics": metrics,
            "model_path": str(MODEL_PATH)
        }

        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "target": info["target"],
                "numeric_features": numeric_features,
                "categorical_options": categorical_options,
                "results": results
            },
            MODEL_PATH
        )
        return redirect(url_for("index", tab="results"))

    except Exception as exc:
        app.logger.exception("Error durante el entrenamiento")
        flash(f"No se pudo procesar el dataset: {exc}")
        return redirect(url_for("index", tab="upload"))

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists():
        flash("Primero entrena un modelo subiendo un dataset.")
        return redirect(url_for("index", tab="predict"))

    try:
        saved = joblib.load(MODEL_PATH)

        features = saved["features"]
        model = saved["model"]
        feature_fields = get_prediction_fields(saved)

        values = {}

        for field in feature_fields:
            feature = field["name"]
            raw_value = request.form.get(feature, "").strip()

            if raw_value == "":
                raise ValueError(
                    f"El campo '{feature}' es obligatorio."
                )

            if field["numeric"]:
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"El campo '{feature}' debe ser numérico."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"El campo '{feature}' debe ser un número finito."
                    )
                values[feature] = value
            else:
                values[feature] = raw_value

        sample = pd.DataFrame(
            [values],
            columns=features
        )

        prediction = model.predict(sample)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            predicted_index = list(model.classes_).index(prediction)
            probability = round(
                float(probabilities[predicted_index]) * 100,
                2
            )

        prediction_result = {
            "label": str(prediction),
            "probability": probability
        }

        return render_template(
            "index.html",
            active_tab="predict",
            results=saved.get("results"),
            feature_fields=feature_fields,
            prediction_result=prediction_result
        )

    except Exception as exc:
        app.logger.exception("Error durante la predicción")
        flash(f"No se pudo generar la predicción: {exc}")
        return redirect(url_for("index", tab="predict"))

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
