"""Curvas, probabilidades agregadas e importancias sobre validación/prueba."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from depi_ml.analysis.dataset import write_json
from depi_ml.analysis.reports import plotting
from depi_ml.evaluation.metrics import binary_metrics


def evaluate_partition(pipeline, frame, output, features, threshold=.5, min_group=30, importance_samples=2000, seed=42, timezone="America/Lima", explain=False):
    output.mkdir(parents=True, exist_ok=False)
    X, y = frame.loc[:, features], frame.target.to_numpy(dtype=int)
    probability = pipeline.predict_proba(X)[:, 1]
    metrics = binary_metrics(y, probability, threshold)
    write_json(output / "metrics.json", metrics)
    table = pd.DataFrame({"clinic_id": frame.clinic_id.to_numpy(),
                          "period": frame.appointment_at.dt.tz_convert(timezone).dt.strftime("%Y-%m").to_numpy(),
                          "target": y, "probability": probability})
    subgroups = []
    for group in ["clinic_id", "period"]:
        for key, part in table.groupby(group, dropna=False):
            if len(part) >= min_group:
                subgroups.append({"group": group, "value": str(key), **binary_metrics(part.target, part.probability, threshold)})
    write_json(output / "subgroups.json", subgroups)
    bins = np.linspace(0, 1, 21)
    hist = []
    for target in [0, 1]:
        counts, _ = np.histogram(probability[y == target], bins)
        for i, count in enumerate(counts):
            hist.append({"target": target, "lower": float(bins[i]), "upper": float(bins[i + 1]), "count": int(count)})
    pd.DataFrame(hist).to_csv(output / "probability_distribution.csv", index=False)
    plt = plotting()
    for name in ["roc", "precision_recall", "probabilities", "calibration"]:
        fig, ax = plt.subplots()
        if name == "roc":
            fpr, tpr, _ = roc_curve(y, probability)
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.set(xlabel="False positive rate", ylabel="True positive rate")
            pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(output / "roc.csv", index=False)
        elif name == "precision_recall":
            precision, recall, _ = precision_recall_curve(y, probability)
            ax.plot(recall, precision)
            ax.axhline(y.mean(), linestyle="--", color="grey")
            ax.set(xlabel="Recall", ylabel="Precision")
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(output / "precision_recall.csv", index=False)
        elif name == "probabilities":
            for target in [0, 1]:
                ax.hist(probability[y == target], bins=bins, alpha=.5, label=str(target))
            ax.set(xlabel="Probabilidad de candidato no-show", ylabel="Frecuencia")
            ax.legend()
        else:
            # No aplica calibración: solo diagnóstico de confiabilidad en bins suficientemente grandes.
            cells = []
            for i in range(10):
                mask = (probability >= i / 10) & ((probability < (i + 1) / 10) if i < 9 else (probability <= 1))
                if int(mask.sum()) >= min_group:
                    cells.append({"mean_probability": float(probability[mask].mean()), "observed_rate": float(y[mask].mean()), "rows": int(mask.sum())})
            pd.DataFrame(cells, columns=["mean_probability", "observed_rate", "rows"]).to_csv(output / "calibration.csv", index=False)
            if cells:
                ax.plot([r["mean_probability"] for r in cells], [r["observed_rate"] for r in cells], marker="o")
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.set(xlabel="Probabilidad media", ylabel="Frecuencia observada")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png")
        plt.close(fig)
    sample = frame.sample(n=min(importance_samples, len(frame)), random_state=seed).sort_index()
    if sample.target.nunique() == 2 and not pipeline.named_steps["model"].__class__.__name__.startswith("Dummy"):
        result = permutation_importance(pipeline, sample.loc[:, features], sample.target.astype(int),
                                        scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=1)
        pd.DataFrame({"variable": features, "mean_ap_decrease": result.importances_mean,
                      "std_ap_decrease": result.importances_std}).to_csv(output / "permutation_importance.csv", index=False)
        write_json(output / "importance_metadata.json", {"scoring": "average_precision", "samples": len(sample), "repeats": 5,
                   "seed": seed, "warning": "Correlación entre variables puede reducir/redistribuir importancias; no son causales."})
        if explain:
            shap_sample = sample.sample(n=min(200, len(sample)), random_state=seed)
            shap_importance(pipeline, shap_sample.loc[:, features], output)
    else:
        write_json(output / "importance_metadata.json", {"status": "skipped", "reason": "Baseline trivial o muestra con una sola clase."})
    return metrics


def shap_importance(pipeline, X, output):
    import shap
    transformed = pipeline[:-1].transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    estimator = pipeline.named_steps["model"]
    values = np.asarray(shap.TreeExplainer(estimator).shap_values(transformed))
    if values.ndim == 3:
        values = values[:, :, 1]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    # Solo promedios absolutos, no explicaciones/valores de citas individuales.
    pd.DataFrame({"encoded_feature": names, "mean_absolute_shap": np.abs(values).mean(axis=0)}).to_csv(output / "shap_importance.csv", index=False)
    write_json(output / "shap_metadata.json", {"samples": len(X), "aggregation": "mean absolute", "output_space": "probability for RandomForest; raw margin for XGBoost",
               "warning": "Importancias no comparables directamente entre espacios/modelos; no causales."})
