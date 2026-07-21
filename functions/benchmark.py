import pandas as pd

from functions.metrics import calculate_metrics


def benchmark_model(model,
                    train_dl,
                    val_dl,
                    labels,
                    n_runs=10,
                    patience=15
                    ):

    all_metrics = []

    for run in range(n_runs):
        print(f"{model.name}: {run+1}/{n_runs}")

        
        model.reset_weights()

        model.retrain(
            train_dl,
            val_dl,
            patience=patience)
            

        prediction_df = model.predict_dl(val_dl)
        prediction_df['label'] = labels


        metrics = calculate_metrics(prediction_df)
        metrics["run"] = run + 1

        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)