"""
Example: Loading a trained MLPipe pipeline and generating predictions.
"""

from mlpipe import Pipeline


def main():
    # 1. Train a quick model first or load an existing one
    pipeline = Pipeline(
        target="churn",
        mode="fast",
    )
    pipeline.fit("demo_data/customer_churn.csv")

    # 2. Save pipeline
    model_path = pipeline.save("./trained_models/churn_fast")

    # 3. Load pipeline in a clean instance
    loaded_pipeline = Pipeline.load(model_path)

    # 4. Generate predictions on new data
    predictions = loaded_pipeline.predict("demo_data/customer_churn.csv")

    print(f"Generated {len(predictions)} predictions.")
    print("Sample predictions:", predictions[:5])


if __name__ == "__main__":
    main()
