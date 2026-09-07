"""
Example: Basic automated classification using MLPipe.
"""

from mlpipe import Pipeline


def main():
    # 1. Initialize automated pipeline
    pipeline = Pipeline(
        target="churn",
        task="auto",       # Automatically detects Binary Classification
        mode="balanced",   # Moderate tuning budget
        output_dir="./mlpipe_runs",
    )

    # 2. Fit pipeline on customer churn data
    print("Fitting MLPipe classification pipeline...")
    result = pipeline.fit("demo_data/customer_churn.csv")

    # 3. Inspect results
    print("\n" + "=" * 50)
    print("Classification Pipeline Completed!")
    print("=" * 50)
    print(f"Best Model:     {result.best_model}")
    print(f"Primary Metric: {result.primary_metric}")
    print(f"Best CV Score:  {result.best_cv_score:.4f}")
    print(f"Test Score:     {result.test_score:.4f}")
    print(f"Artifacts:      {result.artifacts_dir}")

    # 4. Save pipeline
    saved_path = pipeline.save("./trained_models/churn_pipeline")
    print(f"Saved pipeline to: {saved_path}")


if __name__ == "__main__":
    main()
