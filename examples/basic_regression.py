"""
Example: Basic automated regression using MLPipe.
"""

from mlpipe import Pipeline


def main():
    # 1. Initialize automated pipeline
    pipeline = Pipeline(
        target="price",
        task="auto",       # Automatically detects Regression
        mode="balanced",   # Moderate tuning budget
        output_dir="./mlpipe_runs",
    )

    # 2. Fit pipeline on house prices data
    print("Fitting MLPipe regression pipeline...")
    result = pipeline.fit("demo_data/house_prices.csv")

    # 3. Inspect results
    print("\n" + "=" * 50)
    print("Regression Pipeline Completed!")
    print("=" * 50)
    print(f"Best Model:     {result.best_model}")
    print(f"Primary Metric: {result.primary_metric}")
    print(f"Best CV Score:  {result.best_cv_score:.4f}")
    print(f"Test Score:     {result.test_score:.4f}")
    print(f"Artifacts:      {result.artifacts_dir}")


if __name__ == "__main__":
    main()
