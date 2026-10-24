"""Spark ML anomaly detection for weekly behavioral feature vectors."""

from __future__ import annotations

from math import sqrt

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession


ML_FEATURES = [
    "purchase_record_count",
    "survey_response_diversity",
    "unique_product_code_count",
    "unique_category_count",
    "unique_shipping_state_count",
    "average_quantity",
    "quantity_stddev",
    "average_purchase_price_per_unit",
    "purchase_price_per_unit_stddev",
    "missing_value_rate",
    "duplicate_rate",
]


def run_behavioral_anomaly_model(spark: SparkSession, weekly_records: list[dict]) -> dict:
    """Train a two-cluster KMeans model and flag weeks far from their cluster center."""
    rows = []
    for record in weekly_records:
        comparison = record.get("feature_comparison", {})
        if all(feature in comparison and comparison[feature].get("current") is not None for feature in ML_FEATURES):
            rows.append({"week": record["week"], **{feature: float(comparison[feature]["current"]) for feature in ML_FEATURES}})

    if len(rows) < 8:
        raise ValueError("At least eight complete weekly observations are required for the ML model.")

    source = spark.createDataFrame(rows)
    assembled = VectorAssembler(inputCols=ML_FEATURES, outputCol="raw_features").transform(source)
    scaled = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(assembled).transform(assembled)
    model = KMeans(k=2, seed=42, maxIter=50, featuresCol="features", predictionCol="cluster").fit(scaled)
    predictions = model.transform(scaled).select("week", "cluster", "features").collect()
    centers = model.clusterCenters()

    scored = []
    for row in predictions:
        vector = row["features"]
        center = centers[row["cluster"]]
        distances = [abs(float(vector[index]) - float(center[index])) for index in range(len(ML_FEATURES))]
        distance = sqrt(sum(value * value for value in distances))
        top_feature = ML_FEATURES[distances.index(max(distances))]
        scored.append({
            "Week": row["week"],
            "Cluster": int(row["cluster"]),
            "Anomaly Score": round(distance, 4),
            "Most Influential Feature": top_feature,
        })

    scores = sorted(item["Anomaly Score"] for item in scored)
    threshold = scores[min(len(scores) - 1, max(0, int(len(scores) * 0.95)))]
    for item in scored:
        item["ML Result"] = "ANOMALOUS" if item["Anomaly Score"] >= threshold else "NORMAL"

    scored.sort(key=lambda item: item["Week"])
    return {
        "rows": scored,
        "threshold": round(threshold, 4),
        "anomaly_count": sum(item["ML Result"] == "ANOMALOUS" for item in scored),
        "feature_count": len(ML_FEATURES),
        "algorithm": "KMeans (k=2) with standardized Spark ML features",
    }
