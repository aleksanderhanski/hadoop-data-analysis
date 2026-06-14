"""One-time CSV -> Parquet conversion. Run inside the app container:
    docker compose exec app python convert_to_parquet.py
"""
from pyspark.sql import SparkSession, functions as F

CSV_PATH = "hdfs://namenode:9000/data/*.csv"
PARQUET_PATH = "hdfs://namenode:9000/parquet/yellow_taxi"

spark = (
    SparkSession.builder
    .appName("csv-to-parquet")
    .master("local[*]")
    .getOrCreate()
)

# Read as strings (no inferSchema pass), then cast explicitly - faster and predictable
raw = spark.read.csv(CSV_PATH, header=True)

clean = (
    raw.select(
        F.col("tpep_pickup_datetime").cast("timestamp").alias("pickup_ts"),
        F.col("tpep_dropoff_datetime").cast("timestamp").alias("dropoff_ts"),
        F.col("passenger_count").cast("int").alias("passenger_count"),
        F.col("trip_distance").cast("double").alias("trip_distance"),
        F.col("pickup_longitude").cast("double").alias("pickup_lon"),
        F.col("pickup_latitude").cast("double").alias("pickup_lat"),
        F.col("payment_type").cast("int").alias("payment_type"),
        F.col("fare_amount").cast("double").alias("fare_amount"),
        F.col("tip_amount").cast("double").alias("tip_amount"),
        F.col("total_amount").cast("double").alias("total_amount"),
    )
    # basic cleaning
    .filter(F.col("pickup_ts").isNotNull())
    .filter(F.col("fare_amount") > 0)
    .filter(F.col("trip_distance") > 0)
    .filter(F.col("trip_distance") < 100)
    .filter(F.col("passenger_count") > 0)
    # derived columns
    .withColumn("pickup_hour", F.hour("pickup_ts"))
    .withColumn("pickup_date", F.to_date("pickup_ts"))
    .withColumn("pickup_month", F.date_format("pickup_ts", "yyyy-MM"))
)

clean.write.mode("overwrite").parquet(PARQUET_PATH)
print(f"Wrote Parquet to {PARQUET_PATH}")
spark.stop()