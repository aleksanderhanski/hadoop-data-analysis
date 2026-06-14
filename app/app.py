import streamlit as st
from pyspark.sql import SparkSession

HDFS_PATH = "hdfs://namenode:9000/data/*.csv"


# Cache the SparkSession so Streamlit's rerun-on-every-interaction
# doesn't spin up a new one each time (that would be slow and eventually break).
@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("taxi-connection-check")
        .master("local[*]")
        .getOrCreate()
    )


st.title("NYC Taxi — HDFS connection check")

spark = get_spark()

try:
    df = spark.read.csv(HDFS_PATH, header=True)   # reads just the header → fast
    st.success("Connected to HDFS and read the dataset.")

    st.subheader("Columns")
    st.write(df.columns)

    st.subheader("Sample rows")
    st.dataframe(df.limit(10).toPandas())

    # Counting scans the whole 1.8 GB, so make it opt-in rather than on every rerun.
    if st.button("Count all rows (full scan — slow)"):
        with st.spinner("Counting…"):
            st.metric("Total rows", f"{df.count():,}")

except Exception as e:
    st.error("Could not read from HDFS — see details below.")
    st.exception(e)