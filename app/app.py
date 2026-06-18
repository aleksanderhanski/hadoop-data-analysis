import streamlit as st
from pyspark.sql import SparkSession, functions as F
import pandas as pd
import numpy as np
import pydeck as pdk

PARQUET_PATH = "hdfs://namenode:9000/parquet/yellow_taxi"
st.set_page_config(page_title="NYC Taxi Explorer", layout="wide")


@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("taxi-explorer").master("local[4]").getOrCreate()


@st.cache_resource
def load_df():
    return get_spark().read.parquet(PARQUET_PATH)


@st.cache_data
def available_months():
    rows = load_df().select("pickup_month").distinct().orderBy("pickup_month").collect()
    return [r["pickup_month"] for r in rows]


def filtered_df(months):
    df = load_df()
    return df.filter(F.col("pickup_month").isin(list(months))) if months else df


# ---------- queries ----------
@st.cache_data
def q_demand_by_hour(months):
    return (filtered_df(months).groupBy("pickup_hour")
            .agg(F.count("*").alias("trips"))
            .orderBy("pickup_hour").toPandas())


@st.cache_data
def q_fare_per_mile_by_hour(months):
    return (filtered_df(months)
            .groupBy("pickup_hour")
            .agg(F.round(F.sum("fare_amount") / F.sum("trip_distance"), 2)
                 .alias("fare_per_mile"))
            .orderBy("pickup_hour").toPandas())


@st.cache_data
def q_pickup_hotspots(months):
    return (filtered_df(months)
            .filter(F.col("pickup_lat").between(40.4, 41.0) &
                    F.col("pickup_lon").between(-74.3, -73.6))
            .withColumn("lat", F.round("pickup_lat", 3))
            .withColumn("lon", F.round("pickup_lon", 3))
            .groupBy("lat", "lon").count()
            .filter(F.col("count") >= 30)
            .orderBy(F.desc("count")).limit(5000).toPandas())


@st.cache_data
def q_dropoff_hotspots(months):
    return (filtered_df(months)
            .filter(F.col("dropoff_lat").between(40.4, 41.0) &
                    F.col("dropoff_lon").between(-74.3, -73.6))
            .withColumn("lat", F.round("dropoff_lat", 3))
            .withColumn("lon", F.round("dropoff_lon", 3))
            .groupBy("lat", "lon").count()
            .filter(F.col("count") >= 30)
            .orderBy(F.desc("count")).limit(5000).toPandas())


@st.cache_data
def q_tipping_by_hour(months):
    return (filtered_df(months)
            .filter(F.col("payment_type") == 1)          # card only - cash tips aren't recorded
            .withColumn("tip_pct", F.col("tip_amount") / F.col("fare_amount") * 100)
            .groupBy("pickup_hour")
            .agg(F.round(F.avg("tip_pct"), 1).alias("avg_tip_pct"))
            .orderBy("pickup_hour").toPandas())


@st.cache_data
def q_distance_vs_fare(months):
    return (filtered_df(months)
            .withColumn("miles", F.floor("trip_distance").cast("int"))
            .filter(F.col("miles") <= 30)
            .groupBy("miles")
            .agg(F.round(F.sum("fare_amount") / F.sum("trip_distance"), 2)
                 .alias("avg_fare_per_mile"),
                 F.count("*").alias("trips"))
            .orderBy("miles").toPandas())


@st.cache_data
def q_volume_by_weekday(months):
    return (filtered_df(months)
            .withColumn("weekday", F.date_format("pickup_ts", "EEEE"))
            .groupBy("weekday")
            .agg(F.count("*").alias("total_trips"),
                 F.countDistinct("pickup_date").alias("num_days"))
            .withColumn("avg_trips",
                        F.round(F.col("total_trips") / F.col("num_days")).cast("long"))
            .toPandas())


# graded map: brighter + bigger where busier
def graded_map(df):
    d = df.copy()
    c = d["count"].astype(float)
    # log scale, because counts are very skewed (a few huge cells, many small)
    lo, hi = np.log(c.min()), np.log(c.max())
    norm = ((np.log(c) - lo) / (hi - lo + 1e-9)).clip(0, 1)
    # dim blue (rare) -> bright yellow (common)
    d["color"] = [[int(40 + 215 * n), int(40 + 180 * n), int(140 * (1 - n) + 30), 200]
                  for n in norm]
    d["radius"] = 30 + 170 * norm
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=d,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=2,
        radius_max_pixels=22,
        opacity=0.8,
        pickable=True,
    )
    view = pdk.ViewState(latitude=40.75, longitude=-73.98, zoom=10)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             map_style=None, tooltip={"text": "{count} trips"}))


# ----------------------------- UI -----------------------------
st.title("🚕 NYC Taxi Explorer 🚕")
months = tuple(st.sidebar.multiselect("Filter by month",
                                       available_months(), default=available_months()))
view = st.sidebar.radio("Choose an analysis", [
    "Demand by hour", "Fare per mile by hour",
    "Pickup hotspots", "Dropoff hotspots", "Tipping by hour (card)",
    "Distance vs fare", "Trips by day of week",
])

if view == "Demand by hour":
    st.subheader("Trips by hour of day")
    d = q_demand_by_hour(months).rename(columns={"pickup_hour": "pickup hour"})
    st.bar_chart(d, x="pickup hour", y="trips")
    st.dataframe(d, use_container_width=True)

elif view == "Fare per mile by hour":
    st.subheader("Average fare per mile by hour of day")
    d = q_fare_per_mile_by_hour(months).rename(
        columns={"pickup_hour": "pickup hour", "fare_per_mile": "fare per mile"})
    st.line_chart(d, x="pickup hour", y="fare per mile")
    st.dataframe(d, use_container_width=True)

elif view == "Pickup hotspots":
    st.subheader("Busiest pickup locations")
    st.caption("Brighter and bigger = more pickups. Faint blue points are quieter spots.")
    d = q_pickup_hotspots(months)
    graded_map(d)
    st.dataframe(d, use_container_width=True)

elif view == "Dropoff hotspots":
    st.subheader("Busiest dropoff locations")
    st.caption("Brighter and bigger = more dropoffs. Faint blue points are quieter spots.")
    d = q_dropoff_hotspots(months)
    graded_map(d)
    st.dataframe(d, use_container_width=True)

elif view == "Tipping by hour (card)":
    st.subheader("Average tip % by hour")
    st.caption("Cash tips aren't recorded in the data, so only credit-card trips are counted.")
    d = q_tipping_by_hour(months).rename(
        columns={"pickup_hour": "pickup hour", "avg_tip_pct": "avg tip percent"})
    st.line_chart(d, x="pickup hour", y="avg tip percent")
    st.dataframe(d, use_container_width=True)

elif view == "Distance vs fare":
    st.subheader("Average fare per mile by trip distance")
    d = q_distance_vs_fare(months).rename(columns={"avg_fare_per_mile": "avg fare per mile"})
    st.line_chart(d, x="miles", y="avg fare per mile")
    st.dataframe(d, use_container_width=True)

else:
    st.subheader("Average trips per day of the week")
    d = q_volume_by_weekday(months)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    d["weekday"] = pd.Categorical(d["weekday"], categories=order, ordered=True)
    d = d.sort_values("weekday")
    d = d.rename(columns={"avg_trips": "avg trips"})
    st.bar_chart(d, x="weekday", y="avg trips")
    st.dataframe(d, use_container_width=True)