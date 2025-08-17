# Databricks notebook source
# MAGIC %md
# MAGIC ## Create spark session

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from shapely.geometry import Point, shape
from pyspark.sql import functions as F
spark = SparkSession.builder.getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading data set CSV and save as parquet

# COMMAND ----------

Taxi_df = spark.read.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/FileStore/tables/sample.csv")

#Sometimes when compute is terminated the hive table didnot work correctly so we needed to delete the path itself from the DBFS
dbutils.fs.rm("dbfs:/user/hive/warehouse/nyctaxi", recurse=True)

#Data is saved in parquet format as it is faster in terms of I/O than CSV when continuing procesing the data in the pipline
Taxi_df.write.format('parquet').mode("overwrite").saveAsTable("NYCTaxi")


# COMMAND ----------

# MAGIC %md
# MAGIC ## processing dataset
# MAGIC 1- selecting needed columns only to decrease the size of data need to be processed.  
# MAGIC 2- applying filter early in pipeline to decrease the size of data need to be processed.  
# MAGIC 3- applying quality checks on non logical durations.

# COMMAND ----------

taxi_df_filtered = spark.read.table("NYCTaxi")

# Select relevant columns
taxi_df_filtered = taxi_df_filtered.select(
    col("pickup_datetime").alias("pickup_time"),
    col("dropoff_datetime").alias("dropoff_time"),
    col("pickup_longitude"),
    col("pickup_latitude"),
    col("dropoff_longitude"),
    col("dropoff_latitude"),
    col("hack_license"),  # Unique identifier for taxi
    col("medallion")      # Taxi medallion number
).withColumn("pickup_time", col("pickup_time").cast("timestamp"))\
  .withColumn("dropoff_time", col("dropoff_time").cast("timestamp"))\
    .withColumn("duration",(col("dropoff_time").cast("long") - col("pickup_time").cast("long"))/60)

#taxi_df_filtered.count()
#taxi_df_filtered.select(col("duration")).show()



# COMMAND ----------

taxi_df_filtered = taxi_df_filtered.filter(col("pickup_time") <= col("dropoff_time"))\
  .filter((col("duration") > 0) & (col("duration") <= 240)) #remove trips with time more than 4 hrs.

taxi_df_filtered.count()

dbutils.fs.rm("dbfs:/user/hive/warehouse/nyctaxi_filtered", recurse=True)

taxi_df_filtered.write.format('parquet').mode("overwrite").saveAsTable("NYCTaxi_filtered")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from NYCTaxi_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading GeoJson 

# COMMAND ----------

#used multline to ensure right reading of data as records are written on multiple lines

GeoJson_DF = spark.read.format("json")\
.option("header","true")\
.option("inferSchema","true")\
.option("multiline","true")\
.load("dbfs:/FileStore/tables/nyc_boroughs.geojson")

GeoJson_DF.show()

# COMMAND ----------

from shapely.geometry import Point, Polygon
from pyspark.sql.functions import col

# Extract polygons and boroughs from GeoJSON
#Since the features column is an array of objectsflatMap unpacks this array into individual elements.

geojson_data = GeoJson_DF.select("features")\
.rdd.flatMap(lambda row: row[0])\
  .collect() #Since data is not large we can use collect otherwise it is costly in large data set

#print(geojson_data[0])
#print(geojson_data[0].geometry)
#print(geojson_data[0].borough)

# Convert to a list of tuples (borough, polygon)
borough_polygons = [
    (feature.properties.borough, Polygon(feature.geometry.coordinates[0]))
    for feature in geojson_data
]

# Broadcast the polygons for efficient lookup
broadcast_polygons = spark.sparkContext.broadcast(borough_polygons)


# COMMAND ----------

print(borough_polygons)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating UDF to assign borough for each trip

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# UDF to find the borough for a given latitude and longitude
def find_borough(lat, lon):
    point = Point(lon, lat)  # Create a Shapely Point (Note: longitude comes first in Point)
    for borough, polygon in broadcast_polygons.value:
        if polygon.contains(point):  # Check if the point is inside the polygon
            return borough
    return None  # Return None if no matching polygon is found

# Register the UDF
find_borough_udf = udf(find_borough, StringType())


# COMMAND ----------

taxi_df_filtered = spark.read.table("NYCTaxi_filtered")

# COMMAND ----------

# Apply the UDF
result_df = taxi_df_filtered.withColumn("Pickup_borough", find_borough_udf(col("pickup_latitude"), col("pickup_longitude")))\
.withColumn("dropoff_borough", find_borough_udf(col("dropoff_latitude"), col("dropoff_longitude")))

# Show the result
result_df.show(truncate=False)

dbutils.fs.rm("dbfs:/user/hive/warehouse/nyctaxi_borough", recurse=True)

result_df.write.format('parquet').mode("overwrite").saveAsTable("NYCTaxi_Borough")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from NYCTaxi_Borough

# COMMAND ----------

taxi_data = spark.read.table("NYCTaxi_Borough")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculating AVG Idle Time

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, unix_timestamp, when, lag, sum as spark_sum
# Window partitioned by taxi and ordered by pickup time
window_spec = Window.partitionBy("medallion").orderBy("pickup_time")

# Calculate idle time as the time difference between a dropoff and the next pickup
taxi_data = taxi_data.withColumn("previous_dropoff", lag("dropoff_time").over(window_spec)) \
                     .withColumn("idle_time", (col("pickup_time").cast("long") - col("previous_dropoff").cast("long")) / 60)  # Idle time in minutes

#taxi_data.select(col("idle_time")).show()

# Filter trips with idle times exceeding 4 hours (new session starts)
taxi_data = taxi_data.withColumn("is_new_session", when(col("idle_time") > 240, 1).otherwise(0))


# COMMAND ----------

taxi_data.createOrReplaceTempView("Idle_Time")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from Idle_Time

# COMMAND ----------

# Group by dropoff borough to calculate average idle time
average_idle_time = taxi_data.groupBy("dropoff_borough") \
                             .agg(spark_sum("idle_time").alias("total_idle_time"),
                                  spark_sum("is_new_session").alias("total_trips")) \
                             .withColumn("avg_idle_time", col("total_idle_time") / col("total_trips"))

average_idle_time.show()


# COMMAND ----------

# Count trips within and between boroughs
within_borough_trips = taxi_data.filter(col("pickup_borough") == col("dropoff_borough")) \
                                .groupBy("pickup_borough") \
                                .count() \
                                .withColumnRenamed("count", "within_borough_trips")

between_borough_trips = taxi_data.filter(col("pickup_borough") != col("dropoff_borough")) \
                                 .groupBy("pickup_borough", "dropoff_borough") \
                                 .count() \
                                 .withColumnRenamed("count", "between_borough_trips")

within_borough_trips.show()
between_borough_trips.show()

