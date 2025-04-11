from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, MinHashLSH
from pyspark.sql.functions import col, lit, concat_ws, expr, regexp_replace, lower, trim, when
from configs import db_credentials, sources, canonicals, mappings, pivot_cols
#import findspark
#findspark.init()

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PostgresJDBC") \
    .config("spark.jars", "./lib/postgresql-42.7.5.jar") \
    .getOrCreate()

# PostgreSQL connection properties
jdbc_url = f"jdbc:postgresql://{db_credentials['host']}:{db_credentials['port']}/{db_credentials['database']}"
db_properties = {
    "user": db_credentials['username'],
    "password": db_credentials['password'],
    "driver": "org.postgresql.Driver"
}

# Sources
sources = sources

# Columns to pivot and 1 hot encode
pivot_cols = pivot_cols

# Master map of all canonicals, with binary for whether or not to clean
canonicals = canonicals

# Define mappings of canonical field: raw field
mappings = mappings

# Read data from PostgreSQL database
raw_tables = {}
for source in sources:
    if source != "device_events":
        raw_tables[source] = spark.read.jdbc(url=jdbc_url, table=source, properties=db_properties)
    else:
        # join events back onto devices to create device_events table
        devices = spark.read.jdbc(url=jdbc_url, table="device", properties=db_properties)
        events = spark.read.jdbc(url=jdbc_url, table="device_event", properties=db_properties)
        raw_tables[source] = devices.join(events.select("event_key", "event_location", "event_type", "remedial_action", "date_of_event", "event_id"), "event_id", "left")

# Map to canonicals
def build_schema(df, mapping, canonicals):
    canonical_list = canonicals.keys()
    raw = [raw for canonical, raw in mapping.items()]
    df = df.select(raw) # select columns that will get mapped. helps avoid duplicate cols and other schema conflict issues later
    for canonical, raw in mapping.items():
        df = df.withColumnRenamed(raw, canonical) # map raw to canonical
        df = df.withColumn(canonical, concat_ws(",",col(canonical))) # convert arrays to strings
        if canonicals.get(canonical) == 1:
            df = clean_canon(df, canonical)
    df = df.select([col if col in df.columns else lit(None).cast("string").alias(col) for col in canonical_list]) # ensure all canonical fields and only canonical fields are present
    return df

def clean_canon(df, canonical):
    # replace symbols with whitespace; standardize whitespace; trim lower; get rid of any lingering spaces
    df = df.withColumn(canonical, regexp_replace(col(canonical), r"[^a-zA-Z0-9]", " ")) \
           .withColumn(canonical, regexp_replace(col(canonical), r"\s{2,}", " ")) \
           .withColumn(canonical, lower(trim(col(canonical)))) \
           .withColumn(canonical, when(col(canonical) == "", None).otherwise(col(canonical)))
    return df

# Harmonize schemas
tables_to_union = []
for source in sources:
    print(source)
    mapping = mappings.get(source)
    df = raw_tables.get(source)
    tables_to_union.append(build_schema(df.withColumn("source", lit(source)), mapping, canonicals).withColumn("pkey", expr("uuid()")))

# Merge harmonized tables into a master df
master = tables_to_union[0]  # initialize merge with first df
for table in tables_to_union[1:]:
    master = master.union(table).na.replace("", None)

# Pivot and 1 hot encode
for column in pivot_cols:
    values = master.select(column).distinct().filter(col(column) != None).rdd.flatMap(lambda x: x).collect()
    master = master \
        .groupBy(master.columns) \
        .pivot(column) \
        .agg(lit(1)) \
        .fillna(0) \
        .drop("null")

master = master.select([col(column).alias(column.replace(' ', '_').lower()) for column in master.columns])
master.write.jdbc(url=jdbc_url, table="integrated", mode="overwrite", properties=db_properties)