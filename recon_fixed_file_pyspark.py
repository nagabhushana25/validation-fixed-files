from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace, lit, when, array, size

spark = SparkSession.builder.appName("FixedWidthValidation").getOrCreate()

# Copybook schema: (name, start, end, type)
copybook_schema = [
    ("CUSIP", 1, 9, 'STRING'),
    ("BROKERAGE_ACCT", 10, 21, 'STRING'),
    ("POSITION_ID", 22, 31, 'STRING'),
    ("PARTITION_ID", 32, 36, 'STRING')
]

# 🔹 define PK fields
primary_keys = ["CUSIP", "POSITION_ID"]

def parse_fixed_width(line, schema):
    values = []
    for name, start, end, ftype in schema:
        values.append(line[start-1:end].rstrip())
    return values

def load_fixed_width(path, schema):
    rdd = spark.sparkContext.textFile(path).map(lambda line: parse_fixed_width(line, schema))
    return rdd.toDF([f[0] for f in schema])

def normalize_spaces(df):
    return df.select([
        regexp_replace(trim(col(c)), " +", " ").alias(c) for c in df.columns
    ])

# Load both files
legacy_df = normalize_spaces(load_fixed_width("legacy_file.txt", copybook_schema))
modern_df = normalize_spaces(load_fixed_width("modern_file.txt", copybook_schema))

# 🔹 join on primary key(s)
join_cond = [legacy_df[k] == modern_df[k] for k in primary_keys]
joined = legacy_df.alias("l").join(modern_df.alias("m"), join_cond, "outer")

# Compare fields
mismatch_exprs = []
for field, _, _, _ in copybook_schema:
    mismatch_exprs.append(
        when(col(f"l.{field}") != col(f"m.{field}"), lit(field)).otherwise(lit(None))
    )

joined = joined.withColumn("mismatched_fields", array(*mismatch_exprs))
result = joined.withColumn("mismatch_count", size(col("mismatched_fields")))

# 🔹 save all results
result.write.mode("overwrite").csv("validation_results")

# 🔹 save only bad records
bad_records = result.filter(col("mismatch_count") > 0)
bad_records.write.mode("overwrite").csv("bad_records")
