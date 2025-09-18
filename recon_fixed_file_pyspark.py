from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace, lit, when, array, size

spark = SparkSession.builder.appName("FixedWidthValidation").getOrCreate()

# Example copybook schema (start and end positions are 1-based, inclusive)
copybook_schema = [
    ("CUSIP", 1, 9, 'STRING'),
    ("BROKERAGE_ACCT", 10, 21, 'STRING'),
    ("POSITION_ID", 22, 31, 'STRING'),
    ("PARTITION_ID", 32, 36, 'STRING')
]

# Function to slice fixed-width fields
def parse_fixed_width(line, schema):
    values = []
    for name, start, end, ftype in schema:
        # convert from 1-based inclusive to Python slice [start-1:end]
        values.append(line[start-1:end].rstrip())
    return values

# Load legacy file
legacy_rdd = spark.sparkContext.textFile("legacy_file.txt").map(
    lambda line: parse_fixed_width(line, copybook_schema)
)
legacy_df = legacy_rdd.toDF([f[0] for f in copybook_schema])

# Load modern file
modern_rdd = spark.sparkContext.textFile("modern_file.txt").map(
    lambda line: parse_fixed_width(line, copybook_schema)
)
modern_df = modern_rdd.toDF([f[0] for f in copybook_schema])

# Normalize spaces (trim + collapse multiple spaces)
def normalize_spaces(df):
    return df.select([
        regexp_replace(trim(col(c)), " +", " ").alias(c) for c in df.columns
    ])

legacy_df = normalize_spaces(legacy_df)
modern_df = normalize_spaces(modern_df)

# Add row_id for joining line-by-line
legacy_df = legacy_df.rdd.zipWithIndex().map(
    lambda x: (*x[0], x[1])
).toDF([*legacy_df.columns, "row_id"])

modern_df = modern_df.rdd.zipWithIndex().map(
    lambda x: (*x[0], x[1])
).toDF([*modern_df.columns, "row_id"])

# Join datasets on row_id
joined = legacy_df.alias("l").join(modern_df.alias("m"), "row_id")

# Compare fields and collect mismatches
mismatch_exprs = []
for field, _, _, _ in copybook_schema:
    mismatch_exprs.append(
        when(col(f"l.{field}") != col(f"m.{field}"), lit(field)).otherwise(lit(None))
    )

joined = joined.withColumn("mismatched_fields", array(*mismatch_exprs))
result = joined.withColumn("mismatch_count", size(col("mismatched_fields")))

# Save results (bad + good records)
result.write.mode("overwrite").csv("validation_results")
