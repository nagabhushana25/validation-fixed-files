from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace, lit, when, array, size

spark = SparkSession.builder.appName("FixedWidthValidation").getOrCreate()

# Example: copybook fields extracted manually or with parser
copybook_schema = [
    ("CUSIP", 9),
    ("BROKERAGE_ACCT", 12),
    ("POSITION_ID", 10),
    ("PARTITION_ID", 5)
]

# Function to slice fixed-width fields
def parse_fixed_width(line, schema):
    values = []
    pos = 0
    for name, length in schema:
        values.append(line[pos:pos+length])
        pos += length
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

# Normalize spaces
def normalize_spaces(df):
    return df.select([
        regexp_replace(trim(col(c)), " +", " ").alias(c) for c in df.columns
    ])

legacy_df = normalize_spaces(legacy_df)
modern_df = normalize_spaces(modern_df)

# Join by record index (assuming same order)
legacy_df = legacy_df.withColumn("row_id", lit(None)).rdd.zipWithIndex().map(
    lambda x: (*x[0], x[1])
).toDF([*legacy_df.columns, "row_id"])

modern_df = modern_df.withColumn("row_id", lit(None)).rdd.zipWithIndex().map(
    lambda x: (*x[0], x[1])
).toDF([*modern_df.columns, "row_id"])

# Join
joined = legacy_df.join(modern_df, "row_id", "outer").select(
    *[col(f"{c}") for c in legacy_df.columns if c != "row_id"],
    *[col(f"{c}") for c in modern_df.columns if c != "row_id"],
    "row_id"
)

# Compare fields and collect mismatches
mismatch_exprs = []
for field, _ in copybook_schema:
    mismatch_exprs.append(
        when(col(f"{field}") != col(f"{field}_1"), lit(field)).otherwise(lit(None))
    )

joined = joined.withColumn("mismatched_fields", array(*mismatch_exprs))
result = joined.withColumn("mismatch_count", size(col("mismatched_fields")))

# Save results
result.write.mode("overwrite").csv("validation_results")
