import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    sum, when, col, expr, count, avg, min, max, corr,
    row_number, current_date, add_months, explode,
    split, trim, broadcast, lower
)
from pyspark.sql.types import (
    IntegerType, LongType, FloatType, DoubleType,
    DecimalType, ShortType
)
from pyspark.sql.window import Window


# ---------------------------
# Spark Session
# ---------------------------
spark = SparkSession.builder \
    .appName("assignment-app-amitkumarsingh89") \
    .getOrCreate()


# ---------------------------
# Utility Functions
# ---------------------------
def get_column_types(df):
    numeric_types = (
        IntegerType, LongType, FloatType,
        DoubleType, DecimalType, ShortType
    )
    numerical_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, numeric_types)
    ]
    categorical_cols = [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, numeric_types)
    ]
    return numerical_cols, categorical_cols


def get_missing_value(df):
    total_rows = df.count()
    agg_exprs = [
        sum(
            when((col(c).isNull()) | (col(c) == ""), 1)
            .otherwise(0)
        ).alias(c)
        for c in df.columns
    ]
    agg_df = df.agg(*agg_exprs)
    stack_expr = "stack({}, {}) as (column_name, null_count)".format(
        len(df.columns),
        ",".join([f"'{c}', `{c}`" for c in df.columns])
    )
    pivot_df = agg_df.select(expr(stack_expr))
    return pivot_df.withColumn(
        "null_percentage",
        (col("null_count") / total_rows) * 100
    )


# ---------------------------
# KPI Functions
# ---------------------------
def top_categories_jobs_posting(df):
    return (
        df
        .groupBy("Job Category")
        .count()
        .withColumnRenamed("count", "total_job_posted")
        .orderBy(col("total_job_posted").desc())
        .limit(10)
    )


def salary_distribution_per_category(df):
    df_salary = df.select(
        col("Job Category"),
        col("Salary Range From").cast("double"),
        col("Salary Range To").cast("double")
    ).withColumn(
        "avg_salary",
        (col("Salary Range From") + col("Salary Range To")) / 2
    )
    return (
        df_salary
        .groupBy("Job Category")
        .agg(
            count("*").alias("job_count"),
            min("Salary Range From").alias("min_salary"),
            max("Salary Range To").alias("max_salary"),
            avg("avg_salary").alias("mean_salary"),
            expr("percentile_approx(avg_salary, 0.5)").alias("median_salary"),
            expr("percentile_approx(avg_salary, 0.25)").alias("p25_salary"),
            expr("percentile_approx(avg_salary, 0.75)").alias("p75_salary")
        )
    )


def degree_salary_correlation(df):
    df_salary = df.select(
        col("Minimum Qual Requirements"),
        ((col("Salary Range From") + col("Salary Range To")) / 2)
        .alias("avg_salary")
    )
    df_degree = df_salary.withColumn(
        "degree_score",
        when(lower(col("Minimum Qual Requirements")).contains("phd"), 5)
        .when(lower(col("Minimum Qual Requirements")).contains("doctorate"), 5)
        .when(lower(col("Minimum Qual Requirements")).contains("master"), 4)
        .when(lower(col("Minimum Qual Requirements")).contains("bachelor"), 3)
        .when(lower(col("Minimum Qual Requirements")).contains("associate"), 2)
        .when(lower(col("Minimum Qual Requirements")).contains("high school"), 1)
        .otherwise(None)
    ).filter(col("degree_score").isNotNull())
    return df_degree.agg(
        corr("degree_score", "avg_salary")
        .alias("degree_salary_correlation")
    )


def highest_salary_job_per_agency(df):
    df_salary = df.select(
        col("Agency"),
        col("Job ID"),
        col("Business Title"),
        ((col("Salary Range From") + col("Salary Range To")) / 2)
        .alias("avg_salary")
    )
    window_spec = Window.partitionBy("Agency") \
                        .orderBy(col("avg_salary").desc())
    return (
        df_salary
        .withColumn("rank", row_number().over(window_spec))
        .filter(col("rank") == 1)
        .drop("rank")
    )

def avg_salary_per_agency_last_2_years(df):
    df_filtered = df.select(
        col("Agency"),
        col("Posting Date").cast("date"),
        ((col("Salary Range From") + col("Salary Range To")) / 2)
        .alias("avg_salary")
    ).filter(
        col("Posting Date") >= add_months(current_date(), -24)
    )
    return (
        df_filtered
        .groupBy("Agency")
        .agg(
            avg("avg_salary").alias("avg_salary_last_2_years")
        )
        .orderBy(col("avg_salary_last_2_years").desc())
    )


def highest_paid_skills_us(df, us_cities_list, top_n=10):
    cities_df = spark.createDataFrame(
        [(city.lower(),) for city in us_cities_list],
        ["city"]
    )
    df_us = (
        df
        .withColumn("city_lower", lower(col("Work Location")))
        .join(
            broadcast(cities_df),
            col("city_lower") == col("city"),
            "inner"
        )
    )
    df_skills = (
        df_us
        .withColumn(
            "avg_salary",
            (col("Salary Range From") + col("Salary Range To")) / 2
        )
        .withColumn("skill", explode(split(col("Preferred Skills"), ",")))
        .withColumn("skill", lower(trim(col("skill"))))
    )
    return (
        df_skills
        .groupBy("skill")
        .agg(avg("avg_salary").alias("avg_salary"))
        .orderBy(col("avg_salary").desc())
        .limit(top_n)
    )


# ---------------------------
# MAIN
# ---------------------------
def main():

    # Schema inference from 1% sample
    df_sample = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("samplingRatio", 0.01) \
        .csv("/dataset/nyc-jobs.csv")

    inferred_schema = df_sample.schema

    df = spark.read \
        .option("header", True) \
        .schema(inferred_schema) \
        .csv("/dataset/nyc-jobs.csv") \
        .coalesce(2)

    agency_df = df.select(
        "Job ID",
        "Agency",
        "Posting Type",
        "Business Title",
        "Level",
        "Job Category",
        "Salary Range From",
        "Salary Range To",
        "Work Location",
        "Work Location 1",
        "Division/Work Unit",
        "Minimum Qual Requirements",
        "Preferred Skills",
        "Posting Date"
    )

    print("Total Records:", agency_df.count())

    # Convert blanks to null
    df_clean = agency_df
    for c in df_clean.columns:
        df_clean = df_clean.withColumn(
            c,
            F.when(F.trim(F.col(c)) == "", None)
             .otherwise(F.col(c))
        )

    # Fill selected categorical columns
    df_clean = df_clean.fillna("Unknown", subset=[
        "Job Category",
        "Minimum Qual Requirements",
        "Preferred Skills",
        "Work Location 1"
    ])

    # Cache and materialize
    df_cache = df_clean.cache()
    df_cache.count()

    # US Cities
    us_cities = {
        "new york", "los angeles", "chicago", "houston",
        "phoenix", "philadelphia", "san antonio",
        "san diego", "dallas", "san jose"
    }

    # Execute KPIs
    top_categories_jobs_posting(df_cache).show(truncate=False)
    salary_distribution_per_category(df_cache).show(truncate=False)
    degree_salary_correlation(df_cache).show(truncate=False)
    highest_salary_job_per_agency(df_cache).show(truncate=False)
    avg_salary_per_agency_last_2_years(df_cache).show(truncate=False)
    highest_paid_skills_us(df_cache, us_cities).show(truncate=False)

    ############################################
	########## SAVE KPI OUTPUTS ################
	############################################

	base_output_path = "/dataset/processed/kpi_outputs"

	#Top 10 Categories
	top_categories_jobs_posting(df_cache) \
	    .write.mode("overwrite") \
	    .parquet(f"{base_output_path}/top_categories")

	#Salary Distribution
	salary_distribution_per_category(df_cache) \
	    .write.mode("overwrite") \
	    .parquet(f"{base_output_path}/salary_distribution")

	#Degree Salary Correlation
	degree_salary_correlation(df_cache) \
	    .write.mode("overwrite") \
	    .parquet(f"{base_output_path}/degree_salary_correlation")

	#Highest Salary Job Per Agency
	highest_salary_job_per_agency(df_cache) \
	    .write.mode("overwrite") \
	    .parquet(f"{base_output_path}/highest_salary_per_agency")

	#Avg Salary Per Agency Last 2 Years
	avg_salary_per_agency_last_2_years(df_cache) \
	    .write.mode("overwrite") \
	    .parquet(f"{base_output_path}/avg_salary_last_2_years")

	#Highest Paid Skills US
	highest_paid_skills_us(df_cache, us_cities) \
	    .write.mode("overwrite") \
	    .parquet(f"{base_output_path}/highest_paid_skills_us")

	print("All KPI outputs successfully stored.")


if __name__ == "__main__":
    main()