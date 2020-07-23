# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, isnull,from_unixtime,instr,when,regexp_replace
import sys

spark = SparkSession.builder.appName("Spark").getOrCreate()


def load_spark_data(path):
    df = spark.read.json(path)
    df.persist()
    return df

def clean_data(df):
    df = df.dropna(how="any", subset=["userId", "sessionId"])
    df = df.filter(df.userId != '')
    churn_df = df.filter((df.page == "Cancellation Confirmation") & (df.level.isin("free", "paid"))).select(
        "userId").distinct()
    df = df.withColumn("churn", df.userId.isin([(i[0]) for i in churn_df.collect()]).cast("int"))
    df = df.withColumn("date_time", from_unixtime(df.ts / 1000))
    return df


def save_level_churn_byUser(df,save_path):
    level_a_df = df.select(["churn", "userId", "level"]).distinct().toPandas()
    level_a_df.to_csv(save_path,index=False)
    print ("level_churn_byUser saved.")


def save_page_churn_byUser(df,save_path):
    df_temp1 = df.groupby(["churn", "page"]).count().withColumnRenamed("count", "count_s")
    df_temp2 = df.groupby(["page"]).count()
    df_temp1 = df_temp1.join(df_temp2, ["page"], "left")
    df_temp1 = df_temp1.withColumn("prop", df_temp1["count_s"] / df_temp1["count"]).sort("page").toPandas()
    df_temp1.to_csv(save_path, index=False)
    print("page_churn_byUser saved.")


def save_numofsongs_byUser(df,save_path):
    df_temp3 = df.select(["churn","userId","song"]).distinct().where(df["song"].isNotNull()).groupby(["churn","userId"]).count().select(["churn","count"]).toPandas()
    df_temp3.to_csv(save_path, index=False)
    print("numofsongs_byUser saved.")


def features_persis(df,save_path):

    df = df.withColumn('new_gender', when(df.gender == 'F', 0).when(df.gender == 'M', 1).otherwise(2))
    df = df.withColumn('new_level', when(df.level == 'free', 0).when(df.level == 'paid', 1).otherwise(2))
    f_length_df = df.groupby("userId").agg(avg(df.length).alias("avgLength"), max(df.length).alias("maxLength"),
                                           min(df.length).alias("minLength"))
    songs_count_df = df.select(["userId", "song"]).distinct().where(df["song"].isNotNull()).groupby(
        ["userId"]).count().withColumnRenamed("count", "songs_count")
    session_count_df = df.select(["userId", "sessionId"]).distinct().where(df["sessionId"].isNotNull()).groupby(
        ["userId"]).count().withColumnRenamed("count", "session_count")
    f_page_df = df.groupby(["userId", "page"]).count().withColumnRenamed("count", "page_count").sort("userId")
    pages = f_page_df.select(regexp_replace(col('page'), "\\s+", "")).alias("new_page").distinct().rdd.flatMap(
        lambda x: x).collect()
    exp = [when(f_page_df.page == p, f_page_df.page_count).otherwise(0).alias(str(p)) for p in pages]
    f_page_df = f_page_df.select(["userId"] + exp)
    f_page_df = f_page_df.groupby('userId').agg(*[max(p).alias(p) for p in pages])
    print(f_page_df.show(5))
    user_item = df.groupby(["userId", "churn", "new_gender"]).agg(max(df.new_level).alias("max_level")).sort("userId")

    user_item = user_item.join(f_length_df, ["userId"], "left")
    user_item = user_item.join(songs_count_df, ["userId"], "left")
    user_item = user_item.join(session_count_df, ["userId"], "left")
    user_item = user_item.join(f_page_df, ["userId"], "left")
    user_item.persist()
    user_item = user_item.select(
        [p for p in user_item.columns if p not in ["Cancellation Confirmation", "Cancel"]]).sort("userId")
    # user_item.write.parquet('user_item.parquet', mode='overwrite')
    user_item.write.csv(save_path, header = True, mode = 'overwrite')

def main():
    if len(sys.argv) == 6:
        print("load spark data.")
        df = load_spark_data(sys.argv[1])

        print("clean spark data, drop null userId and sessionId, define churn and transform ts to data_time.")
        df = clean_data(df)

        print("save level_churn_byUser.csv and use for visualization.")
        save_level_churn_byUser(df,sys.argv[2])

        print("save page_churn_byUser.csv and use for visualization.")
        save_page_churn_byUser(df,sys.argv[3])

        print("save numofsongs_byUser.csv and use for visualization.")
        save_numofsongs_byUser(df,sys.argv[4])

        print("save user_item.csv and use for model.")
        features_persis(df,sys.argv[5])
    else:
        print('Please provide the filepaths of the event data ' \
              'datasets as the first five arguments respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the sixth argument. \n\nExample: python process_data.py ' \
              'mini_sparkify_event_data.json level_churn_byUser.csv page_churn_byUser.csv ' \
              'numofsongs_byUser.csv user_item.csv')


if __name__ == '__main__':
    main()