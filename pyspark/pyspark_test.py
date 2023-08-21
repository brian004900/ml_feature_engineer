import sys

from pyspark import StorageLevel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession

# spark-submit --jars mariadb-java-client-3.0.8.jar hw3.py


def main():
    appName = "hw3"
    master = "local"
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    user = "root"
    password = ""

    jdbc_url = "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.batter_counts")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df1.createOrReplaceTempView("batter_counts")
    df1.persist(StorageLevel.DISK_ONLY)

    df2 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.battersInGame")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df2.createOrReplaceTempView("battersInGame")
    df2.persist(StorageLevel.DISK_ONLY)

    df3 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.game_temp")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df3.createOrReplaceTempView("game_temp")
    df3.persist(StorageLevel.DISK_ONLY)

    t1 = spark.sql(
        """
        select battersInGame.game_id
        , battersInGame.batter, local_date
        , batter_counts.Hit as hit, batter_counts.atBat as atbat
        from battersInGame
        join game_temp
        on battersInGame.game_id = game_temp.game_id
        join batter_counts
        on battersInGame.batter = batter_counts.batter and battersInGame.game_id = batter_counts.game_id
        order by battersInGame.batter;
        """
    )

    t1.createOrReplaceTempView("t1")
    t1.persist(StorageLevel.DISK_ONLY)

    query = """
        select
              a.game_id
            , a.batter
            , a.local_date
            , case when SUM(b.atbat) = 0 then 0
            else cast(SUM(b.hit) as float )/cast(SUM(b.atbat) as float) end as rolling_batting
        from __THIS__ a
        join __THIS__ b
        on b.batter = a.batter
        where b.local_date between date_add(a.local_date, -100) and date_add(a.local_date,-1)
        group by a.game_id, a.batter, a.local_date
        order by a.batter, a.local_date;
        """

    sqlTrans = SQLTransformer().setStatement(query)

    sqlTrans.transform(t1).show()


if __name__ == "__main__":
    sys.exit(main())
