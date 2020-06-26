from pyspark import SparkContext
from pyspark.sql import *

# agg_files_path = "/home/sapchat/IdeaProjects/PUBGAggregation/inputs/aggregate/agg_match_stats_0.csv"
# kill_files_path = "/home/sapchat/IdeaProjects/PUBGAggregation/inputs/deaths/kill_match_stats_final_0.csv"
# master = "local"


agg_files_path = "hdfs://juneau:46730/pubg/aggregate"
kill_files_path = "hdfs://juneau:46730/pubg/deaths"
master = "spark://juneau:46750"


def returnIndex(l, ind):  # Transform (Split line on ,) line and return required index
    sp = l.split(",")
    if len(sp) >= ind:
        return sp[ind]
    else:
        return None


def transformTime(t):  # Bracketing time into buckets of 50
    if t is not None and "survive" not in t:
        return 50 * round(float(t) / 50.0)
    else:
        return t


def weapon_kills(sc):
    global kill_files_path
    kill_csv = sc.textFile(kill_files_path)
    counts = kill_csv.map(lambda line: returnIndex(line, 0)) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)

    counts.coalesce(1).saveAsTextFile("hdfs://juneau:46730/pubgDemo/weapon_kills")


def survival_numbers(sc):
    global agg_files_path
    kill_csv = sc.textFile(agg_files_path)
    counts = kill_csv.map(lambda line: transformTime(returnIndex(line, 12))) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)

    counts.coalesce(1).saveAsTextFile("hdfs://juneau:46730/pubgDemo/survivors")


def survival_rank(sc):
    global agg_files_path, master
    spark = SparkSession.builder \
        .master(master) \
        .appName("rank_survival") \
        .config(conf=sc.getConf()) \
        .getOrCreate()

    df = spark.read.format('csv').options(header='true', inferSchema='true').load(agg_files_path)
    df = df.filter(df.player_survive_time < 2500)
    res = df.groupBy("team_placement").agg({"player_survive_time": "avg"})
    res.coalesce(1).write.format("csv").save("hdfs://juneau:46730/pubgDemo/survival_times")


def team_size_placement(sc):
    global agg_files_path, master
    spark = SparkSession.builder \
        .master(master) \
        .appName("try_stuff") \
        .config(conf=sc.getConf()) \
        .getOrCreate()

    df = spark.read.format('csv').options(header='true', inferSchema='true').load(agg_files_path)
    df = df.filter(df.player_survive_time < 2500)
    res = df.groupBy("team_placement").pivot("game_size").agg({"player_survive_time": "avg"})
    res.coalesce(1).write.format("csv").save("hdfs://juneau:46730/pubgDemo/survival_team_size_placement")


def survival_party(sc):
    global agg_files_path, master
    spark = SparkSession.builder \
        .master(master) \
        .appName("party_survival") \
        .config(conf=sc.getConf()) \
        .getOrCreate()

    df = spark.read.format('csv').options(header='true', inferSchema='true').load(agg_files_path)
    df = df.filter(df.player_survive_time < 2500)
    res = df.groupBy("party_size").agg({"player_survive_time": "avg"})
    res.coalesce(1).write.format("csv").save("hdfs://juneau:46730/pubgDemo/survival_party")


def damage_pos(sc):
    global agg_files_path, master
    spark = SparkSession.builder \
        .master(master) \
        .appName("damage_position") \
        .config(conf=sc.getConf()) \
        .getOrCreate()

    df = spark.read.format('csv').options(header='true', inferSchema='true').load(agg_files_path)
    res = df.groupBy("team_placement").agg({"player_dmg": "avg"})
    res.coalesce(1).write.format("csv").save("hdfs://juneau:46730/pubgDemo/average_damage")


if __name__ == '__main__':
    global master
    sc = SparkContext(master, "PUBG PySpark")
    weapon_kills(sc)
    survival_numbers(sc)
    survival_rank(sc)
    survival_party(sc)
    damage_pos(sc)
    team_size_placement(sc)
