import sys
from text_processor import *
from transformers import AutoTokenizer
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, IntegerType
  
#Setting up spark and glue session
# sc = SparkContext.getOrCreate()
# glueContext = GlueContext(sc)
# spark = glueContext.spark_session
# job = Job(glueContext)

MAX_LENGTH = 512
processor = TextProcessor(MAX_LENGTH)
tokenize_udf = f.udf(processor.tokenize, ArrayType(IntegerType()))
mask_udf = f.udf(processor.mask, ArrayType(IntegerType()))





def get_glue_context():
  sc = SparkContext.getOrCreate()
  glueContext = GlueContext(sc)
  return glueContext

def get_dataframe_count(df):
  return df.count()

def get_data_frame(glueContext, mode = "test"):
  if mode != "test":
    dyf = glueContext.create_dynamic_frame.from_catalog(database='wcd_project_training', table_name='training')
  else:
    dyf = glueContext.create_sample_dynamic_frame_from_catalog(database='wcd_project_training', num=15,table_name='training')
  df = dyf.toDF()
  dyf_index = glueContext.create_dynamic_frame.from_catalog(database='wcd_project_training', table_name='index')
  df_index = dyf_index.toDF()
  last_index = df_index.first()['index']
  return df, last_index


def get_joined_dataframe(df, last_index):
  df2 = df.select('*').where(f'{last_index} <= id and id < {last_index + 1500}')
  df3 = df2.groupby('id', 'account', 'sentiment').agg(f.collect_list('review').alias('collections'))
  df4 = df3.select('id', 'account', 'sentiment', f.array_join('collections', ' ').alias('review'))
  return df4, last_index + 1500

def get_tokens_and_mask(df):
  return df.withColumn("tokens", tokenize_udf(df["review"])).withColumn("mask", mask_udf(df["review"])).withColumn("date-time", f.current_timestamp())

def write_dataframe_to_s3(glueContext, df, last_index):
  spark = glueContext.spark_session
  main_sink = glueContext.getSink(connection_type="s3", path= "s3://wcd-project-twitter-datasets/data/features/",
                                  updateBehaviour="UPDATE_IN_DATABASE",
                                  partitionKeys=["date-time"],
                                  compression = "gzip",
                                  enableUpdateCatalog=True,
                                  transformation_ctx="main_sink")
  main_sink.setCatalogInfo(catalogDatabase="wcd_project_training", catalogTableName="features")
  main_sink.setFormat("csv")
  main_sink.writeFrame(df)


  #create a dataframe with the index and then write to s3
  index_df = spark.createDataFrame({'index': last_index})
  index_sink = glueContext.getSink(connection_type="s3", path= "s3://wcd-project-twitter-datasets/data/index/",
                                  updateBehaviour="UPDATE_IN_DATABASE",
                                  enableUpdateCatalog=True,
                                  transformation_ctx="index_sink")
  index_sink.setCatalogInfo(catalogDatabase="wcd_project_training", catalogTableName="index")
  index_sink.setFormat("csv")
  index_sink.writeFrame(index_df)

def pipeline(glueContext, mode = "test"):
  df, last_index = get_data_frame(glueContext, mode)
  df2, last_index2 = get_joined_dataframe(df, last_index)
  count = get_dataframe_count(df2)
  print(last_index2, count)
  if last_index2 > count:
    print("not enough new data")

  df3 = get_tokens_and_mask(df2)
  write_dataframe_to_s3(glueContext, df3, last_index2)

if __name__ == '__main__':
  glueContext = get_glue_context()
  job = Job(glueContext)
  pipeline(glueContext, mode="real")
  print("done")
  job.commit()
