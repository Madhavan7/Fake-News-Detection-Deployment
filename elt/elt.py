import sys
from text_processor import *
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as f
  
#Setting up spark and glue session
# sc = SparkContext.getOrCreate()
# glueContext = GlueContext(sc)
# spark = glueContext.spark_session
# job = Job(glueContext)

processor = TextProcessor()

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
  df2 = df.select('*').where(f'{last_index} <= id and id <= {last_index + 2500}')
  df3 = df2.groupby('id', 'account', 'sentiment').agg(f.collect_list('review').alias('collections'))
  df4 = df3.select('id', 'account', 'sentiment', f.array_join('collections', ' ').alias('review'))
  return df4, last_index + 2500

def get_tokens_and_mask(df):
  return df.select('id, account, sentiment, processor.tokenize(review)[0], processor.tokenize(review)[1]')


# last_index += 2500

# job.commit()
