from elt import *
import pytest
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as f

glueContext = get_glue_context()
spark = glueContext.spark_session

df, last_index = get_data_frame(glueContext, "test")
df2, last_index2 = get_joined_dataframe(df, last_index)
df2.show()
assert last_index2 - last_index == 2500

df3 = get_tokens_and_mask(df2)

df3.show()






