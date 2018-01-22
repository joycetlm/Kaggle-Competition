import functions as fc
import pandas as pd
import os
from sklearn import metrics
import tensorflow.contrib.learn as learn
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

print("start prediction")
path = "./data"
filename_submit = os.path.join(path,"submission.csv")
filename = os.path.join(path,"test.csv")

df_test = pd.read_csv(filename,na_values=['NA','?'])

print("preprocessing data")
id = df_test['id']
df_test.drop('id', axis=1, inplace=True)
df_test.drop('pickup_datetime', axis=1, inplace=True)
#df_test.drop('dropoff_datetime', axis=1, inplace=True)

fc.encode_text_index(df_test, 'vendor_id')
fc.encode_text_index(df_test, 'store_and_fwd_flag')
fc.encode_numeric_zscore(df_test, 'passenger_count')
fc.encode_numeric_zscore(df_test, 'pickup_longitude')
fc.encode_numeric_zscore(df_test, 'pickup_latitude')
fc.encode_numeric_zscore(df_test, 'dropoff_longitude')
fc.encode_numeric_zscore(df_test, 'dropoff_latitude')

x = df_test.as_matrix().astype(np.float32)

model_dir = "./dnn/NYC"
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x.shape[1])]

print("building network")
regressor = learn.DNNRegressor(
    model_dir= model_dir,
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1),
    feature_columns=feature_columns,
    hidden_units=[50, 25, 10])
    #hidden_units=[2, 5, 1])

print("predict")
pred = list(regressor.predict(x, as_iterable=True))

df_submit = pd.DataFrame(pred)
df_submit.insert(0,'id',id)
df_submit.columns = ['id','trip_duration']
df_submit['trip_duration'] = df_submit['trip_duration'].abs()
df_submit.to_csv(filename_submit, index=False)
