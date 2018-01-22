import functions as fc
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow.contrib.learn as learn
import time

tf.logging.set_verbosity(tf.logging.ERROR)

path = "./data/"

filename = os.path.join(path, "train.csv")
df = pd.read_csv(filename,na_values=['NA','?'])


df.drop('id', axis=1, inplace=True)
df.drop('pickup_datetime', axis=1, inplace=True)
df.drop('dropoff_datetime', axis=1, inplace=True)

fc.encode_text_index(df, 'vendor_id')
fc.encode_text_index(df, 'store_and_fwd_flag')
fc.encode_numeric_zscore(df, 'passenger_count')
fc.encode_numeric_zscore(df, 'pickup_longitude')
fc.encode_numeric_zscore(df, 'pickup_latitude')
fc.encode_numeric_zscore(df, 'dropoff_longitude')
fc.encode_numeric_zscore(df, 'dropoff_latitude')

x, y = fc.to_xy(df,'trip_duration')
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

model_dir = fc.get_model_dir('NYC',True)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x.shape[1])]

regressor = learn.DNNRegressor(
    model_dir= model_dir,
    #dropout=0.2,
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1),
    feature_columns=feature_columns,
    hidden_units=[50, 25, 10])
    #hidden_units=[2, 5, 1])


validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    x_test,
    y_test,
    every_n_steps=500,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=50)
start_time = time.time()
print("start training")
#show.now()
regressor.fit(x_train, y_train,monitors=[validation_monitor],steps=100000)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(fc.hms_string(elapsed_time)))

print("Best step: {}, Last successful step: {}".format(
validation_monitor.best_step,validation_monitor._last_successful_step))

# Predict
pred = list(regressor.predict(x_test, as_iterable=True))
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))
