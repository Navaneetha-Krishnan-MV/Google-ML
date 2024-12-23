import tensorflow as tf
import numpy as np
from tensorflow import keras

#model creation - single layer
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#loss and optimizing
model.compile(optimizer='sgd', loss='mean_squared_error')

#data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


model.fit(xs, ys, epochs=50)

# Convert input to NumPy array and predict
input_value = np.array([10.0])  # Convert list to NumPy array
prediction = model.predict(input_value)

print(prediction)