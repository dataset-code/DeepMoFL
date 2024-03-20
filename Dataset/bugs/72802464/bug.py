import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np

dataset = datasets.cifar100

(training_images, training_labels), (validation_images, validation_labels) = dataset.load_data()

training_images = training_images / 255.0
validation_images = validation_images / 255.0

model = tf.keras.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(32,32,3)),
                                    tf.keras.layers.Dense(500, activation='relu'),
                                    tf.keras.layers.Dense(300, activation='relu'),
                                    tf.keras.layers.Dense(10, activation= 'softmax')
                                    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(training_images,
                    training_labels,
                    batch_size=100,
                    epochs=10,
                    validation_data = (validation_images, validation_labels)
                    )

# model = tf.keras.models.load_model("model.h5")
# s = model.predict(validation_images)
# print(s)