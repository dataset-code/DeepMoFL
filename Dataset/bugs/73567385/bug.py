from keras.layers import Dense, LSTM
from keras import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# X, y = make_classification(1000, 150*150*3, random_state=42)
# X = X.reshape(1000, 150, 150, 3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

batch_size = 32
img_height = 150
img_width = 150

# dataset_url = "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip"
# print(dataset_url)
# data_dir = tf.keras.utils.get_file(origin=dataset_url,
#                                     fname='CNR-EXT-Patches-150x150',
#                                     untar=True)

data_dir = "CNR-EXT-Patches-150x150"            
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
num_classes = 1

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Dense(num_classes)
])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)



    


i = 0
x = []
y = []
for example in val_ds:
    print(i)
    i+=1
    x.extend(list(example[0].numpy()))
    y.extend(list(example[1].numpy()))
# print(np.array(x).shape)
x = np.array(x)
y = np.array(y)
