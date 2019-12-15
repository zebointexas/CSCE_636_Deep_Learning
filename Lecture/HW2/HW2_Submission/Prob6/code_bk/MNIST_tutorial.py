"""https://www.tensorflow.org/tutorials/"""
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# reference: https://www.tensorflow.org/guide/keras?hl=zh-CN 
model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(),

  # 512 here kind of meaning principle component | relu | # Adds a densely-connected layer with 64 units to the model:
  tf.keras.layers.Dense(512, activation=tf.nn.relu),

  # rf: https://blog.csdn.net/o0haidee0o/article/details/80514578 
  tf.keras.layers.Dropout(0.2),

  # Add a softmax layer with 10 output units:
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(test_loss)
print(test_accuracy)
