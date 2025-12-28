import tensorflow as tf

# 1. LOAD & PREP: Get data and normalize pixels to 0-1 range
(x, y), _ = tf.keras.datasets.mnist.load_data()
x = x.reshape(-1, 28, 28, 1) / 255.0

# 2. ARCHITECTURE: Conv -> Flatten -> Output
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. COMPILE: Set the strategy (Optimizer + Loss)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. FIT: Train for 1 pass (epoch) through the data
model.fit(x, y, epochs=1)
model.save('mnist_model.h5')
