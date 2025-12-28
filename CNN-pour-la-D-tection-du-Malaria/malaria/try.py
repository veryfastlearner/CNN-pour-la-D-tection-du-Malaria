import tensorflow as tf
data_dir = "/Users/Administrateur/.cache/kagglehub/datasets/iarunava/cell-images-for-detecting-malaria/versions/1/cell_images/cell_images"
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(64,64), batch_size=32, validation_split=0.2, subset="training", seed=123)
model = tf.keras.Sequential([tf.keras.layers.Rescaling(1./255), 
                             tf.keras.layers.Conv2D(16,3,activation='relu'), 
                             tf.keras.layers.Flatten(), 
                             tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.fit(train_ds, epochs=8)
model.save('malaria_mini.h5')

