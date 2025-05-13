model = tf.keras.Sequential([
    tf.keras.layers.Dense(...),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])