import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense

class Baseline(tf.keras.Model):
    def __init__(self, strategy="last", feature_index=0, stdev_index=-2):
        super().__init__()
        self.strategy = strategy
        self.feature_index = feature_index
        self.stdev_index = stdev_index

    def call(self, inputs):
        # Shape: [batch, time, features]
        last_step = inputs[:, -1, self.feature_index]
        if self.strategy == "last":
            # Predict class of 5th minute based on whether the last given time step
            # is relation to the 2nd to last (compare 79th to 80th minute)
            prev_step = inputs[:, -2, self.feature_index]
            diff = last_step - prev_step
            roll_stdev = inputs[:, -2, self.stdev_index]

            labels = tf.where(diff > roll_stdev, 1,
                     tf.where(diff < -roll_stdev, -1, 0))
            
            # reshape to [batch, 1, 1]
            return tf.expand_dims(tf.expand_dims(tf.cast(labels, tf.float32), axis=1), axis=2)
        else:
            # Predict 1 always
            batch_size = tf.shape(inputs)[0]
            return tf.ones([batch_size, 1, 1], dtype=tf.float32)
        
"""
Plot predictions:
wide_window = WindowGenerator(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=['label'],     # your class labels: -1, 0, or 1
    input_width=80,              # still use 80 minutes of input
    label_width=24,              # predict 24 labels at once
    offset=5                     # still predicting 5 minutes ahead
)
wide_window.plot(baseline)
"""

class SimpleDense(tf.keras.Model):
    """
    Treats input window as flat feature vector, without any concept of time or sequentiality
    Outputs classification into 3 classes using softmax
    """
    def __init__(self, input_width=80, num_features=19):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten() # [batch, time, features] -> [batch, time * features]
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)
    
"""
model = SimpleDense()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(window.train, validation_data=window.val, epochs=10)
"""

class SimpleCNN(tf.keras.Model):
    """
    1D CNN model that applies Conv1D over time dimension
    """
    def __init__(self, filters=32, kernel=3):
        super().__init__()
        self.conv = Conv1D(filters, kernel, activation='relu', padding='same')
        self.pool = GlobalMaxPooling1D()
        self.dense1 = Dense(64, activation='relu')
        self.out = Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.dense1(x)
        return self.out(x)

class InceptionCNN(tf.keras.Model):
    """
    Inception-style 1D CNN with multiple kernel widths to capture multi-scale temporal patterns
    """
    def __init__(self, filters=32):
        super().__init__()
        # Define layers for each branch
        self.branch1 = Conv1D(filters, kernel_size=1, activation='relu', padding='same')

        self.branch2_1 = Conv1D(filters, kernel_size=1, activation='relu', padding='same')
        self.branch2_2 = Conv1D(filters, kernel_size=3, activation='relu', padding='same')

        self.branch3_1 = Conv1D(filters, kernel_size=1, activation='relu', padding='same')
        self.branch3_2 = Conv1D(filters, kernel_size=5, activation='relu', padding='same')

        self.branch4_pool = MaxPooling1D(pool_size=3, strides=1, padding='same')
        self.branch4_conv = Conv1D(filters, kernel_size=1, activation='relu', padding='same')

        self.concat = Concatenate()
        self.pool = GlobalMaxPooling1D()
        self.fc = Dense(64, activation='relu')
        self.out = Dense(3, activation='softmax')

    def call(self, inputs):
        b1 = self.branch1(inputs)

        b2 = self.branch2_1(inputs)
        b2 = self.branch2_2(b2)

        b3 = self.branch3_1(inputs)
        b3 = self.branch3_2(b3)

        b4 = self.branch4_pool(inputs)
        b4 = self.branch4_conv(b4)

        x = self.concat([b1, b2, b3, b4])
        x = self.pool(x)
        x = self.fc(x)
        return self.out(x)