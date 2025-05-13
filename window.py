import pandas as pd
import numpy as np
import tensforflow as tf
import matplotlib as plt

class WindowGenerator():
    """
    Generates set of predictions based on window of consecutive samples
    """
    def __init__(self, train_df, val_df, test_df, label_columns=['label'],
                 input_width=80, label_width=1, offset=4):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset

        self.total_window_size = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        """
        Splits a window of time series data into input and label tensors.

            Parameters:
                features (Tensor): A 3D tensor of shape [batch_size, total_window_size, num_features],
                                representing a batch of sliding windows from the dataset.

            Returns:
                inputs (Tensor): A 3D tensor of shape [batch_size, input_width, num_features],
                                containing the input portion of each window.
                                
                labels (Tensor): A 3D tensor of shape [batch_size, label_width, num_label_columns],
                                containing the label portion of each window. If label_columns is specified,
                                only those columns are extracted from the label slice.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # if only using the 'label' column
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Price', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 4 * max_subplots))  # Smaller height per subplot

        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Input', marker='.', zorder=-10)

            # Extract scalar label for this window (e.g., -1, 0, 1)
            true_label = int(labels[n, 0, 0])
            label_text = { -1: '↓ Down', 0: '→ Flat', 1: '↑ Up' }.get(true_label, str(true_label))
            plt.title(f"True label: {label_text}", loc='left')

            if model is not None:
                predictions = model(inputs)
                pred_label = tf.argmax(predictions[n, 0]).numpy() if predictions.shape[-1] > 1 else int(tf.round(predictions[n, 0, 0]).numpy())
                pred_text = { -1: '↓ Down', 0: '→ Flat', 1: '↑ Up' }.get(pred_label, str(pred_label))
                plt.title(f"Predicted: {pred_text}", loc='right')

            if n == 0:
                plt.legend()

        plt.xlabel('Time [min]')
    
    def make_dataset(self, data):
        """
        Converts a time series DataFrame into a tf.data.Dataset of 
        (input_window, classification_label) pairs for model training.

        This method:
        - Uses a sliding window of length `total_window_size` (85 minutes)
        - Generates overlapping sequences with stride = 1. 
        - Converts the data into batches (default size = 32).
        - Applies the `split_window` method to extract input and label slices.
        - Assumes labels (-1, 0, 1) are already precomputed and stored in a label column.

        Parameters:
            data (pd.DataFrame or np.ndarray): Time series data with features and a 'label' column.

        Returns:
            tf.data.Dataset: A dataset of (inputs, labels), where:
                            - inputs: shape (batch_size, input_width, num_features)
                            - labels: shape (batch_size, label_width, 1) with values -1, 0, or 1
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=64)

        ds = ds.map(self.split_window)

        return ds

    @property
    def example(self):
        """
        Returns a cached example batch (inputs, labels) from the training dataset.
        Used for visualization and quick inspection.

        Caches the first batch from self.train for reuse.
        """
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
    



