# Custom L1 Distance layer module | Why ? : it's needed to load the custom model

# Import
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Cstom L1 Distance layer from Jupyter
# Siamese L1 distance class
class L1Dist(Layer):
    # init methot
    def __init__(self, **kwargs):
        super().__init__()
    
    # similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)