import tensorflow as tf
import numpy as np

# Load the TensorFlow SavedModel
saved_model_dir = "yolov5x_tf_saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for quantization
def representative_data_gen():
    for _ in range(100):
        # Generate input data in the expected shape of (1, 640, 640, 3) for YOLOv5x
        yield [np.random.rand(1, 640, 640, 3).astype(np.float32)]

converter.representative_dataset = representative_data_gen

# Ensure input/output tensors are also quantized to int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Ensure input is quantized to int8
converter.inference_output_type = tf.int8  # Ensure output is quantized to int8

# Convert the model
quantized_model = converter.convert()

# Save the quantized model to file
with open("yolov5x_quantized_int8.tflite", "wb") as f:
    f.write(quantized_model)

print("Model quantized and saved as yolov5x_quantized_int8.tflite")
