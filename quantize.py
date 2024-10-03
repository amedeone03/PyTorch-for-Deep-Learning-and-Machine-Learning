import tensorflow as tf

# Load the original SavedModel directory
saved_model_dir = "path_to_your_saved_model_directory"

# Set up the converter to convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable optimization - this applies quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optionally, define a representative dataset to ensure better accuracy during quantization.
def representative_data_gen():
    for _ in range(100):
        # Provide a sample input image from your dataset (size should match your input size, e.g., 640x640 for YOLOv5)
        yield [np.random.rand(1, 640, 640, 3).astype(np.float32)]

# Set representative dataset for integer quantization (8-bit)
converter.representative_dataset = representative_data_gen

# Specify supported types (only for full integer quantization)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ensure input/output is also quantized to int8
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# Convert the model
quantized_tflite_model = converter.convert()

# Save the quantized model to file
with open("yolov5x_quantized.tflite", "wb") as f:
    f.write(quantized_tflite_model)
