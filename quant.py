import tensorflow as tf

# Load your existing .tflite model file (quantization won't be as effective)
converter = tf.lite.TFLiteConverter.from_tflite_model("yolov5x.tflite")

# Apply dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save the quantized model
quantized_tflite_model = converter.convert()

with open("yolov5x_dynamic_quantized.tflite", "wb") as f:
    f.write(quantized_tflite_model)
