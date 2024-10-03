import onnx
from onnxruntime.quantization import QuantType, QuantizationMode, quantize_static, QuantFormat, CalibrationDataReader
import onnxruntime
import cv2
import os
import numpy as np

# Format YOLOv5 input
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

# Preprocess images for calibration
def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocesses them
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names

    batch_data = []
    for image_name in batch_filenames:
        image_filepath = os.path.join(images_folder, image_name)
        pillow_img = cv2.imread(image_filepath)
        pillow_img = format_yolov5(pillow_img)
        nchw_data = cv2.dnn.blobFromImage(pillow_img, 1 / 255.0, (height, width), swapRB=True)
        batch_data.append(nchw_data)

    return batch_data

# Define a calibration data reader
class yolov5_cal_data_reader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert images to input data
        self.nchw_data_list = _preprocess_images(calibration_image_folder, height, width, size_limit=0)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nchw_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: nchw_data} for nchw_data in self.nchw_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

# Perform static quantization
model_path = './best.onnx'  # Path to your original ONNX model
output_model_quant = './best_quant.onnx'  # Path to save the quantized model
calibration_data_folder = './calibration_data/'  # Folder containing calibration images

# Initialize the data reader
calibration_reader = yolov5_cal_data_reader(calibration_image_folder=calibration_data_folder, model_path=model_path)

# Perform static quantization with calibration data
quantize_static(
    model_path,
    output_model_quant,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QOperator,  # Quantize operator weights and activations
    activation_type=QuantType.QInt8,     # Quantize activations to int8
    weight_type=QuantType.QInt8,         # Quantize weights to int8
    optimize_model=True,                 # Optimize the quantized model
)

print(f"Quantized model saved to: {output_model_quant}")
