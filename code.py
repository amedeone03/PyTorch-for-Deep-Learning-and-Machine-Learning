import argparse
import os
import sys
import re
import numpy as np
import logging
import subprocess
import onnxruntime as ort

available_backends = []

DIRTY = 1
CLEAR_DIRTY = 0

TIME = 'avg inference time:'
INIT_TIME = TIME + ' FAILED'

class ModelIO:
    def __init__(self):
        self.intput_size = (-1,-1,-1,-1)
        self.intput_type = 'unknown'
        self.output_count = 0
        self.recommand_compile_options = ''

# Limit the available backends to MDLA 3.0 and VPU
def query_backends():
    available_backends.extend(['mdla_3.0', 'vpu'])
    logging.info(f"Backends limited to: {available_backends}")

def query_tflite_model_io_info(model):
    ret = ModelIO()

    intput_size = (-1,-1,-1,-1)
    intput_type = 'kTfLiteUInt8'
    output_count = 0
    batcmd = ('ncc-tflite --show-io-info %s' % model )
    res = subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout
    
    logging.debug(result)

    # Find output count
    match = re.search('of output tensors: (\d+)', result)
    if match:
        output_count = int(match.group(1))
    else:
        logging.error("FAIL to find output count of model")
        return ret

    ret.output_count = output_count

    # Find input size
    tmp = result.partition("# of output tensors")[0]
    match = re.search('Shape: {(\d+),(\d+),(\d+),(\d+)}', tmp)
    if match:
        intput_size = ( int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)) )
    else:
        logging.error("FAIL to find input size of model")
        return ret

    ret.intput_size = intput_size

    # Find input type
    match = re.search('Type: (\w+)', tmp)
    if match:
        intput_type = (match.group(1))
    else:
        logging.error("FAIL to find input type of model")
        return ret

    ret.intput_type = intput_type
    if intput_type == 'kTfLiteFloat32':
        ret.recommand_compile_options = ' --relax-fp32 '

    return ret

def gen_input_bin(model_io, filename):
    # Generate input data, format can be uint8 or float32
    ndarray = np.random.randint(0, high=255, size=model_io.intput_size, dtype=np.uint8)
    if model_io.intput_type == 'kTfLiteFloat32':
        ndarray = np.float32(ndarray) / 255.0
    ndarray.tofile(filename)

def tflite_2_dla(tflite_model, target, dla_model, compile_options=""):
    if os.path.exists(dla_model):
        return

    batcmd = ('ncc-tflite -arch %s %s -o %s %s' % (target, tflite_model, dla_model, compile_options))
    res = subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout
    logging.debug(result)

def benchmark_tflite_model(model, target, count=100, compile_options="", cache_dla=False, profile=False):
    dla = model.replace('.tflite', "-"+ target + '.dla')
    tflite_2_dla(model, target, dla, compile_options)
    if not os.path.exists(dla):
        logging.error("FAIL to convert model to dla with target %s, %s" % (target, model))
        return

    # Find input size and output count of model
    modeil_io = query_tflite_model_io_info(model)

    # Generate random input.bin
    input_bin = 'input.bin'
    gen_input_bin(modeil_io, input_bin)

    batcmd = 'neuronrt -m hw -a %s -c %s -b 100 -i %s ' % (dla, count, input_bin)
    for i in range(modeil_io.output_count):
        batcmd = batcmd + (' -o output_%s.bin ' % i)

    if profile is True:
        logging.info("%s, %s, inference start" % (model, target))

    res = subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    if profile is True:
        logging.info("%s, %s, inference stop" % (model, target))
        print(res.stdout)

    result = res.stdout     
    logging.debug(result)

    if not cache_dla:
        os.remove(dla)

    # Find 'Total inference time' in output log
    match = re.search('Total inference time = (\d+)', result)
    if match:
        time = float(match.group(1))
        # Calculate average inference time
        avg_time = time / count
    else:
        logging.error("FAIL to find avg inference time of model")
        return

    logging.info('%s, %s, avg inference time: %s' % (model, target, avg_time))


# ONNX Inference Logic
def benchmark_onnx_model(model_path, count=100):
    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Generate random input data based on the model's first input shape
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_data = np.random.random_sample(input_shape).astype(np.float32)

    logging.info(f"Running ONNX inference for model: {model_path}")

    total_time = 0

    # Run the inference `count` times and calculate the average time
    for _ in range(count):
        start_time = time.time()
        ort_session.run(None, {input_name: input_data})
        end_time = time.time()

        total_time += (end_time - start_time)

    avg_time = total_time / count
    logging.info(f"ONNX Model: {model_path}, Avg inference time: {avg_time:.6f} seconds")


# Argument parsing
query_backends()

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--auto', action='store_true', help='Auto run, find all models in current folder and inference them one by one.')
parser.add_argument('--stress', action='store_true', help='Stress test, find all models in current folder and inference them one by one.')
parser.add_argument('--file', default="", help='Input a tflite or onnx model to inference.')
parser.add_argument('--target', default=available_backends[0], choices=available_backends, help='Choose a backend to inference. Default: %s' % available_backends[0])
parser.add_argument('--count', default=100, type=int, help='How many times of inference. Default: 100')
parser.add_argument('--profile', action='store_true', help='print profile information')
parser.add_argument('--options', default="", help='Additional ncc-tflite options')

args = parser.parse_args()

# Prepare logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if args.auto is True or args.stress is True:
    cache_dla = args.stress
    for file in os.listdir(os.getcwd()):
        if file.endswith('.tflite'):
            modeil_io = query_tflite_model_io_info(file)
            compile_options = args.options + modeil_io.recommand_compile_options
            benchmark_tflite_model(file, args.target, args.count, compile_options, cache_dla, args.profile)
        elif file.endswith('.onnx'):
            benchmark_onnx_model(file, args.count)
else:
    if os.path.exists(args.file):
        if args.file.endswith('.tflite'):
            modeil_io = query_tflite_model_io_info(args.file)
            compile_options = args.options + modeil_io.recommand_compile_options
            benchmark_tflite_model(args.file, args.target, args.count, compile_options, False, args.profile)
        elif args.file.endswith('.onnx'):
            benchmark_onnx_model(args.file, args.count)
    else:
        logging.error('Error to load model %s', args.file)
