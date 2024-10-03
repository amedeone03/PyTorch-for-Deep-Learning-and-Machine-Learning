import argparse
import os
import sys
import re
import numpy as np
import logging
import subprocess

available_backends = []

DIRTY = 1
CLEAR_DIRTY = 0

TIME = 'avg inference time:'
INIT_TIME = TIME + ' FAILED'

class ModelIO:
    def __init__(self):
        self.intput_size = (-1, -1, -1, -1)
        self.intput_type = 'unknown'
        self.output_count = 0
        self.recommand_compile_options = ''

def query_backends():
    # Manually set backends to mdla2.0 and vpu_fpu
    available_backends.extend(['mdla2.0', 'vpu_fpu'])
    logging.info(f"Available backends: {available_backends}")

def query_tflite_model_io_info(model):
    # find input size and output count of tflite model
    ret = ModelIO()

    intput_size = (-1, -1, -1, -1)
    intput_type = 'kTfLiteUInt8'
    output_count = 0
    batcmd = ('ncc-tflite --show-io-info %s' % model)
    res = subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout

    logging.debug(result)

    # find output count
    match = re.search('of output tensors: (\d+)', result)
    if match:
        output_count = int(match.group(1))
    else:
        logging.error("FAIL to find output count of model")
        return ret

    ret.output_count = output_count

    # find input size
    tmp = result.partition("# of output tensors")[0]
    match = re.search('Shape: {(\d+),(\d+),(\d+),(\d+)}', tmp)
    if match:
        intput_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
    else:
        logging.error("FAIL to find input size of model")
        return ret

    ret.intput_size = intput_size

    # find input type
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
    # generate input data, format can be uint8 or float32
    ndarray = np.random.randint(0, high=255, size=model_io.intput_size, dtype=np.uint8)
    if model_io.intput_type == 'kTfLiteFloat32':
        ndarray = np.float32(ndarray) / 255.0
    ndarray.tofile(filename)

def tflite_2_dla(tflite_model, target, dla_model, compile_options=""):
    # compile for specific target
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

    # find input size and output count of model
    model_io = query_tflite_model_io_info(model)

    # gen rand input.bin
    input_bin = 'input.bin'
    gen_input_bin(model_io, input_bin)

    batcmd = 'neuronrt -m hw -a %s -c %s -b 100 -i %s ' % (dla, count, input_bin)
    for i in range(model_io.output_count):
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

    # find 'Total inference time' in output log
    match = re.search('Total inference time = (\d+)', result)
    if match:
        time = float(match.group(1))
        # calculate avg inference time
        avg_time = time / count
    else:
        logging.error("FAIL to find avg inference time of model")
        return

    logging.info('%s, %s, avg inference time: %s' % (model, target, avg_time))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Query available backends (mdla2.0 and vpu_fpu)
query_backends()

# Argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
        '--file',
        default="yolov5x_quantized.tflite",  # Default to quantized model
        help='Input a tflite model to inference. Default: yolov5x_quantized.tflite')

parser.add_argument(
        '--target',
        default=available_backends[0],
        choices=available_backends,
        help='Choose a backend to inference. Default: %s' % available_backends[0])

parser.add_argument(
        '--count',
        default=100,
        type=int,
        help='How many times of inference. Default: 100')

parser.add_argument(
        '--profile',
        action='store_true',
        help='print profile information')

parser.add_argument(
        '--options',
        default="",
        help='Add additional ncc-tflite options. Please find options by \'ncc-tflite -h\'\n'
             'Ex. --options=\'--relax-fp32\'')

args = parser.parse_args()

# Run inference with both mdla2.0 and vpu_fpu
for backend in ['mdla2.0', 'vpu_fpu']:
    logging.info(f"Running inference on {backend}")
    benchmark_tflite_model(args.file, backend, args.count, args.options, profile=args.profile)
