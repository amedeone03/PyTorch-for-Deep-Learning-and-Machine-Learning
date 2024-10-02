import argparse
import os
import sys
import re
import numpy as np
import logging
import subprocess

available_backends = []

# You can define the backends manually here
available_backends = ['mdla2.0', 'vpu_fpu', 'tflite_cpu']

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

def query_backends():
    """Sets available backends."""
    for backend in available_backends:
        logging.info(f"Available backend: {backend}")

def tflite_2_dla(tflite_model, target, dla_model, compile_options=""):
    """Runs Neuropilot's ncc-tflite compiler for the target backend."""
    if os.path.exists(dla_model):
        return
    
    batcmd = f'ncc-tflite -arch {target} {tflite_model} -o {dla_model} {compile_options}'
    logging.info(f"Running: {batcmd}")
    res = subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout
    logging.debug(result)

def benchmark_tflite_model(model, target, count=100, compile_options="", cache_dla=False, profile=False):
    """Runs inference on a compiled TFLite model."""
    dla = model.replace('.tflite', f"-{target}.dla")
    tflite_2_dla(model, target, dla, compile_options)
    
    if not os.path.exists(dla):
        logging.error(f"Failed to convert model to DLA with target {target}")
        return

    # Generate random input.bin
    modeil_io = query_tflite_model_io_info(model)
    input_bin = 'input.bin'
    gen_input_bin(modeil_io, input_bin)

    # Run inference
    batcmd = f'neuronrt -m hw -a {dla} -c {count} -b 100 -i {input_bin} '
    for i in range(modeil_io.output_count):
        batcmd += f' -o output_{i}.bin '
    
    logging.info(f"Running inference on {target} for model {model}")

    res = subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout
    logging.debug(result)

    # Calculate average inference time
    match = re.search('Total inference time = (\d+)', result)
    if match:
        total_time = float(match.group(1))
        avg_time = total_time / count
        logging.info(f"{model}, {target}, Avg Inference Time: {avg_time}")
    else:
        logging.error(f"Failed to find inference time in output.")

# Argument parsing and logger setup
query_backends()
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--file', default="", help='Input TFLite model')
parser.add_argument('--target', default=available_backends[0], choices=available_backends, help='Backend for inference')
parser.add_argument('--count', default=100, type=int, help='Number of inference runs (default: 100)')
parser.add_argument('--profile', action='store_true', help='Print profile information')
parser.add_argument('--options', default="", help='Additional NCC-TFLite compile options')

args = parser.parse_args()

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure the model exists and run inference
if os.path.exists(args.file):
    benchmark_tflite_model(args.file, args.target, args.count, args.options, False, args.profile)
else:
    logging.error(f"Model {args.file} not found")

