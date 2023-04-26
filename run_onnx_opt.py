import onnx
import onnx.helper
import numpy as np
import onnxruntime as ort
from functools import wraps

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[OPTPROC]")

from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *
from optConvert.opt_convert_func import *
from optConvert.onnxInOutOpt import *

def convert_dtnamic_batch(onnx_model):
    graph_input = onnx_model.graph.input
    for value in graph_input:
        value.type.tensor_type.shape.dim[0].dim_value = 1
        #value.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    return onnx_model

class OnnxConvertOptimizer(object):
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model
    
    #@classmethod
    def opt(self):
        self.onnx_model = opt_deleteGatherInput(self.onnx_model)
        self.onnx_model = opt_fusionSeparatedLayerNormal(self.onnx_model)
        return self.onnx_model         
        
