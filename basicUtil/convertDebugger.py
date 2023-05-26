from functools import wraps

from basicUtil.inference_onnx import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[OPTPROC]")

class OnnxDebuggerMeet(object):
    @staticmethod
    def opt_convert_wrapper(func):   
        @wraps(func)
        def loop_run_func(*arg, **kwargs):
            onnx_model = arg[0]
            restart = True
            while(restart):
                restart = False
                for node_index, node in enumerate(onnx_model.graph.node):
                    arg_new = (onnx_model, node, node_index)
                    onnx_model, restart = func(*arg_new, **kwargs)
                    if restart:
                        logger.info("Graph optimization completed --> "+func.__name__+ ", node_name: " + node.name)
                        break
            if not restart:
                onnx.save_model(onnx_model, "/workspace/nxu/project/Transformer/annotated-transformer/annatatedTransformer-multi30k-opt2.onnx")
                onnx_model = infer_model_shape(onnx_model)
            return onnx_model            
        
        return loop_run_func 
    
    @staticmethod
    def opt_inout_wrapper(func):   
        @wraps(func)
        def loop_run_func(*arg, **kwargs):
            onnx_model = arg[0]
            restart = True
            while(restart):
                restart = False
                arg_new = (onnx_model, )
                onnx_model, restart = func(*arg_new, **kwargs)
                if restart:
                    logger.info("InputOutput optimization completed --> "+func.__name__) 
            return onnx_model            
        
        return loop_run_func