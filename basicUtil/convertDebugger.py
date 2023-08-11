from functools import wraps

from basicUtil.inference_onnx import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[OPTPROC]")

class OnnxDebuggerMeet(object):
    debug_mode = 'release'
    opset_version = 11
    save_path = ''
    INOUT_SCREEN_FUNC = [
        'opt_deleteGatherInput',
        'opt_mulReplaceWhereBoolInput',
        'opt_deleteUnsqueezeCastLessNotUnSqueezeNotSliceSliceForInput',
        'opt_deleteSqueezeCastReduceSumCastForOutput',
        'opt_replaceInputSqueezeCastEqueezeWhereOrNotWhereWithMul'
    ]
    GENERAL_SCREEN_FUNC = [
        'opt_splitVxSoftmax2DynamicConv',
        'opt_convertViT_attention',
        'opt_convertCustomThrConvKQV',
        'opt_convertMultiKMultiHeadAttentionKQV',
        'opt_convertCalculateTransposeReshapeSoftmax'
    ]
 
    @staticmethod
    def set_debug_mode(debug_mode='release'):
        OnnxDebuggerMeet.debug_mode = debug_mode
    
    @staticmethod
    def get_opset_version(onnx_model):
        OnnxDebuggerMeet.opset_version = onnx_model.opset_import[0].version
    
    @staticmethod
    def get_save_path(save_path):
        OnnxDebuggerMeet.save_path = save_path

    @staticmethod
    def opt_convert_wrapper(func):   
        @wraps(func)
        def loop_run_func(*arg, **kwargs):
            onnx_model = arg[0]
            onnx_model_old = copy.deepcopy(onnx_model)
            restart = True
            do_opt = False
            try:
                while(restart):
                    restart = False
                    for node_index, node in enumerate(onnx_model.graph.node):
                        arg_new = (onnx_model, node, node_index)
                        onnx_model, restart = func(*arg_new, **kwargs)
                        if restart:
                            do_opt = True
                            logger.info("Graph optimization completed --> "+func.__name__+ ", node_name: " + node.name)
                            if OnnxDebuggerMeet.debug_mode == 'debug':
                                onnx_model = infer_model_shape(onnx_model)
                                if OnnxDebuggerMeet.opset_version not in [11, 12]:
                                    check_opt_precision(onnx_model_old, onnx_model, func.__name__) 
                                elif func.__name__ not in OnnxDebuggerMeet.GENERAL_SCREEN_FUNC:
                                    check_opt_precision(onnx_model_old, onnx_model, func.__name__)      
                                onnx_model_old = copy.deepcopy(onnx_model)
                                print('')
                            break
                if not restart and do_opt:
                    if OnnxDebuggerMeet.debug_mode == 'release':
                        onnx_model = infer_model_shape(onnx_model)
                        if OnnxDebuggerMeet.opset_version != 11:
                            check_opt_precision(onnx_model_old, onnx_model, func.__name__) 
                        elif func.__name__ not in OnnxDebuggerMeet.GENERAL_SCREEN_FUNC:
                            check_opt_precision(onnx_model_old, onnx_model, func.__name__)
                        print('')
            except Exception as e:
                debug_path = OnnxDebuggerMeet.save_path[:-5] + '-debug.onnx'
                onnx.save_model(onnx_model, debug_path)
                logging.info('Save debug model to %s'%debug_path)
                logging.warning(e)
                raise ValueError("'{}' converted failed, please check it!".format(func.__name__))
            return onnx_model            
        
        return loop_run_func 
    
    @staticmethod
    def opt_inout_wrapper(func):   
        @wraps(func)
        def loop_run_func(*arg, **kwargs):
            onnx_model = arg[0]
            onnx_model_old = copy.deepcopy(onnx_model)
            restart = True
            while(restart):
                restart = False
                arg_new = (onnx_model, )
                try:
                    onnx_model, restart = func(*arg_new, **kwargs)
                    if restart:
                        logger.info("InputOutput optimization completed --> "+func.__name__) 
                        onnx_model = infer_model_shape(onnx_model)
                        if func.__name__ not in OnnxDebuggerMeet.INOUT_SCREEN_FUNC:
                            check_opt_precision(onnx_model_old, onnx_model, func.__name__)
                        onnx_model_old = copy.deepcopy(onnx_model)
                        print('')
                except Exception as e:
                    debug_path = OnnxDebuggerMeet.save_path[:-5] + '-debug.onnx'
                    onnx.save_model(onnx_model, debug_path)
                    logging.info('Save debug model to %s'%debug_path)
                    raise ValueError("'{}' converted failed, please check it!".format(func.__name__))
            return onnx_model            
        
        return loop_run_func