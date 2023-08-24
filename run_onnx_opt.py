import onnx
import onnx.helper
import numpy as np
import re
import onnxruntime as ort
from functools import wraps
from onnxsim import simplify

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *
from optConvert.opt_convert_func import *
from optConvert.onnxInOutOpt import *
from optConvert.opt_about_transformer_func import *

def convert_dtnamic_batch(onnx_model):
    graph_input = onnx_model.graph.input
    for value in graph_input:
        value.type.tensor_type.shape.dim[0].dim_value = 1
        #value.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    return onnx_model

def delete_node_description(onnx_model):
    for node in onnx_model.graph.node:
        node.doc_string = ""
    return onnx_model
    
def refine_tensor_name(onnx_model):
    re_str = r"[\/\\\:\*\?\"\<\>\|]"
    for index, node in enumerate(onnx_model.graph.node):
        if len(node.name) == 0:
            node.name = node.op_type + "_%d"%index
        node.name = re.sub(re_str, "_", node.name) if re.search(re_str, node.name) else node.name
        for id, input in enumerate(node.input):
            node.input[id] = re.sub(re_str, "_", input) if re.search(re_str, input) else input
        for id, output in enumerate(node.output):
            node.output[id] = re.sub(re_str, "_", output) if re.search(re_str, output) else output
    for initial in onnx_model.graph.initializer:
        initial.name = re.sub(re_str, "_", initial.name) if re.search(re_str, initial.name) else initial.name
    for value_info in onnx_model.graph.value_info:
        value_info.name = re.sub(re_str, "_", value_info.name) if re.search(re_str, value_info.name) else value_info.name
    for net_input in onnx_model.graph.input:
        net_input.name = re.sub(re_str, "_", net_input.name) if re.search(re_str, net_input.name) else net_input.name
    for net_output in onnx_model.graph.output:
        net_output.name = re.sub(re_str, "_", net_output.name) if re.search(re_str, net_output.name) else net_output.name
    return onnx_model

def model_preprocess(onnx_model):
    logger = logging.getLogger("[PreProcess]")
    logger.info("Start convert dynamic batch ...")
    onnx_model = convert_dtnamic_batch(onnx_model)
    logger.info("Convert dynamic batch finish.")
    logger.info("Start simplifier before graph optimization ...")
    onnx_model, check = simplify(onnx_model)
    logger.info("Finish simplifier before graph optimization")
    logger.info("Start delete description from node ...")
    onnx_model = delete_node_description(onnx_model)
    logger.info("Finish delete description from node.")
    logger.info("Start refine onnx name ...")
    onnx_model = refine_tensor_name(onnx_model)
    logger.info("Finish refine onnx name.")
    return onnx_model   

class OnnxConvertOptimizer(object):
    def __init__(self, onnx_model, debug_mode, save_path='./result.onnx'):
        self.onnx_model = onnx_model
        self.debug_mode = debug_mode
        self.save_path = save_path
    
    #@classmethod
    def opt(self):
        OnnxDebuggerMeet.set_debug_mode(self.debug_mode)
        OnnxDebuggerMeet.get_opset_version(self.onnx_model)
        OnnxDebuggerMeet.get_save_path(self.save_path)
        
        
        self.onnx_model = opt_deleteGatherInput(self.onnx_model)
        self.onnx_model = opt_mulReplaceWhereBoolInput(self.onnx_model)
        self.onnx_model = opt_deleteUselessReshapeSlice(self.onnx_model)

        self.onnx_model = opt_deleteUnsqueezeCastLessNotUnSqueezeNotSliceSliceForInput(self.onnx_model)
        self.onnx_model = opt_deleteSqueezeCastReduceSumCastForOutput(self.onnx_model)
        self.onnx_model = opt_replaceInputSqueezeCastEqueezeWhereOrNotWhereWithMul(self.onnx_model)
        if self.onnx_model.opset_import[0].version >= 17:
            self.onnx_model = opt_fusionSeparatedLayerNormal(self.onnx_model)
        self.onnx_model = opt_replaceWhere(self.onnx_model)
        self.onnx_model = opt_fusionMultiMulDiv(self.onnx_model)
        self.onnx_model = opt_replaceDivByMul(self.onnx_model)
        self.onnx_model = opt_fusionMultiSubReduceMean(self.onnx_model)
        self.onnx_model = opt_convertConv1DTo2D(self.onnx_model)
        self.onnx_model = opt_convertMultiKMultiHeadAttentionKQV(self.onnx_model)
        self.onnx_model = opt_convert3dimMultiAttentionKQVTo4dim(self.onnx_model)
        self.onnx_model = opt_splitMatMulQK2DynamicConv(self.onnx_model)
        self.onnx_model = opt_splitVxSoftmax2DynamicConv(self.onnx_model)
        self.onnx_model = opt_3dimMultiAttentionxWto4dimConv(self.onnx_model)
        self.onnx_model = opt_fusionTransposeTranspose(self.onnx_model)
        self.onnx_model = opt_fusionMultiBranchReshapeTranspose(self.onnx_model)
        self.onnx_model = opt_3dimFeedForwardTo4dim(self.onnx_model)
        self.onnx_model = opt_convertCalculateTransposeReshapeMul(self.onnx_model)
        self.onnx_model = opt_3dimResidualAddTo4dim(self.onnx_model)
        self.onnx_model = opt_transposeReshape3dimAddTo4dimAdd(self.onnx_model)
        self.onnx_model = opt_3dimLayerNormalTo4dim(self.onnx_model)
        self.onnx_model = opt_convert3DimTransposeXReshapeOrReshapeXTransposeXMulOrAdd(self.onnx_model)
        #self.onnx_model = opt_convert3DimReshapeMulOrAddTranspose(self.onnx_model)
        self.onnx_model = opt_fusionMaskMulTranspose(self.onnx_model)
        self.onnx_model = opt_moveForwardInputReshapeTranspose(self.onnx_model)
        self.onnx_model = opt_fusionTransposeReshapeReshapeTranspose(self.onnx_model)
        self.onnx_model = opt_fusionConvConvAdd(self.onnx_model)
        self.onnx_model = opt_fusionMultiConcat(self.onnx_model)
        
        self.onnx_model = opt_fusionTransposeReshapeTransposeReshapeTranspose(self.onnx_model)
        self.onnx_model = opt_convertViT_attention(self.onnx_model)
        self.onnx_model = opt_convertMultiBatchConvToOneBatch(self.onnx_model)
        self.onnx_model = opt_convertMultiBatchSplit2OneBatchSliceConcat(self.onnx_model)
        self.onnx_model = opt_convertMultiBatchAddReshapeToOneBatchReshapeAdd(self.onnx_model)
        self.onnx_model = opt_convertMultiBatchReshapeConcatReshapeToOneBatchSliceConcat(self.onnx_model)
        self.onnx_model = opt_convertMultiBatchReshapeSliceReshapeToOneBatchSliceConcat(self.onnx_model)
        
        self.onnx_model = opt_convertCustomThrConvKQV(self.onnx_model)
        self.onnx_model = opt_replaceReshapeReduceMean2Conv(self.onnx_model)
        
        self.onnx_model = opt_fusionTransposeReshapeSMishReshapeTranspose(self.onnx_model)
        self.onnx_model = opt_fusionMultiTransposeReshapeXMultiReshapeTranspose(self.onnx_model)
        self.onnx_model = opt_fusionReshapeSplitSMishReshape(self.onnx_model)
        self.onnx_model = opt_convertCalculateTransposeReshapeSoftmax(self.onnx_model)
        self.onnx_model = opt_fusionTransposeReshapeTransposeReshapeOrReverse(self.onnx_model)
        self.onnx_model = opt_convertSliceConcatTranspose2ConcatTransposeConv(self.onnx_model)
        self.onnx_model = opt_moveForwardUnsqueeze(self.onnx_model)
        self.onnx_model = opt_moveForwardTranspose(self.onnx_model)
        
        self.onnx_model = opt_3dimInputReshapeTo4dim(self.onnx_model)
        self.onnx_model = opt_fusionInputTranspose(self.onnx_model)
        
        self.onnx_model = opt_fusionConcatSlice(self.onnx_model)
        self.onnx_model = opt_convertMSliceConcatNxMSlice(self.onnx_model)
        self.onnx_model = opt_convertConcatNxSliceAddToNxAdd(self.onnx_model)
        self.onnx_model = opt_convertInputW1ToH1(self.onnx_model)

        return self.onnx_model         
        
