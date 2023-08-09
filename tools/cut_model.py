import argparse
import onnx
import onnx.helper
import onnx.version_converter
from onnxsim import simplify
import copy

import sys
sys.path.append('./basicUtil')
from baseUtil import *
#from ..basicUtil.baseUtil import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", type=str, default="/workspace/nxu/model/customer/gelingshentong/20230717/encoder_new_preopt.onnx", help="input onnx model path")
    parser.add_argument("-o", "--output_model", type=str, default="/home/nxu/workspace/model/customer/gelingshentong/20230717/encoder_new_preopt_strip.onnx", help="output onnx model path")
    parser.add_argument("-v", "--set_opset_version", type=int, default=11, help="Target opset version")
    parser.add_argument("-t", "--tensor_names", nargs='+', type=str, default=['encoder_out_new', 'onnx__Add_2938_convOut'], help="A list of names of nodes to delete")
    parser.add_argument("-u", "--untruncated_branch", nargs='+', type=str, default=['MatMul_1909_Conv'], help="Branches that need to be kept during truncation, in list format, fill in node name")
    parser.add_argument("--convert_opset", action='store_true', help="whether to convert opsert version, defualt no modify")
    args = parser.parse_args()
    return args

def create_tensor_without_outconnect_to_netoutput(onnx_model):
    netOutNames = [netOutput.name for netOutput in onnx_model.graph.output]
    for node in onnx_model.graph.node:
        for output in node.output:
            outNodesList = get_node_by_input(onnx_model, [output])
            if outNodesList or output in netOutNames:
                continue
            outShape = get_shape_by_name(onnx_model, output)
            outDtype = get_dtype_by_name(onnx_model, output)
            newNetOut = onnx.helper.make_tensor_value_info(output, outDtype, outShape)
            onnx_model.graph.output.append(newNetOut)
            netOutNames.append(output)
    return onnx_model

def convert_opset_version(onnx_model, dst_version):
    onnx_model = onnx.version_converter.convert_version(onnx_model, dst_version)
    onnx_model.ir_version = 7 if onnx_model.ir_version > 7 else onnx_model.ir_version
    return onnx_model

args = parse_args()
srcPath = args.input_model
dstPath = args.output_model
dst_opset_version = args.set_opset_version
tensor_names_list = args.tensor_names
kept_branchs_list = args.untruncated_branch
convert_opset = args.convert_opset

logger = logging.getLogger("[ToolCutModel]")
logger.info('Start chopping the model based on the specified information ...')
onnx_model = onnx.load_model(srcPath)
onnx_modelCp = copy.deepcopy(onnx_model)
netOutNames = [netOutput.name for netOutput in onnx_modelCp.graph.output]
for node in onnx_modelCp.graph.node:
    for output in node.output:
        if output not in tensor_names_list:
            continue
        logger.info('Now, processing %s'%output)
        cur_shape = get_shape_by_name(onnx_modelCp, output)
        curOutNodesList = get_node_by_input(onnx_modelCp, [output])
        while curOutNodesList:
            curOutsList = []
            for curOutNode in curOutNodesList:
                if curOutNode.name in kept_branchs_list:
                    continue
                curOutsList += list(curOutNode.output)
                onnx_model.graph.node.remove(curOutNode)
            curOutNodesList = get_node_by_input(onnx_modelCp, curOutsList)
        outDtype = get_dtype_by_name(onnx_modelCp, output)
        output_value_info = onnx.helper.make_tensor_value_info(output, outDtype, cur_shape)
        onnx_model.graph.output.append(output_value_info)
        
onnx_model = delete_useless_outputOfModel(onnx_model)
onnx_model = create_tensor_without_outconnect_to_netoutput(onnx_model)
onnx_model = delete_useless_input_in_initializer(onnx_model)
onnx_model = delete_useless_value_info(onnx_model)
logger.info('Finish chop the model!')

if convert_opset:
    logger.info('Start converting opset version for the model ...')
    onnx_model = convert_opset_version(onnx_model, dst_opset_version)
    logger.info('Finish convert opset version!')

onnx_model, check = simplify(onnx_model)
logger.info("Saving output model to '%s'"%dstPath)
onnx.save_model(onnx_model, dstPath)
logger.info('Process Finish!')