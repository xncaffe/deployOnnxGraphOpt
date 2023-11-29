import argparse
import onnx
import onnx.helper
import onnx.version_converter
from onnxsim import simplify
import copy

import sys
sys.path.append('./basicUtil')
from baseUtil import *
# from ..basicUtil.baseUtil import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", type=str, default="/workspace/nxu/customer/gelingshentong/20230828/conformer_6017_preopt.onnx", help="input onnx model path")
    parser.add_argument("-o", "--output_model", type=str, default="/workspace/nxu/customer/gelingshentong/20230828/conformer_6017_preopt_softmax-axis-1.onnx", help="output onnx model path")
    parser.add_argument("-v", "--set_opset_version", type=int, default=11, help="Target opset version")
    parser.add_argument("-t", "--tensor_names", nargs='+', type=str, default=[], help="A list of names of nodes to delete")
    parser.add_argument("-u", "--untruncated_branch", nargs='+', type=str, default=[], help="Branches that need to be kept during truncation, in list format, fill in node name")
    parser.add_argument("-s", "--uniform_split_output", nargs='+', type=str, default=[], help="Uniform split output")
    parser.add_argument("-a", "--split_output_axes", nargs='+', type=int, default=[], help="Split output axis")
    parser.add_argument("--convert_opset", action='store_true', help="whether to convert opsert version, defualt no modify")
    parser.add_argument("--convert_softmax", action='store_true', help="When opset equals 11, softmax all four-dimensional inputs with axis=1 to axis=3.")
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
    dst_ir_version = 6 if dst_version <= 11 else 8
    onnx_model.ir_version = dst_ir_version if onnx_model.ir_version != dst_ir_version else onnx_model.ir_version
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
                if curOutNode in list(onnx_model.graph.node):
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
    
uniform_split_output_names = args.uniform_split_output
if uniform_split_output_names:
    split_axes = args.split_output_axes
    for idx, cur_output_name in enumerate(uniform_split_output_names):
        cur_axis = split_axes[idx] if len(split_axes) > idx else 1
        cur_output_shape = get_shape_by_name(onnx_model, cur_output_name)
        cur_split_num = cur_output_shape[cur_axis]
        cur_split_output_names = ['_Split_'+cur_output_name+'_output_%d'%i for i in range(cur_split_num)]
        cur_split_node = onnx.helper.make_node(name=cur_output_name+'_split',
                                               op_type='Split',
                                               inputs=[cur_output_name],
                                               outputs=cur_split_output_names,
                                               axis=cur_axis)
        net_output_id = 0
        out_dtype = get_dtype_by_name(onnx_model, cur_output_name)
        for sub_idx, net_output in enumerate(onnx_model.graph.output):
            if net_output.name == cur_output_name:
                onnx_model.graph.output.remove(net_output)
                net_output_id = sub_idx
                break
        onnx_model.graph.node.append(cur_split_node)
        src_output_value_info = onnx.helper.make_tensor_value_info(cur_output_name, out_dtype, cur_output_shape)
        onnx_model.graph.value_info.append(src_output_value_info)
        split_out_shape = copy.deepcopy(cur_output_shape)
        split_out_shape[cur_axis] = 1
        cur_split_output_names.reverse()
        for split_out_name in cur_split_output_names:
            split_out_proto = onnx.helper.make_tensor_value_info(split_out_name, out_dtype, split_out_shape)
            onnx_model.graph.output.append(split_out_proto)
args.convert_softmax = True
if args.convert_softmax and onnx_model.opset_import[0].version <= 12:
    onnx_model_cp = copy.deepcopy(onnx_model)
    for node in onnx_model_cp.graph.node:
        if node.op_type != 'Softmax':
            continue
        cur_in_shape = get_shape_by_name(onnx_model, node.input[0])
        cur_axis = attribute_to_dict(node.attribute).get('axis', -1)
        if len(cur_in_shape) != 4 or cur_axis not in [1, -3]:
            continue
        cur_dtype = get_dtype_by_name(onnx_model, node.input[0])
        ori_softmax_node = get_node_by_output(onnx_model, node.output[0])
        if ori_softmax_node is None or ori_softmax_node.name != node.name:
            continue
        del ori_softmax_node.attribute[:]
        new_axis_attr = onnx.helper.make_attribute('axis', -1)
        ori_softmax_node.attribute.append(new_axis_attr)
        top_transpose_node = onnx.helper.make_node(name=node.name+'_top_transpose',
                                                   op_type='Transpose',
                                                   inputs=node.input,
                                                   outputs=[node.input[0] + '_axis3'],
                                                   perm=[0, 3, 2, 1])
        ori_softmax_node.input[0] = top_transpose_node.output[0]
        ori_softmax_node.output[0] = node.output[0] + '_axis3'
        bot_transpose_node = onnx.helper.make_node(name=node.name+'_bot_transpose',
                                                   op_type='Transpose',
                                                   inputs=[ori_softmax_node.output[0]],
                                                   outputs=[node.output[0]],
                                                   perm=[0, 3, 2, 1])
        cur_softmax_idx = get_node_id(onnx_model, ori_softmax_node)
        onnx_model.graph.node.insert(cur_softmax_idx+1, bot_transpose_node)
        onnx_model.graph.node.insert(cur_softmax_idx, top_transpose_node)
        top_transpose_out_shape = [cur_in_shape[i] for i in [0, 3, 2, 1]]
        top_transpose_out_value = onnx.helper.make_tensor_value_info(top_transpose_node.output[0], cur_dtype, top_transpose_out_shape)
        new_softmax_out_value = onnx.helper.make_tensor_value_info(ori_softmax_node.output[0], cur_dtype, top_transpose_out_shape)
        onnx_model.graph.value_info.append(top_transpose_out_value)
        onnx_model.graph.value_info.append(new_softmax_out_value)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        logger.info("Convert {} from axis=1 to axis=3".format(node.name))

# for idx, net_output in enumerate(onnx_model.graph.output):
#     if net_output.name == "5328_reserve":
#         cur_out_node = get_node_by_output(onnx_model, net_output.name)
#         new_net_output = onnx.helper.make_tensor_value_info('5328', 1, [1, 3, 224, 320])
#         cur_out_node.output[0] = new_net_output.name
#         onnx_model.graph.output.remove(net_output)
#         onnx_model.graph.output.insert(idx, new_net_output)
#         break
        
onnx_model, check = simplify(onnx_model)
logger.info("Saving output model to '%s'"%dstPath)
onnx.save_model(onnx_model, dstPath)
logger.info('Process Finish!')