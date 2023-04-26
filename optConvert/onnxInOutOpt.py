import copy
from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_deleteGatherInput(onnx_model):
    restart = False
    onnx_modelCp = copy.deepcopy(onnx_model)
    for input_index, input in enumerate(onnx_modelCp.graph.input):
        nodes = get_node_by_input(onnx_modelCp, [input.name])
        for node in nodes:
            if node.op_type == "Gather" and \
                (find_init_by_name(onnx_modelCp, node.input[0]) or find_init_by_name(onnx_modelCp, node.input[1])):
                    gather_outShape = get_shape_by_name(onnx_model, node.output[0])
                    onnx_model.graph.node.remove(node)
                    gather_outType = get_dtype_by_name(onnx_model, node.output[0])
                    new_input_value_info = onnx.helper.make_tensor_value_info(node.output[0], gather_outType, gather_outShape)
                    onnx_model.graph.input.append(new_input_value_info)
                    restart = True
    if restart:
        onnx_model = delete_useless_inputOfModel(onnx_model)
    return onnx_model, restart