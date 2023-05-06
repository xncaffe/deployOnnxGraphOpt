import onnx
import copy
import onnx.numpy_helper
import onnx.helper
import numpy as np
from onnx import TensorProto

NPDTYPE_2_ONNXDTYPE = {
    'float32':  TensorProto.FLOAT,      #index = 1
    'uint8':    TensorProto.UINT8,      #index = 2
    'int8':     TensorProto.INT8,       #index = 3
    'uint16':   TensorProto.UINT16,     #index = 4
    'int16':    TensorProto.INT16,      #index = 5
    'int32':    TensorProto.INT32,      #index = 6
    'int64':    TensorProto.INT64,      #index = 7
    'object':   TensorProto.STRING,     #index = 8
    '<U0':      TensorProto.STRING,     #index = 8
    'bool':     TensorProto.BOOL,       #index = 9
    'float16':  TensorProto.FLOAT16,    #index = 10
    'float64':  TensorProto.DOUBLE,     #index = 11
    'uint32':   TensorProto.UINT32,     #index = 12
    'uint64':   TensorProto.UINT64,     #index = 13
    np.dtype(np.float32):   TensorProto.FLOAT,      #index = 1
    np.dtype(np.uint8):     TensorProto.UINT8,      #index = 2
    np.dtype(np.int8):      TensorProto.INT8,       #index = 3
    np.dtype(np.uint16):    TensorProto.UINT16,     #index = 4
    np.dtype(np.ushort):    TensorProto.UINT16,     #index = 4
    np.dtype(np.short):     TensorProto.INT16,      #index = 5
    np.dtype(np.int16):     TensorProto.INT16,      #index = 5
    np.dtype(np.int32):     TensorProto.INT32,      #index = 6
    np.dtype(np.int64):     TensorProto.INT64,      #index = 7
    #np.dtype(np.int):      TensorProto.INT64,      #index = 7
    np.dtype(np.str_):      TensorProto.STRING,     #index = 8
    #np.dtype(np.bool):     TensorProto.BOOL,       #index = 9
    np.dtype(np.bool_):     TensorProto.BOOL,       #index = 9
    np.dtype(np.bool8):     TensorProto.BOOL,       #index = 9
    np.dtype(np.float16):   TensorProto.FLOAT16,    #index = 10
    #np.dtype(np.float):    TensorProto.DOUBLE,     #index = 11
    np.dtype(np.float64):   TensorProto.DOUBLE,     #index = 11
    np.dtype(np.uint32):    TensorProto.UINT32,     #index = 12
    np.dtype(np.uint64):    TensorProto.UINT64,     #index = 13
    np.dtype(np.uint):      TensorProto.UINT64,     #index = 13
}

ONNXDTYPE_2_NPDTYPE = {
    TensorProto.FLOAT:  np.dtype(np.float32),
    TensorProto.UINT8:  np.dtype(np.uint8),
    TensorProto.INT16:  np.dtype(np.int8),
    TensorProto.UINT16: np.dtype(np.uint16),
    TensorProto.UINT32: np.dtype(np.uint32),
    TensorProto.INT32:  np.dtype(np.int32),
    TensorProto.INT64:  np.dtype(np.int64),
    TensorProto.UINT64: np.dtype(np.uint64),
    TensorProto.DOUBLE: np.dtype(np.float64),
}

ONNXDTYPE = {
    1:  TensorProto.FLOAT,      #index = 1
    2:  TensorProto.UINT8,      #index = 2
    3:  TensorProto.INT8,       #index = 3
    4:  TensorProto.UINT16,     #index = 4
    5:  TensorProto.INT16,      #index = 5
    6:  TensorProto.INT32,      #index = 6
    7:  TensorProto.INT64,      #index = 7
    8:  TensorProto.STRING,     #index = 8
    9:  TensorProto.BOOL,       #index = 9
    10: TensorProto.FLOAT16,    #index = 10
    11: TensorProto.DOUBLE,     #index = 11
    12: TensorProto.UINT32,     #index = 12
    13: TensorProto.UINT64,     #index = 13
}

def delete_initializer_by_name(onnx_model, name):
    initialCp = copy.deepcopy(onnx_model.graph.initializer)
    for init in initialCp:
        if init.name == name:
            onnx_model.graph.initializer.remove(init)
    return onnx_model

def find_other_node_by_input(onnx_model, n_node, input):
    for node in onnx_model.graph.node:
        if input in node.input and node.name != n_node.name:
            return True
    return False

def get_value_info_by_name(onnx_model, name):
    for x in onnx_model.graph.input:
        if x.name == name:
            return x
    for x in onnx_model.graph.output:
        if x.name == name:
            return x
    for x in onnx_model.graph.value_info:
        if x.name == name:
            return x
    return None

def find_input_from_initializer(onnx_model, n_node):
    reInputs = {}
    for in_id, input in enumerate(n_node.input):
        for initial in onnx_model.graph.initializer:
            if input == initial.name:
                reInputs[input] = in_id
    return reInputs          

def delete_useless_inputOfModel(onnx_model):
    onnx_modelCp = copy.deepcopy(onnx_model)
    for input in onnx_modelCp.graph.input:
        if not get_node_by_input(onnx_model, [input.name]):
            onnx_model.graph.input.remove(input)
    del onnx_modelCp
    return onnx_model

def get_dtype_by_name(onnx_model, name):
    graph_input = onnx_model.graph.input
    for value in graph_input:
        if value.name == name:
            return value.type.tensor_type.elem_type     
    
    value_info = onnx_model.graph.value_info
    for value in value_info:
        if value.name == name:
            return value.type.tensor_type.elem_type
    
    graph_output = onnx_model.graph.output
    for value in graph_output:
        if value.name == name:
            return value.type.tensor_type.elem_type

    initial = onnx_model.graph.initializer
    for value in initial:
        if value.name == name:
            return value.data_type
    
    return None

def insert_node_by_list(onnx_model, nodes_list, index):
    for node in nodes_list:
        onnx_model.graph.node.insert(index, node)
    return onnx_model  

def find_init_by_name(onnx_model, name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return True
    return False

def delete_value_info_by_name(onnx_model, name):
    for value_info in onnx_model.graph.value_info:
        if value_info.name == name:
            onnx_model.graph.value_info.remove(value_info)
            return onnx_model
    return onnx_model

def delete_useless_value_info(onnx_model):
    value_infosCp = copy.deepcopy(onnx_model.graph.value_info)
    for value_info in value_infosCp:
        delFlag = True
        for node in onnx_model.graph.node:
            if value_info.name in node.input:
                delFlag = False
                break
        if delFlag:
            onnx_model.graph.value_info.remove(value_info)
    return onnx_model

def delete_useless_input_in_initializer(onnx_model):
    ini_to_keep_list = []
    init_need_remove = []
    input_need_remove = []
    ini_remove_num = 0
    for node in onnx_model.graph.node:
        ini_to_keep_list.extend(node.input)
    for init in onnx_model.graph.initializer:
        if init.name not in ini_to_keep_list and init.name not in init_need_remove:
            init_need_remove.append(init)
            ini_remove_num += 1
    for input_info in onnx_model.graph.input:
        if input_info.name not in ini_to_keep_list:
            input_need_remove.append(input_info)
    for i in init_need_remove:
        onnx_model.graph.initializer.remove(i)
    for i in input_need_remove:
        onnx_model.graph.input.remove(i)
    return onnx_model

def delete_nodes(onnx_model, delete_nodes):
    for delete_node in delete_nodes:
        onnx_model.graph.node.remove(delete_node)
    return onnx_model

def bytes_to_str(s):
    if isinstance(s, bytes):
        return s.decode()
    return s

def attribute_to_dict(attribute):
    attri_dict = {}
    for att in attribute:
        value = bytes_to_str(onnx.helper.get_attribute_value(att))
        if isinstance(value, list):
            value = [bytes_to_str(item) for item in value]
        attri_dict[att.name] = value
    return attri_dict

def get_tensor_from_initializer(onnx_model, name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    for node in onnx_model.graph.node:
        if node.op_type == "Constant" and name == node.output[0]:
            return onnx.numpy_helper.to_array(node.attribute[0].t)
    return np.array([])   

def get_shape_by_name(onnx_model, name):
    graph_input = onnx_model.graph.input
    for value in graph_input:
        if value.name == name:
            dim_list = []
            for dim in value.type.tensor_type.shape.dim:
                dim_list.append(int(dim.dim_value))
            if (len(dim_list) > 0 and dim_list[0] == 0):
                dim_list[0] = 1
            return dim_list
    value_info = onnx_model.graph.value_info
    for value in value_info:
        if value.name == name:
            dim_list = []
            for dim in value.type.tensor_type.shape.dim:
                dim_list.append(int(dim.dim_value))
            if (len(dim_list) > 0 and dim_list[0] == 0):
                dim_list[0] = 1
            return dim_list
    graph_output = onnx_model.graph.output
    for value in graph_output:
        if value.name == name:
            dim_list = []
            for dim in value.type.tensor_type.shape.dim:
                dim_list.append(int(dim.dim_value))
            if (len(dim_list) > 0 and dim_list[0] == 0):
                dim_list[0] = 1
            return dim_list
    tensor = get_tensor_from_initializer(onnx_model, name)
    dim_list = []
    try:
        for s in tensor.shape:
            dim_list.append(int(s))
        return dim_list
    except:
        return [1]

def get_node_id(onnx_model, n_node):
    for id, node in enumerate(onnx_model.graph.node):
        if node.name == n_node.name:
            return id
              
def get_node_by_output(onnx_model, name: str):
    for node in onnx_model.graph.node:
        if name in node.output:
            return node
    return None

def get_node_by_input(onnx_model, input_list):
    nodes = []
    for i in input_list:
        for node in onnx_model.graph.node:
            if i in node.input:
                nodes.append(node)
    return nodes

def get_node_serial_group(onnx_model, node, op_patch_list):
    node_serial_list = []
    for list_index in range(len(op_patch_list)):
        if list_index > 0:
            nodes = get_node_by_input(onnx_model, node.output)
            node = nodes[0]
            assert (len(nodes) == 1)
        if node.op_type == op_patch_list[list_index]:
            node_serial_list.append(node)
    return node_serial_list

def check_node_serial_group(onnx_model, node, op_patch_list):
    for list_index in range(len(op_patch_list)):
        if list_index > 0:
            nodes = get_node_by_input(onnx_model, node.output)
            if len(nodes) != 1:
                return False
            node = nodes[0]
        if node.op_type != op_patch_list[list_index]:
            return False
    return True