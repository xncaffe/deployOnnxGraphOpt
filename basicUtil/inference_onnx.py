import os
import onnx
import copy
import onnx.helper
import numpy as np
import onnxruntime as ort
from collections import OrderedDict

from basicUtil.baseUtil import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[OPTPROC]")

def forward_by_onnxruntime(onnx_model):
    ort_session = ort.InferenceSession(onnx_model.SerializeToString())
    ort_inputs={}
    for net_input_index in range(len(ort_session.get_inputs())):
        net_input=onnx_model.graph.input[net_input_index]
        input_shape = net_input.type.tensor_type.shape.dim
        input_shape_list=[]
        for _i in input_shape:
            val = int(_i.dim_value)
            if type(val) == int:
                if val < 1:
                    val = 1
            else: 
                val = 1
            input_shape_list.append(val)
        input_size = np.array(input_shape_list, dtype=np.int64).prod(dtype=np.int64)
        np.random.seed(1)
        net_input_type = net_input.type.tensor_type.elem_type
        if net_input_type != TensorProto.BOOL:
            img_array = np.array(np.random.random(size=tuple([input_size,])),
                                 dtype=ONNXDTYPE_2_NPDTYPE[net_input_type])
        else:
            img_array = np.random.choice(a=[False, True], size=tuple([input_size,]), p=[0.5, 0.5])
        img_array = np.reshape(img_array, tuple(input_shape_list)) 
        ort_inputs[ort_session.get_inputs()[net_input_index].name]=img_array
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)

    return OrderedDict(zip(outputs, ort_outs)) 

def infer_model_shape(onnx_model):
    onnx_model_all_output=copy.deepcopy(onnx_model)
    onnx_model_shape=copy.deepcopy(onnx_model)
    del onnx_model_shape.graph.value_info[:]
    del onnx_model_all_output.graph.value_info[:]
    #onnx.save(onnx_model_shape, "del_allnode.onnx")
    ori_model_output_list=[]
    for out in onnx_model.graph.output:
        ori_model_output_list.append(out.name)
    for node in onnx_model_all_output.graph.node:
        for output in node.output:
            if output not in ori_model_output_list:
                onnx_model_all_output.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_outs = forward_by_onnxruntime(onnx_model_all_output)
    
    for node in onnx_model_shape.graph.node:
        for output in node.output:
            use_value_info = onnx.helper.make_tensor_value_info(output, 
                                                                NPDTYPE_2_ONNXDTYPE[ort_outs[output].dtype], 
                                                                ort_outs[output].shape)
            onnx_model_shape.graph.value_info.append(use_value_info)
    
    del onnx_model_shape.graph.output[:]
    for tensor_name in ori_model_output_list:
        tensor_info = get_value_info_by_name(onnx_model_all_output, tensor_name)
        tensor_shape = ort_outs[tensor_name].shape
        if tensor_info is not None:
            value_info = onnx.helper.make_tensor_value_info(tensor_name,
                                                            ONNXDTYPE[tensor_info.type.tensor_type.elem_type],
                                                            tensor_shape)
            onnx_model_shape.graph.output.append(value_info)
        else:
            logging.error("Find outpt tensor value info failed!")
    #onnx.save(onnx_model_shape, "add_alloutput_model_shape.onnx")
    return onnx_model_shape

def ort_compare_result(model_old, model_new):
    out_old = forward_by_onnxruntime(model_old)
    out_new = forward_by_onnxruntime(model_new)
    for k in out_old.keys():
        _o_tensor = np.reshape(out_old[k], -1)
        _n_tensor = np.reshape(out_new[k], -1)
        abs_max = np.max(abs(_o_tensor - _n_tensor))
        mean_ref = np.mean(abs(_o_tensor)) / 2 ** 10
        max_ref = np.max(abs(_o_tensor)) / 2 ** 16
        cmp = np.isnan(abs_max)
        cmp |= abs_max > mean_ref and abs_max > max_ref
        cmp &= not (mean_ref == 0 and max_ref == 0 and abs_max < 1e-3)
        if get_node_by_output(model_old, k).op_type in ['Sigmoid', 'Tanh']:
            cmp &= not (abs_max < 1e-6)
        if cmp:
            logging.warning(f"COMPARE FAILURE: {k}, abs_max={abs_max}, mean_ref={mean_ref}, max_ref={max_ref}")
            return False
        logging.info(f"COMPARE SUCCESS: {k}, abs_max={abs_max}, mean_ref={mean_ref}, max_ref={max_ref}")
    return True

def check_opt_precision(model_old, model_new, func_name):
    try:
        cmp_rslt = ort_compare_result(model_old, model_new)
        if cmp_rslt:
            return model_new
        else:
            logging.warning("Failing to execute: ONNXRUNTIME COMPARE!")
            raise ValueError("After the conversion of '{}', the output result has a relatively large error, please check!".format(func_name))
    except Exception as e:
        logging.warning("Failing to execute: ONNXRUNTIME COMPARE!")
        logging.warning(e)
        raise ValueError("After '{}' is converted, the result comparison is complete, and there may be problems with the model".format(func_name))