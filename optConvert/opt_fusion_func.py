from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionSpecialConvPad(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Conv', 'Pad']):
        conv_node, pad_node = get_node_serial_group(onnx_model, node, ['Conv', 'Pad'])
        conv_attr = attribute_to_dict(conv_node.attribute)
        conv_kernel_shape = conv_attr.get('kernel_shape', [1, 1])
        conv_dilations = conv_attr.get('dilations', [1, 1])
        conv_pads = conv_attr.get('pads', [0, 0, 0, 0])
        conv_strides = conv_attr.get('strides', [1, 1])
        pad_size = get_tensor_from_initializer(onnx_model, pad_node.input[1]).tolist()
        pad_attr = attribute_to_dict(conv_node.attribute)
        pad_mode = pad_attr.get('mode', 'constant')
        if pad_mode != 'constant':
            return onnx_model, False
        if len(pad_node.input) >= 3:
            constant_values_list = get_tensor_from_initializer(onnx_model, pad_node.input[2]).flatten().tolist()
            if constant_values_list != [0] * len(constant_values_list):
                return onnx_model, False
        conv_in_shape = get_shape_by_name(onnx_model, conv_node.input[0])
        pad_nc_list = pad_size[:(len(conv_in_shape)-2)] + pad_size[len(conv_in_shape):-2]
        if pad_nc_list != [0] * len(pad_nc_list):
            return onnx_model, False
        pad_hw_before = pad_size[:len(conv_in_shape)][-2:]
        pad_hw_after = pad_size[-2:]
        for i in range(2):
            if pad_hw_before[i] + pad_hw_after[i] > 0 \
                and (conv_kernel_shape[i-2] != 1 or conv_dilations[i-2] != 1 or conv_strides[i-2] != 1):
                    return onnx_model, False
        new_conv_pads = copy.deepcopy(conv_pads)
        new_conv_pads[(len(new_conv_pads) // 2 - 2)] += pad_hw_before[0]
        new_conv_pads[(len(new_conv_pads) // 2 - 1)] += pad_hw_before[1]
        new_conv_pads[-2] += pad_hw_after[0]
        new_conv_pads[-1] += pad_hw_after[1]
        new_conv_pads_attr = onnx.helper.make_attribute('pads', new_conv_pads)
        if 'pads' in conv_attr:
            pads_attr_idx = list(conv_attr.keys()).index('pads')
            del conv_node.attribute[pads_attr_idx]
            conv_node.attribute.insert(pads_attr_idx, new_conv_pads_attr)
        else:
            conv_node.attribute.append(new_conv_pads_attr)
        onnx_model = delete_value_info_by_name(onnx_model, conv_node.output[0])
        conv_node.output[0] = pad_node.output[0]
        onnx_model.graph.node.remove(pad_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False