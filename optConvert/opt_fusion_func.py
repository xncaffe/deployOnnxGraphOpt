from functools import reduce
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

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionReshapeTransposeReshapeTransposeWithIm2Col(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Reshape', 'Transpose', 'Reshape', 'Transpose']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Reshape', 'Transpose', 'Reshape', 'Transpose'])
        top_reshape, top_transpose, mid_reshape, bot_transpose = nodes_serial
        top_reshape_in_shape = get_shape_by_name(onnx_model, top_reshape.input[0])
        if len(top_reshape_in_shape) != 4 or top_reshape_in_shape[0] != 1:
            return onnx_model, False
        top_reshape_out_shape = get_shape_by_name(onnx_model, top_reshape.output[0])
        if len(top_reshape_out_shape) < 4 or reduce(lambda x,y:x*y, top_reshape_out_shape[-4:]) != reduce(lambda x,y:x*y, top_reshape_in_shape) \
            or top_reshape_out_shape[-2]*top_reshape_out_shape[-1] != top_reshape_in_shape[-1] \
                or top_reshape_out_shape[-4] != top_reshape_in_shape[-2]//top_reshape_out_shape[-3]*top_reshape_in_shape[-3]:
                    return onnx_model, False
        top_perm = attribute_to_dict(top_transpose.attribute).get('perm', list(range(len(top_reshape_out_shape))).reverse())
        top_perm_compare = [top_perm[i] - (len(top_reshape_out_shape) - 4) for i in range(len(top_perm))]
        if top_perm_compare[-4:] != [0, 2, 1, 3]:
            return onnx_model, False
        mid_reshape_in_shape = get_shape_by_name(onnx_model, mid_reshape.input[0])
        mid_reshape_out_shape = get_shape_by_name(onnx_model, mid_reshape.output[0])
        if len(mid_reshape_out_shape) < 3 or mid_reshape_in_shape[-2]*mid_reshape_in_shape[-1] != mid_reshape_out_shape[-1] \
            or mid_reshape_in_shape[-4]//mid_reshape_out_shape[-3]*mid_reshape_in_shape[-3] != mid_reshape_out_shape[-2]:
                return onnx_model, False
        bot_perm = attribute_to_dict(bot_transpose.attribute).get('perm', list(range(len(mid_reshape_out_shape))).reverse())
        bot_perm_compare = [bot_perm[i] - (len(mid_reshape_out_shape) - 3) for i in range(len(bot_perm))]
        if bot_perm_compare[-3:] != [2, 1, 0]:
            return onnx_model, False

        new_top_shape = top_reshape_in_shape[:2] + [top_reshape_in_shape[-2]//top_reshape_out_shape[-3]] + top_reshape_out_shape[-3:]
        new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_top_shape, dtype=np.int64))
        if new_top_shape_tensor is None:
            new_top_shape_tensor_name = get_unique_node_tensor_name(onnx_model, top_reshape.input[1]+'_new')
            new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_tensor_name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(new_top_shape)],
                                                           vals=new_top_shape)
            onnx_model.graph.initializer.append(new_top_shape_tensor)
        new_bot_shape = get_shape_by_name(onnx_model, bot_transpose.output[0])
        bot_out_name = bot_transpose.output[0]
        next_nodes_list = get_node_by_input(onnx_model, bot_transpose.output)
        if len(next_nodes_list) == 1 and next_nodes_list[0].op_type == 'Reshape':
            new_bot_shape = get_shape_by_name(onnx_model, next_nodes_list[0].output[0])
            bot_out_name = next_nodes_list[0].output[0]
            onnx_model.graph.node.remove(next_nodes_list[0])
        new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(new_bot_shape, dtype=np.int64))
        if new_bot_shape_tensor is None:
            new_bot_shape_tensor_name = get_unique_node_tensor_name(onnx_model, mid_reshape.input[1]+'_new')
            new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_tensor_name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(new_bot_shape)],
                                                           vals=new_bot_shape)
            onnx_model.graph.initializer.append(new_bot_shape_tensor)
        new_top_reshape = onnx.helper.make_node(name=top_reshape.name,
                                                op_type='Reshape',
                                                inputs=[top_reshape.input[0], new_top_shape_tensor.name],
                                                outputs=[top_reshape.output[0]])
        new_transpose = onnx.helper.make_node(name=top_transpose.name,
                                              op_type='Transpose',
                                              inputs=[new_top_reshape.output[0]],
                                              outputs=[top_transpose.output[0]],
                                              perm=[0, 3, 5, 2, 4, 1])
        new_bot_reshape = onnx.helper.make_node(name=mid_reshape.name,
                                                op_type='Reshape',
                                                inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                outputs=[bot_out_name])
        for src_node in nodes_serial:
            if bot_out_name != src_node.output[0]:
                onnx_model = delete_value_info_by_name(onnx_model, bot_out_name)
            onnx_model.graph.node.remove(src_node)
        onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], node_index)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeReshapeTransposeReshapeWithCol2Im(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Transpose', 'Reshape']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Transpose', 'Reshape'])
        top_transpose, top_reshape, bot_transpose, bot_reshape = nodes_serial
        top_transpose_in_shape = get_shape_by_name(onnx_model, top_transpose.input[0])
        if len(top_transpose_in_shape) != 4 or top_transpose_in_shape[0] != 1:
            return onnx_model, False
        top_perm = attribute_to_dict(top_transpose.attribute).get('perm', list(range(len(top_transpose_in_shape))).reverse())
        if top_perm != [0, 3, 2, 1]:
            return onnx_model, False
        top_reshape_in_shape = get_shape_by_name(onnx_model, top_reshape.input[0])
        top_reshape_out_shape = get_shape_by_name(onnx_model, top_reshape.output[0])
        if len(top_reshape_out_shape) < 4 or reduce(lambda x,y:x*y, top_reshape_in_shape) != reduce(lambda x,y:x*y, top_reshape_out_shape[-4:]) \
            or top_reshape_in_shape[-1] != top_reshape_out_shape[-2]*top_reshape_out_shape[-1] \
                or top_reshape_out_shape[-4]//top_reshape_in_shape[1]*top_reshape_out_shape[-3] != top_reshape_in_shape[-2]:
                    return onnx_model, False
        bot_perm = attribute_to_dict(bot_transpose.attribute).get('perm', list(range(len(top_reshape_out_shape))).reverse())
        bot_perm_compare = [bot_perm[i] - (len(top_reshape_out_shape) - 4) for i in range(len(bot_perm))]
        if bot_perm_compare[-4:] != [0, 2, 1, 3]:
            return onnx_model, False
        bot_reshape_in_shape = get_shape_by_name(onnx_model, bot_reshape.input[0])
        bot_reshape_out_shape = get_shape_by_name(onnx_model, bot_reshape.output[0])
        if len(bot_reshape_out_shape) != 4 or bot_reshape_out_shape[0] != 1 \
            or bot_reshape_out_shape[-1] != bot_reshape_in_shape[-2]*bot_reshape_in_shape[-1] \
                or bot_reshape_in_shape[-4]//bot_reshape_out_shape[-3]*bot_reshape_in_shape[-3] != bot_reshape_out_shape[-2]:
                    return onnx_model, False
        
        top_in_name = top_transpose.input[0]
        pre_node = get_node_by_output(onnx_model, top_transpose.input[0])
        if pre_node.op_type == 'Reshape':
            top_in_name = pre_node.input[0]
            pre_next_nodes_list = get_node_by_input(onnx_model, pre_node.output)
            if len(pre_next_nodes_list) == 1:
                onnx_model = delete_value_info_by_name(onnx_model, pre_node.output[0])
                node_index = get_node_id(onnx_model, pre_node)
                onnx_model.graph.node.remove(pre_node)
        
        new_top_shape = [top_transpose_in_shape[0]] + top_reshape_out_shape[-2:] \
            + [top_transpose_in_shape[-2]//top_reshape_out_shape[-3], top_reshape_out_shape[-3], top_transpose_in_shape[-1]]
        new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_top_shape, dtype=np.int64))
        if new_top_shape_tensor is None:
            new_top_shape_tensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, top_reshape.input[0]+'_new'),
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(new_top_shape)],
                                                           vals=new_top_shape)
            onnx_model.graph.initializer.append(new_top_shape_tensor)
        new_top_reshape = onnx.helper.make_node(name=top_reshape.name,
                                                op_type='Reshape',
                                                inputs=[top_in_name, new_top_shape_tensor.name],
                                                outputs=[top_reshape.output[0]])
        new_transpose = onnx.helper.make_node(name=bot_transpose.name,
                                              op_type='Transpose',
                                              inputs=[new_top_reshape.output[0]],
                                              outputs=[bot_transpose.output[0]],
                                              perm=[0, 5, 3, 1, 4, 2])
        for src_node in [top_transpose, top_reshape, bot_transpose]:
            onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
            onnx_model.graph.node.remove(src_node)
        onnx_model = insert_node_by_list(onnx_model, [new_transpose, new_top_reshape], node_index)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False
        