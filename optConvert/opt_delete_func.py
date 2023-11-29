from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_deleteUselessReshapeExpand(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Reshape', 'Expand']):
        reshape_node, expand_node = get_node_serial_group(onnx_model, node, ['Reshape', 'Expand'])
        reshape_in_shape = get_shape_by_name(onnx_model, reshape_node.input[0])
        expand_out_shape = get_shape_by_name(onnx_model, expand_node.output[0])
        if reshape_in_shape != expand_out_shape:
            return onnx_model, False
        block_out_nodes_list = get_node_by_input(onnx_model, expand_node.output)
        for block_out_node in block_out_nodes_list:
            for idx, cur_input in enumerate(block_out_node.input):
                block_out_node.input[idx] = reshape_node.input[0] \
                    if cur_input == expand_node.output[0] else cur_input
        for cur_node in [expand_node, reshape_node]:
            if not get_node_by_input(onnx_model, cur_node.output):
                onnx_model = delete_value_info_by_name(onnx_model, cur_node.output[0])
                onnx_model.graph.node.remove(cur_node)
            else:
                break
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_deleteUselessShapeSliceSqueezeUnsqueezeSlice(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Shape', 'Slice', 'Squeeze', 'Unsqueeze', 'Slice']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Shape', 'Slice', 'Squeeze', 'Unsqueeze', 'Slice'])
        shape_node, top_slice_node, squeeze_node, unsqueeze_node, bot_slice_node = nodes_serial
        if not find_init_by_name(onnx_model, bot_slice_node.input[0]):
            return onnx_model, False
        block_in_shape = get_shape_by_name(onnx_model, shape_node.input[0])
        if top_slice_node.input[0] != shape_node.output[0]:
            return onnx_model, False
        top_slice_starts_list = get_tensor_from_initializer(onnx_model, top_slice_node.input[1]).tolist()
        top_slice_ends_list = get_tensor_from_initializer(onnx_model, top_slice_node.input[2]).tolist()
        if not top_slice_starts_list or (isinstance(top_slice_starts_list, list) and len(top_slice_starts_list) > 1) \
            or (isinstance(top_slice_ends_list, list) and len(top_slice_starts_list) > 1):
                return onnx_model, False
        top_slice_starts_value = top_slice_starts_list[0] if isinstance(top_slice_starts_list, list) else top_slice_starts_list
        top_slice_ends_value = top_slice_ends_list[0] if isinstance(top_slice_ends_list, list) else top_slice_ends_list
        if len(top_slice_node.input) < 4:
            return onnx_model, False 
        top_slice_axes_list = get_tensor_from_initializer(onnx_model, top_slice_node.input[3]).tolist()
        if not top_slice_axes_list or (isinstance(top_slice_axes_list, list) and len(top_slice_axes_list) > 1):
            return onnx_model, False
        top_slice_axes_value = top_slice_axes_list[0] if isinstance(top_slice_axes_list, list) else top_slice_axes_list
        if top_slice_axes_value != 0:
            return onnx_model, False
        top_slice_out_list = block_in_shape[top_slice_starts_value:top_slice_ends_value]
        if not top_slice_out_list or (isinstance(top_slice_out_list, list) and len(top_slice_out_list) > 1):
            return onnx_model, False
        top_slice_out_value = top_slice_out_list[0]
        if bot_slice_node.input[1] == unsqueeze_node.output[0]:
            bot_slice_start_value = top_slice_out_value
            bot_slice_ends_list = get_tensor_from_initializer(onnx_model, bot_slice_node.input[2]).tolist()
            if not bot_slice_ends_list or (isinstance(bot_slice_ends_list, list) and len(bot_slice_ends_list) > 1):
                return onnx_model, False
            bot_slice_end_value = bot_slice_ends_list[0] if isinstance(bot_slice_ends_list, list) else bot_slice_ends_list
        elif bot_slice_node.input[2] == unsqueeze_node.output[0]:
            bot_slice_end_value = top_slice_out_value
            bot_slice_starts_list = get_tensor_from_initializer(onnx_model, bot_slice_node.input[1]).tolist()
            if not bot_slice_starts_list or (isinstance(bot_slice_starts_list, list) and len(bot_slice_starts_list) > 1):
                return onnx_model, False
            bot_slice_start_value = bot_slice_starts_list[0] if isinstance(bot_slice_starts_list, list) else bot_slice_starts_list
        else:
            return onnx_model, False
        if len(bot_slice_node.input) <= 3:
            return onnx_model, False
        bot_slice_axes_list = get_tensor_from_initializer(onnx_model, bot_slice_node.input[3]).tolist()
        if not bot_slice_axes_list or (isinstance(bot_slice_axes_list, list) and len(bot_slice_axes_list) > 1):
            return onnx_model, False
        bot_slice_axis_value = bot_slice_axes_list[0] if isinstance(bot_slice_axes_list, list) else bot_slice_axes_list
        if len(bot_slice_node.input) == 5:
            bot_slice_steps_list = get_tensor_from_initializer(onnx_model, bot_slice_node.input[4]).tolist()
            if isinstance(bot_slice_steps_list, list) and len(bot_slice_steps_list) > 1:
                return onnx_model, False
            bot_slice_step_value = bot_slice_steps_list
            if not bot_slice_steps_list:
                bot_slice_step_value = None
            elif isinstance(bot_slice_steps_list, list):
                bot_slice_step_value = bot_slice_steps_list[0]
            if bot_slice_step_value != None and bot_slice_step_value != 1:
                return onnx_model, False
        bot_slice_data_arr = get_tensor_from_initializer(onnx_model, bot_slice_node.input[0])
        bot_slice_output_arr = np.split(bot_slice_data_arr, [bot_slice_start_value,bot_slice_end_value], bot_slice_axis_value)[1]
        replace_tensor = get_initial_by_value(onnx_model, bot_slice_output_arr)
        if replace_tensor is None:
            replace_tensor = onnx.helper.make_tensor(name=bot_slice_node.output[0],
                                                    data_type=NPDTYPE_2_ONNXDTYPE[bot_slice_output_arr.dtype],
                                                    dims=bot_slice_output_arr.shape,
                                                    vals=bot_slice_output_arr.flatten().tolist())
            onnx_model.graph.initializer.append(replace_tensor)
        bot_slice_out_nodes_list = get_node_by_input(onnx_model, bot_slice_node.output)
        for cur_node in bot_slice_out_nodes_list:
            for cur_idx, cur_input in enumerate(cur_node.input):
                cur_node.input[cur_idx] = replace_tensor.name if cur_input == bot_slice_node.output[0] else cur_input
        for cur_node in nodes_serial:
            onnx_model = delete_value_info_by_name(onnx_model, cur_node.output[0])
            onnx_model.graph.node.remove(cur_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False
        
@OnnxDebuggerMeet.opt_convert_wrapper
def opt_deleteUselessShapeGatherUnsqueezeConcat(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Shape', 'Gather', 'Unsqueeze', 'Concat']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Shape', 'Gather', 'Unsqueeze', 'Concat'])
        shape_node, gather_node, unsqueeze_node, concat_node = nodes_serial
        block_in_shape = get_shape_by_name(onnx_model, shape_node.input[0])
        if not find_init_by_name(onnx_model, gather_node.input[1]):
            return onnx_model, False
        gather_indices_list = get_tensor_from_initializer(onnx_model, gather_node.input[1]).tolist()
        if isinstance(gather_indices_list, list) and len(gather_indices_list) > 1:
            return onnx_model, False
        gather_index_value = gather_indices_list[0] if isinstance(gather_indices_list, list) else gather_indices_list
        concat_in_value = block_in_shape[gather_index_value]
        concat_output_values_list = [1 for i in concat_node.input]
        for cur_index, concat_input in enumerate(concat_node.input):
            if concat_input == unsqueeze_node.output[0]:
                concat_output_values_list[cur_index] = concat_in_value
            else:
                if not find_init_by_name(onnx_model, concat_input):
                    return onnx_model, False
                cur_concat_values_list = get_tensor_from_initializer(onnx_model, concat_input).tolist()
                if isinstance(cur_concat_values_list, list) and len(cur_concat_values_list) > 1:
                    return onnx_model, False
                concat_output_values_list[cur_index] = cur_concat_values_list[0] if isinstance(cur_concat_values_list, list) else cur_concat_values_list
        replace_tensor = get_initial_by_value(onnx_model, np.array(concat_output_values_list, dtype=np.int64))
        if replace_tensor is None:
            replace_tensor = onnx.helper.make_tensor(name=concat_node.output[0],
                                                     data_type=TensorProto.INT,
                                                     dims=[len(concat_output_values_list)],
                                                     vals=concat_output_values_list)
            onnx_model.graph.initializer.append(replace_tensor)
        next_out_nodes_list = get_node_by_input(onnx_model, concat_node.output)
        for cur_node in next_out_nodes_list:
            for cur_idx, cur_input in enumerate(cur_node.input):
                cur_node.input[cur_idx] = replace_tensor.name if cur_input == concat_node.output[0] else cur_input
        for cur_node in nodes_serial:
            onnx_model = delete_value_info_by_name(onnx_model, cur_node.output[0])
            onnx_model.graph.node.remove(cur_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_deleteUselessExpand(onnx_model, node, node_index): 
    if node.op_type == 'Expand':
        input_shape = get_shape_by_name(onnx_model, node.input[0])
        output_shape = get_shape_by_name(onnx_model, node.output[0])
        next_nodes_list = get_node_by_input(onnx_model, node.output)
        if input_shape == output_shape:
            for next_node in next_nodes_list:
                for next_idx, next_input in enumerate(next_node.input):
                    next_node.input[next_idx] = node.input[0] if next_input == node.output[0] else next_input
            onnx_model = delete_value_info_by_name(onnx_model, node.outout[0])
            onnx_model.graph.node.remove(node)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        else:
            next_muladds_list = [next_node for next_node in next_nodes_list \
                if next_node.op_type in ['Mul', 'Add', 'Sub', 'Div']]
            run_flag = False
            for cal_node in next_muladds_list:
                other_input = cal_node.input[0] if cal_node.input[1] == node.output[0] else cal_node.input[1]
                if other_input == node.output[0]:
                    continue
                other_in_shape = get_shape_by_name(onnx_model, other_input)
                if other_in_shape != output_shape:
                    continue
                diff_num = 0
                for idx in range(len(other_in_shape)):
                    if other_in_shape[idx] != input_shape[idx]:
                        diff_num += 1
                if diff_num > 1:
                    continue
                cal_node.input[list(cal_node.input).index(node.output[0])] = node.input[0]
                run_flag = True
            if run_flag:
                new_next_nodes_list = get_node_by_input(onnx_model, node.output)
                if not new_next_nodes_list:
                    onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
                    onnx_model.graph.node.remove(node)
                onnx_model = delete_useless_input_in_initializer(onnx_model)
                return onnx_model, True
            else:
                return onnx_model, False
    return onnx_model, False