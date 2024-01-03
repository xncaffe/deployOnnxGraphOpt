from functools import reduce
from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceMultiReshapeScatterNDWithPadConcat(onnx_model, node, node_index): 
    if node.op_type == 'ScatterND':
        cur_reshape_node = get_node_by_output(onnx_model, node.input[2])
        reshape_nodes_list = []
        scatternd_nodes_list = []
        pad_infos_list = []
        scatternd_out_shape = get_shape_by_name(onnx_model, node.output[0])
        cur_scatternd_node = node
        cur_scatternd_idx = scatternd_out_shape[2] - 1
        while cur_scatternd_idx >= 0:
            if cur_reshape_node is None or cur_reshape_node.op_type != 'Reshape':
                return onnx_model, False
            cur_reshape_in_shape = get_shape_by_name(onnx_model, cur_reshape_node.input[0])
            cur_reshape_out_shape = get_shape_by_name(onnx_model, cur_reshape_node.output[0])
            if len(cur_reshape_in_shape) != 4 or len(cur_reshape_out_shape) != 5 or cur_reshape_out_shape[2] != 1 \
                or cur_reshape_out_shape != cur_reshape_in_shape[:2] + [1] + cur_reshape_in_shape[2:]:
                    return onnx_model, False
            pre_scatternd_out_shape = get_shape_by_name(onnx_model, cur_scatternd_node.input[0])
            if pre_scatternd_out_shape != scatternd_out_shape or not find_init_by_name(onnx_model, cur_scatternd_node.input[1]):
                return onnx_model, False
            cur_scatter_indices_arr = get_tensor_from_initializer(onnx_model, cur_scatternd_node.input[1])
            if len(cur_scatter_indices_arr.shape) != 6 or cur_scatter_indices_arr.shape[:5] != tuple(cur_reshape_out_shape) \
                or cur_scatter_indices_arr.shape[-1] != 5:
                    return onnx_model, False
            dim1_indices_list = list(range(scatternd_out_shape[1]))
            dim3_indices_list = list(range(scatternd_out_shape[3]))
            dim4_indices_list = list(range(scatternd_out_shape[4]))
            if cur_scatter_indices_arr[:, :, :, :, :, 0].max() != 0 or cur_scatter_indices_arr[:, :, :, :, :, 0].min() != 0:
                return onnx_model, False
            pad_info = [0 for i in range(8)]
            if scatternd_out_shape[1] != cur_reshape_out_shape[1] or scatternd_out_shape[3] < cur_reshape_out_shape[3] \
                or scatternd_out_shape[4] < cur_reshape_out_shape[4] \
                    or cur_scatter_indices_arr[0, 0, :, 0, 0, 2].tolist()[0] != cur_scatternd_idx:
                        return onnx_model, False
            axes_h_indices_arr = cur_scatter_indices_arr[0, 0, 0, :, 0, 3]
            axes_w_indices_arr = cur_scatter_indices_arr[0, 0, 0, 0, :, 4]
            axes_c_indices_arr = cur_scatter_indices_arr[0, :, 0, 0, 0, 1]
            if axes_h_indices_arr.min() < 0 or axes_h_indices_arr.max() >= scatternd_out_shape[3] \
                or axes_w_indices_arr.min() < 0 or axes_w_indices_arr.max() >= scatternd_out_shape[4] \
                    or axes_c_indices_arr.tolist() != dim1_indices_list:
                        return onnx_model, False
            if dim3_indices_list[int(axes_h_indices_arr.min()):int(axes_h_indices_arr.max() + 1)] != axes_h_indices_arr.tolist():
                return onnx_model, False
            pad_info[2] = int(axes_h_indices_arr.min())
            pad_info[6] = int(dim3_indices_list[-1] - axes_h_indices_arr.max())
            if dim4_indices_list[int(axes_w_indices_arr.min()):int(axes_w_indices_arr.max() + 1)] != axes_w_indices_arr.tolist():
                return onnx_model, False
            pad_info[3] = int(axes_w_indices_arr.min())
            pad_info[7] = dim4_indices_list[-1] - int(axes_w_indices_arr.max())
            reshape_nodes_list.append(cur_reshape_node)
            scatternd_nodes_list.append(cur_scatternd_node)
            pad_infos_list.append(pad_info)
            
            cur_scatternd_node = get_node_by_output(onnx_model, cur_scatternd_node.input[0])
            cur_scatternd_idx -= 1
            if cur_scatternd_node is None or cur_scatternd_node.op_type != 'ScatterND':
                break
            cur_reshape_node = get_node_by_output(onnx_model, cur_scatternd_node.input[2])
            
        if len(scatternd_nodes_list) != scatternd_out_shape[2] \
            or len(reshape_nodes_list) != scatternd_out_shape[2] \
                or len(pad_infos_list) != scatternd_out_shape[2]:
                    return onnx_model, False
        last_scatternd_node = scatternd_nodes_list[-1]
        if not find_init_by_name(onnx_model, last_scatternd_node.input[0]):
            return onnx_model, False
        last_scatternd_data = get_tensor_from_initializer(onnx_model, last_scatternd_node.input[0]).flatten()
        for cur_data in last_scatternd_data:
            if cur_data != last_scatternd_data[0]:
                return onnx_model, False
        pad_constant_value = last_scatternd_data.tolist()[0]
        # if (last_scatternd_data != np.zeros(last_scatternd_data.shape, dtype=np.float32)).all():
        #     return onnx_model, False          
        concat_inputs = []
        new_pad_nodes_list = []
        for idx, reshape_node in enumerate(reshape_nodes_list):
            pad_values = pad_infos_list[idx]
            if pad_values == [0] * 8:
                concat_inputs.append(reshape_node.input[0])
            else:
                pad_value_tensor = get_initial_by_value(onnx_model, np.array(pad_values, dtype=np.int64))
                if pad_value_tensor is None:
                    pad_value_tensor = onnx.helper.make_tensor(name=scatternd_nodes_list[idx].name+'_pads',
                                                               data_type=TensorProto.INT64,
                                                               dims=[len(pad_values)],
                                                               vals=pad_values)
                    onnx_model.graph.initializer.append(pad_value_tensor)
                pad_node_inputs = [reshape_node.input[0], pad_value_tensor.name]
                pad_constant_tensor = get_initial_by_value(onnx_model, np.array((pad_constant_value), dtype=np.float32))
                if pad_constant_tensor is None:
                    pad_constant_tensor = onnx.helper.make_tensor(name=scatternd_nodes_list[idx].name+'_constant_value',
                                                                    data_type=NPDTYPE_2_ONNXDTYPE[last_scatternd_data.dtype],
                                                                    dims=[],
                                                                    vals=(pad_constant_value,))
                    onnx_model.graph.initializer.append(pad_constant_tensor)
                pad_node_inputs.append(pad_constant_tensor.name)
                pad_node = onnx.helper.make_node(name=scatternd_nodes_list[idx].name+'_repad',
                                                 op_type='Pad',
                                                 inputs=pad_node_inputs,
                                                 outputs=[reshape_node.output[0]],
                                                 mode='constant')
                new_pad_nodes_list.append(pad_node)
                concat_inputs.append(pad_node.output[0])
        concat_inputs.reverse()
        concat_node = onnx.helper.make_node(name=node.name+'_toConcat',
                                            op_type='Concat',
                                            inputs=concat_inputs,
                                            outputs=[node.output[0]+'_4d'],
                                            axis=1)
        tiling_concat_shape = [scatternd_out_shape[0], scatternd_out_shape[2], scatternd_out_shape[1]] + scatternd_out_shape[3:]
        tiling_reshape_tensor = get_initial_by_value(onnx_model, np.array(tiling_concat_shape, dtype=np.int64))
        if tiling_reshape_tensor is None:
            tiling_reshape_tensor = onnx.helper.make_tensor(name=node.name+'_new_shape',
                                                            data_type=TensorProto.INT64,
                                                            dims=[len(tiling_concat_shape)],
                                                            vals=tiling_concat_shape)
            onnx_model.graph.initializer.append(tiling_reshape_tensor)
        tiling_reshape_node = onnx.helper.make_node(name=node.name+'_to_reshape',
                                                    op_type='Reshape',
                                                    inputs=[concat_node.output[0], tiling_reshape_tensor.name],
                                                    outputs=[node.output[0]+'_shuffle_tile'])
        shuffle_transpose_node = onnx.helper.make_node(name=node.name+'_to_transpose',
                                                       op_type='Transpose',
                                                       inputs=[tiling_reshape_node.output[0]],
                                                       outputs=[node.output[0]],
                                                       perm=[0, 2, 1, 3, 4])
        concat_out_shape = [scatternd_out_shape[0], scatternd_out_shape[1]*scatternd_out_shape[2]] + scatternd_out_shape[3:]
        out_dtype = get_dtype_by_name(onnx_model, node.output[0])
        concat_out_value_info = onnx.helper.make_tensor_value_info(concat_node.output[0], out_dtype, concat_out_shape)
        onnx_model.graph.value_info.append(concat_out_value_info)
        tiling_reshape_value_info = onnx.helper.make_tensor_value_info(tiling_reshape_node.output[0], out_dtype, tiling_concat_shape)
        onnx_model.graph.value_info.append(tiling_reshape_value_info)
        onnx_model = insert_node_by_list(onnx_model, [shuffle_transpose_node, tiling_reshape_node, concat_node], node_index)
        new_pad_nodes_list.reverse()
        onnx_model = insert_node_by_list(onnx_model, new_pad_nodes_list, node_index)
        for src_reshape_node in reshape_nodes_list:
            onnx_model = delete_value_info_by_name(onnx_model, src_reshape_node.output[0])
            onnx_model.graph.node.remove(src_reshape_node)
        for src_scatternd_node in scatternd_nodes_list:
            onnx_model = delete_value_info_by_name(onnx_model, src_scatternd_node.output[0])
            onnx_model.graph.node.remove(src_scatternd_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False
            
@OnnxDebuggerMeet.opt_convert_wrapper
def opt_separateReshapeInstanceNormalReshape(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Reshape", "InstanceNormalization", "Reshape"]):
        nodes_list = get_node_serial_group(onnx_model, node, ["Reshape", "InstanceNormalization", "Reshape"]) 
        top_reshape, instance_normal, bot_reshape = nodes_list
        top_in_shape = get_shape_by_name(onnx_model, top_reshape.input[0])
        top_out_shape = get_shape_by_name(onnx_model, top_reshape.output[0])
        bot_out_shape = get_shape_by_name(onnx_model, bot_reshape.output[0])
        if len(top_in_shape) != 4 or bot_out_shape != top_in_shape or \
            top_out_shape[-1] != int(np.prod(np.array(top_in_shape, dtype=np.int32))):
                return onnx_model, False
        instance_normal_eps = attribute_to_dict(instance_normal.attribute).get('epsilon', 1e-5)
        in_scale = get_tensor_from_initializer(onnx_model, instance_normal.input[1])
        in_bais = get_tensor_from_initializer(onnx_model, instance_normal.input[2])
        ly_reducemean_0_inputs = [top_reshape.input[0]]
        if onnx_model.opset_import[0].version >= 18:
            ly_reduce_mean_0_axes_tensor = get_initial_by_value(onnx_model, np.array([1, 2, 3], dtype=np.int64))
            if ly_reduce_mean_0_axes_tensor is None:
                ly_reduce_mean_0_axes_tensor = onnx.helper.make_tensor(name=instance_normal.name+'_reducemean_axes_0',
                                                                       data_type=TensorProto.INT64,
                                                                       dims=[3],
                                                                       vals=[1, 2, 3])
                onnx_model.graph.initializer.append(ly_reduce_mean_0_axes_tensor)
            ly_reducemean_0_inputs.append(ly_reduce_mean_0_axes_tensor.name)
        ly_reduce_mean_0 = onnx.helper.make_node(name=instance_normal.name+'_layernormal_reducemean_0',
                                               op_type='ReduceMean',
                                               inputs=ly_reducemean_0_inputs,
                                               outputs=[instance_normal.output[0]+'_reducemean_0'])
        if onnx_model.opset_import[0].version < 18:
            ly_reduce_mean_0_attr = onnx.helper.make_attribute('axes', [1, 2, 3])
            ly_reduce_mean_0.attribute.append(ly_reduce_mean_0_attr)
        ly_sub = onnx.helper.make_node(name=instance_normal.name+'_layernormal_sub',
                                       op_type='Sub',
                                       inputs=[top_reshape.input[0], ly_reduce_mean_0.output[0]],
                                       outputs=[instance_normal.output[0]+'_sub'])
        ly_pow = onnx.helper.make_node(name=instance_normal.name+'_layernormal_pow',
                                       op_type='Mul',
                                       inputs=[ly_sub.output[0], ly_sub.output[0]],
                                       outputs=[instance_normal.output[0]+'_pow'])
        ly_reducemean_1_inputs = [ly_pow.output[0]]
        if onnx_model.opset_import[0].version >= 18:
            ly_reduce_mean_1_axes_tensor = get_initial_by_value(onnx_model, np.array([1, 2, 3], dtype=np.int64))
            if ly_reduce_mean_1_axes_tensor is None:
                ly_reduce_mean_1_axes_tensor = onnx.helper.make_tensor(name=instance_normal.name+'_reducemean_axes_1',
                                                                     data_type=TensorProto.INT64,
                                                                     dims=[3],
                                                                     vals=[1, 2, 3])
                onnx_model.graph.initializer.append(ly_reduce_mean_1_axes_tensor)
            ly_reducemean_1_inputs.append(ly_reduce_mean_1_axes_tensor.name)
        ly_reduce_mean_1 = onnx.helper.make_node(name=instance_normal.name+'_layernormal_reducemean_1',
                                                 op_type='ReduceMean',
                                                 inputs=ly_reducemean_1_inputs,
                                                 outputs=[instance_normal.output[0]+'_reducemean_1'])
        if onnx_model.opset_import[0].version < 18:
            ly_reduce_mean_1_attr = onnx.helper.make_attribute('axes', [1, 2, 3])
            ly_reduce_mean_1.attribute.append(ly_reduce_mean_1_attr)
        ly_add_0_eps_tensor = get_initial_by_value(onnx_model, np.array(instance_normal_eps, dtype=np.float32))
        if ly_add_0_eps_tensor is None:
            ly_add_0_eps_tensor = onnx.helper.make_tensor(name=instance_normal.name+'_add_0_eps',
                                                          data_type=NPDTYPE_2_ONNXDTYPE['float32'],
                                                          dims=(),
                                                          vals=[instance_normal_eps])
            onnx_model.graph.initializer.append(ly_add_0_eps_tensor)
        ly_add_0 = onnx.helper.make_node(name=instance_normal.name+'_add_0',
                                         op_type='Add',
                                         inputs=[ly_reduce_mean_1.output[0], ly_add_0_eps_tensor.name],
                                         outputs=[ly_reduce_mean_1.output[0]+'_add_0'])
        ly_sqrt = onnx.helper.make_node(name=instance_normal.name+'_sqrt',
                                        op_type='Sqrt',
                                        inputs=[ly_add_0.output[0]],
                                        outputs=[ly_reduce_mean_1.output[0]+'_sqrt'])
        ly_div = onnx.helper.make_node(name=instance_normal.name+'_div',
                                       op_type='Div',
                                       inputs=[ly_sub.output[0], ly_sqrt.output[0]],
                                       outputs=[instance_normal.output[0]+'_div'])
        new_insert_nodes_list = [ly_reduce_mean_0, ly_sub, ly_pow, ly_reduce_mean_1, ly_add_0, ly_sqrt, ly_div]
        ly_mul_scale_name = instance_normal.input[1]
        if in_scale.size == top_out_shape[-1]:
            new_in_scale = np.reshape(in_scale, top_in_shape)
            new_in_scale_tensor = get_initial_by_value(onnx_model, new_in_scale)
            if new_in_scale_tensor is None:
                new_in_scale_tensor = onnx.helper.make_tensor(name=instance_normal.input[1]+'_shape_{}x{}x{}x{}'\
                    .format(top_in_shape[0], top_in_shape[1], top_in_shape[2], top_in_shape[3]),
                    data_type=NPDTYPE_2_ONNXDTYPE[new_in_scale.dtype],
                    dims=new_in_scale.shape,
                    vals=new_in_scale.flatten().tolist())
                onnx_model.graph.initializer.append(new_in_scale_tensor)
            ly_mul_scale_name = new_in_scale_tensor.name 
        ly_mul = onnx.helper.make_node(name=instance_normal.name+'_mul',
                                       op_type='Mul',
                                       inputs=[ly_div.output[0], ly_mul_scale_name],
                                       outputs=[instance_normal.output[0]+'_mul'])
        if not (in_scale == np.ones(in_scale.shape, dtype=in_scale.dtype)).any():
            new_insert_nodes_list.append(ly_mul)
        ly_add_bais_name = instance_normal.input[2]
        if in_bais.size == top_out_shape[-1]:
            new_in_bais = np.reshape(in_bais, top_in_shape)
            new_in_bais_tensor = get_initial_by_value(onnx_model, new_in_bais)
            if new_in_bais_tensor is None:
                new_in_bais_tensor = onnx.helper.make_tensor(name=instance_normal.input[2]+'_shape_{}x{}x{}x{}'\
                    .format(top_in_shape[0], top_in_shape[1], top_in_shape[2], top_in_shape[3]),
                    data_type=NPDTYPE_2_ONNXDTYPE[new_in_bais.dtype],
                    dims=new_in_bais.shape,
                    vals=new_in_bais.flatten().tolist())
                onnx_model.graph.initializer.append(new_in_bais_tensor)
            ly_add_bais_name = new_in_bais_tensor.name
        ly_add_1 = onnx.helper.make_node(name=instance_normal.name+'_add_1',
                                       op_type='Add',
                                       inputs=[new_insert_nodes_list[-1].output[0], ly_add_bais_name],
                                       outputs=[bot_reshape.output[0]])
        if not (in_bais == np.zeros(in_bais.shape, dtype=in_bais.dtype)).all():
            new_insert_nodes_list.append(ly_add_1)
        new_insert_nodes_list[-1].output[0] = bot_reshape.output[0]
        new_insert_nodes_list.reverse()
        onnx_model = insert_node_by_list(onnx_model, new_insert_nodes_list, node_index)
        
        onnx_model = delete_value_info_by_name(onnx_model, top_reshape.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, instance_normal.output[0])
        onnx_model = delete_nodes(onnx_model, nodes_list) 
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False 

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceGatherGatherTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Gather", "Gather", "Transpose"]):      
        nodes_serial = get_node_serial_group(onnx_model, node, ["Gather", "Gather", "Transpose"])
        top_gather_node, bot_gather_node, transpose_node = nodes_serial
        top_gather_in_shape = get_shape_by_name(onnx_model, top_gather_node.input[0])
        if len(top_gather_in_shape) != 4:
            return onnx_model, False
        top_gather_indices = get_tensor_from_initializer(onnx_model, top_gather_node.input[1])
        bot_gather_indices = get_tensor_from_initializer(onnx_model, bot_gather_node.input[1])
        top_gather_axis = attribute_to_dict(top_gather_node.attribute).get('axis', 0)
        bot_gather_axis = attribute_to_dict(bot_gather_node.attribute).get('axis', 0)
        top_gather_out_shape = get_shape_by_name(onnx_model, top_gather_node.output[0])
        top_gather_pos_axis = len(top_gather_in_shape) + top_gather_axis if top_gather_axis < 0 else top_gather_axis
        bot_gather_pos_axis = len(top_gather_out_shape) + bot_gather_axis if bot_gather_axis < 0 else bot_gather_axis
        if top_gather_pos_axis < 2 or bot_gather_pos_axis < 2 \
            or len(top_gather_indices.shape) != 2 or len(bot_gather_indices.shape) != 2 \
                or top_gather_indices.size != top_gather_in_shape[top_gather_pos_axis] \
                    or bot_gather_indices.size != top_gather_out_shape[bot_gather_pos_axis]:
                        return onnx_model, False
        gather_axes_list = [top_gather_pos_axis, bot_gather_pos_axis]
        if gather_axes_list != [2, 4] and gather_axes_list != [3, 2]:
            return onnx_model, False
        top_gather_indices_pred = np.array(list(range(top_gather_indices.size)), dtype=top_gather_indices.dtype)
        top_gather_indices_pred = np.reshape(top_gather_indices_pred, (top_gather_indices.shape[1], top_gather_indices.shape[0])).transpose(1, 0)
        bot_gather_indices_pred = np.array(list(range(bot_gather_indices.size)), dtype=bot_gather_indices.dtype)
        bot_gather_indices_pred = np.reshape(bot_gather_indices_pred, (bot_gather_indices.shape[1], bot_gather_indices.shape[0])).transpose(1, 0)
        if not (top_gather_indices_pred == top_gather_indices).all() or not (bot_gather_indices_pred == bot_gather_indices).all():
            return onnx_model, False
        bot_gather_out_shape = get_shape_by_name(onnx_model, bot_gather_node.output[0])
        transpose_perm = attribute_to_dict(transpose_node.attribute).get('perm', list(range(len(bot_gather_out_shape))).reverse())
        # if transpose_perm != [0, 1, 2, 4, 3, 5]:
        #     return onnx_model, False
        new_reshape_shape = bot_gather_out_shape[:2] \
            +   [bot_gather_out_shape[3], bot_gather_out_shape[2], bot_gather_out_shape[5], bot_gather_out_shape[4]]
        new_reshape_shape_tensor = get_initial_by_value(onnx_model, np.array(new_reshape_shape, dtype=np.int64))
        if new_reshape_shape_tensor is None:
            new_reshape_shape_tensor = onnx.helper.make_tensor(name=top_gather_node.input[0]+'_new_shape',
                                                                   data_type=TensorProto.INT64,
                                                                   dims=[len(new_reshape_shape)],
                                                                   vals=new_reshape_shape)
            onnx_model.graph.initializer.append(new_reshape_shape_tensor)
        new_reshape_node = onnx.helper.make_node(name=top_gather_node.name+'_'+bot_gather_node.name+'_toReshape',
                                                     op_type='Reshape',
                                                     inputs=[top_gather_node.input[0], new_reshape_shape_tensor.name],
                                                     outputs=[bot_gather_node.output[0]+'_newShape'])
        new_transpose_perm = [[0, 1, 3, 2, 5, 4][i] for i in transpose_perm]
        new_transpose_perm_attr = onnx.helper.make_attribute('perm', new_transpose_perm)
        del transpose_node.attribute[:]
        transpose_node.attribute.append(new_transpose_perm_attr)
        transpose_node.input[0] = new_reshape_node.output[0]
        tensor_dtype = get_dtype_by_name(onnx_model, top_gather_node.input[0])
        onnx_model = delete_value_info_by_name(onnx_model, bot_gather_node.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, top_gather_node.output[0])
        new_reshape_value_info = onnx.helper.make_tensor_value_info(new_reshape_node.output[0], tensor_dtype, new_reshape_shape)
        onnx_model.graph.value_info.append(new_reshape_value_info)
        onnx_model = delete_nodes(onnx_model, [bot_gather_node, top_gather_node])
        onnx_model.graph.node.insert(node_index, new_reshape_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False
           
@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceReshapeCol2Im(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Reshape", "Col2Im"]):      
        reshape_node, col2im_node = get_node_serial_group(onnx_model, node, ["Reshape", "Col2Im"])       
        reshape_in_shape = get_shape_by_name(onnx_model, reshape_node.input[0])
        reshape_out_shape = get_shape_by_name(onnx_model, reshape_node.output[0])
        if len(reshape_in_shape) != 4 or len(reshape_out_shape) != 3 \
            or reshape_in_shape[1]*reshape_in_shape[2] != reshape_out_shape[1] \
                or reshape_in_shape[-1] != reshape_out_shape[-1]:
                    return onnx_model, False
        col2im_attr = attribute_to_dict(col2im_node.attribute)
        col2im_dilations = col2im_attr.get('dilations', [1, 1])
        col2im_pads = col2im_attr.get('pads', [0, 0, 0, 0])
        if col2im_dilations != [1, 1] or col2im_pads != [0, 0, 0, 0]:
            return onnx_model, False
        image_shape = get_tensor_from_initializer(onnx_model, col2im_node.input[1]).tolist()
        block_shape = get_tensor_from_initializer(onnx_model, col2im_node.input[2]).tolist()
        if len(image_shape) != 2 or len(block_shape) != 2:
            return onnx_model, False
        col2im_out_shape = get_shape_by_name(onnx_model, col2im_node.output[0])
        if reduce(lambda x,y:x*y, col2im_out_shape) != reduce(lambda x,y:x*y, reshape_out_shape):
            return onnx_model, False
        top_shape_h = image_shape[0] // block_shape[0]
        top_shape_w = image_shape[1] // block_shape[1]
        top_shape_c = reshape_out_shape[1] // (block_shape[0] * block_shape[1])
        top_shape = [col2im_out_shape[0], top_shape_c] + block_shape + [top_shape_h, top_shape_w]
        if reduce(lambda x,y:x*y, col2im_out_shape) != reduce(lambda x,y:x*y, top_shape):
            return onnx_model, False
        top_shape_tensor = get_initial_by_value(onnx_model, np.array(top_shape, dtype=np.int64))
        if top_shape_tensor is None:
            top_shape_tensor_name = col2im_node.name+'_top_shape'
            top_shape_tensor = onnx.helper.make_tensor(name=top_shape_tensor_name,
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(top_shape)],
                                                       vals=top_shape)
            onnx_model.graph.initializer.append(top_shape_tensor)
        bot_shape = top_shape[:2] + image_shape
        bot_shape_tensor = get_initial_by_value(onnx_model, np.array(bot_shape, dtype=np.int64))
        if bot_shape_tensor is None:
            bot_shape_tensor_name = col2im_node.name+'_bot_shape'
            bot_shape_tensor = onnx.helper.make_tensor(name=bot_shape_tensor_name,
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(bot_shape)],
                                                       vals=bot_shape)
            onnx_model.graph.initializer.append(bot_shape_tensor)
        new_top_reshape = onnx.helper.make_node(name=col2im_node.name+'_top_reshape',
                                                op_type='Reshape',
                                                inputs=[reshape_node.input[0], top_shape_tensor.name],
                                                outputs=[col2im_node.output[0]+'_top_reshape'])
        new_transpose = onnx.helper.make_node(name=col2im_node.name+'_transpose',
                                              op_type='Transpose',
                                              inputs=[new_top_reshape.output[0]],
                                              outputs=[col2im_node.output[0]+'_transpose'],
                                              perm=[0, 1, 4, 2, 5, 3])
        new_bot_reshape = onnx.helper.make_node(name=col2im_node.name+'_bot_reshape',
                                                op_type='Reshape',
                                                inputs=[new_transpose.output[0], bot_shape_tensor.name],
                                                outputs=[col2im_node.output[0]])
        onnx_model = delete_value_info_by_name(onnx_model, reshape_node.output[0])
        onnx_model = delete_nodes(onnx_model, [reshape_node, col2im_node])
        onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], node_index)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceGatherToSlice(onnx_model, node, node_index):
    if node.op_type == 'Gather':
        input_shape = get_shape_by_name(onnx_model, node.input[0])
        output_shape = get_shape_by_name(onnx_model, node.output[0])
        attr_dict = attribute_to_dict(node.attribute)
        axis = attr_dict.get('axis', 0)          
        axis_pos = len(input_shape) + axis if axis < 0 else axis
        if not find_init_by_name(onnx_model, node.input[1]):
            return onnx_model, False
        indices_tensor = get_tensor_from_initializer(onnx_model, node.input[1])
        if indices_tensor.size != 1:
            return onnx_model, False
        indice = int(indices_tensor) if indices_tensor.shape == () else indices_tensor.to_list()[0]
        slice_start_tensor = get_initial_by_value(onnx_model, np.array([indice], dtype=np.int64))
        if slice_start_tensor is None:
            slice_start_tensor = onnx.helper.make_tensor(name=node.name+'_param0',
                                                         data_type=TensorProto.INT64,
                                                         dims=[1],
                                                         vals=[indice])
            onnx_model.graph.initializer.append(slice_start_tensor)
        slice_end_tensor = get_initial_by_value(onnx_model, np.array([indice+1], dtype=np.int64))
        if slice_end_tensor is None:
            slice_end_tensor = onnx.helper.make_tensor(name=node.name+'_param1',
                                                       data_type=TensorProto.INT64,
                                                       dims=[1],
                                                       vals=[indice+1])
            onnx_model.graph.initializer.append(slice_end_tensor)
        slice_axes_tensor = get_initial_by_value(onnx_model, np.array([axis_pos], dtype=np.int64))
        if slice_axes_tensor is None:
            slice_axes_tensor = onnx.helper.make_tensor(name=node.name+'_param2',
                                                        data_type=TensorProto.INT64,
                                                        dims=[1],
                                                        vals=[axis_pos])
            onnx_model.graph.initializer.append(slice_axes_tensor)
        slice_step_tensor = get_initial_by_value(onnx_model, np.array([1], dtype=np.int64))
        if slice_step_tensor is None:
            slice_step_tensor = onnx.helper.make_tensor(name=node.name+'_param3',
                                                        data_type=TensorProto.INT64,
                                                        dims=[1],
                                                        vals=[1])
            onnx_model.graph.initializer.append(slice_step_tensor)
        slice_inputs = [node.input[0], slice_start_tensor.name, slice_end_tensor.name, 
                        slice_axes_tensor.name, slice_step_tensor.name]
        slice_node = onnx.helper.make_node(name=node.name+'_toSlice',
                                           op_type='Slice',
                                           inputs=slice_inputs,
                                           outputs=[node.output[0]+'_init'])
        new_nodes_list = []
        if len(output_shape) != len(input_shape):
            new_shape_tensor = get_initial_by_value(onnx_model, np.array(output_shape, dtype=np.int64))
            if new_shape_tensor is None:
                new_shape_tensor = onnx.helper.make_tensor(name=node.name+'_shape',
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(output_shape)],
                                                           vals=output_shape)
                onnx_model.graph.initializer.append(new_shape_tensor)
            reshape_node = onnx.helper.make_node(name=node.name+'_reshape',
                                                 op_type='Reshape',
                                                 inputs=[slice_node.output[0], new_shape_tensor.name],
                                                 outputs=[node.output[0]])
            new_nodes_list.append(reshape_node)
        else:
            slice_node.output[0] = node.output[0]
        new_nodes_list.append(slice_node)
        onnx_model.graph.node.remove(node)
        onnx_model = insert_node_by_list(onnx_model, new_nodes_list, node_index)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False