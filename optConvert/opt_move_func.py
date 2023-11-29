from unittest.mock import NonCallableMagicMock
from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_move_wrapper
def opt_moveBackwardCol2ImReshapeTransposeReshape(onnx_model):
    # def backward_move_mobilevitv2kqv(onnx_model, nodes_serial:list, kqv_serial:list, src_perm:list):
    #     split_node, softmax_node, top_mul_node, reducesum_node, relu_node, bot_mul_node = kqv_serial
    #     blk_in_shape = get_shape_by_name(onnx_model, nodes_serial[0].input[0])
    def backward_move_im2col(onnx_model, nodes_serial:list, im2col_serial:list, src_perm:list): 
        im2col_top_reshape, im2col_transpose, im2col_bot_reshape = im2col_serial
        col2im_top_reshape, col2im_transpose, col2im_bot_reshape = nodes_serial
        im2col_bot_in_shape = get_shape_by_name(onnx_model, im2col_bot_reshape.input[0])
        col2im_top_out_shape = get_shape_by_name(onnx_model, col2im_top_reshape.output[0])
        if im2col_bot_in_shape != col2im_top_out_shape:
            return onnx_model, False
        im2col_top_out_shape = get_shape_by_name(onnx_model, im2col_transpose.input[0])
        col2im_bot_in_shape = get_shape_by_name(onnx_model, col2im_transpose.output[0])
        if im2col_top_out_shape != col2im_bot_in_shape:
            return onnx_model, False
        im2col_top_in_shape = get_shape_by_name(onnx_model, im2col_top_reshape.input[0])
        col2im_bot_out_shape = get_shape_by_name(onnx_model, col2im_bot_reshape.output[0])
        if im2col_top_in_shape != col2im_bot_out_shape:
            return onnx_model, False
        col2im_next_nodes_list = get_node_by_input(onnx_model, col2im_bot_reshape.output)
        for col2im_next_node in col2im_next_nodes_list:
            for col2im_next_in_idx, col2im_next_input in enumerate(col2im_next_node.input):
                col2im_next_node.input[col2im_next_in_idx] = im2col_top_reshape.input[0] \
                    if col2im_next_input == col2im_bot_reshape.output[0] else col2im_next_input
        for src_node in nodes_serial:
            onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
            onnx_model.graph.node.remove(src_node)
        im2col_next_nodes_list = get_node_by_input(onnx_model, im2col_bot_reshape.output)
        if not im2col_next_nodes_list:
            for im2col_next_node in im2col_next_nodes_list:
                onnx_model = delete_value_info_by_name(onnx_model, im2col_next_node)
                onnx_model.graph.node.remove(im2col_next_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    def backward_move_swish(onnx_model, nodes_serial:list, pre_serial:list):
        swish_mul, swish_sigmoid = pre_serial
        swish_in_shape = get_shape_by_name(onnx_model, swish_sigmoid.input[0])
        blk_in_shape = get_shape_by_name(onnx_model, swish_mul.output[0])
        if swish_in_shape != blk_in_shape:
            return onnx_model, False
        next_nodes_list = get_node_by_input(onnx_model, nodes_serial[-1].output)
        pre_next_nodes_list = get_node_by_input(onnx_model, swish_mul.output)
        new_top_reshape = copy.deepcopy(nodes_serial[0])
        new_top_reshape.input[0] = swish_sigmoid.input[0]
        new_transpose = copy.deepcopy(nodes_serial[1])
        new_bot_reshape = copy.deepcopy(nodes_serial[-1])
        swish_mul.input[list(swish_mul.input).index(swish_sigmoid.input[0])] = new_bot_reshape.output[0]
        swish_sigmoid.input[0] = new_bot_reshape.output[0]
        for next_node in next_nodes_list:
            for next_idx, next_input in enumerate(next_node.input):
                next_node.input[next_idx] = swish_mul.output[0] \
                    if next_input == new_bot_reshape.output[0] else next_input
        swish_node_id = min(get_node_id(onnx_model, swish_mul), get_node_id(onnx_model, swish_sigmoid))
        inverse_bot_shape = get_shape_by_name(onnx_model, nodes_serial[0].input[0])
        inverse_top_shape = get_shape_by_name(onnx_model, nodes_serial[-1].input[0])
        for src_node in nodes_serial:
            onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
            onnx_model.graph.node.remove(src_node)
        onnx_model = delete_value_info_by_name(onnx_model, swish_mul.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, swish_sigmoid.output[0])
        onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], swish_node_id)
        
        if len(pre_next_nodes_list) > 1:
            inverse_top_shape_tensor = get_initial_by_value(onnx_model, np.array(inverse_top_shape, dtype=np.int64))
            if inverse_top_shape_tensor is None:
                inverse_top_shape_tensor = onnx.helper.make_tensor(
                    name=get_unique_node_tensor_name(onnx_model, swish_mul.output[0]+'_top_shape'),
                    data_type=TensorProto.INT64,
                    dims=[len(inverse_top_shape)],
                    vals=inverse_top_shape)
                onnx_model.graph.initializer.append(inverse_top_shape_tensor)
            inverse_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(inverse_bot_shape, dtype=np.int64))
            if inverse_bot_shape_tensor is None:
                inverse_bot_shape_tensor = onnx.helper.make_tensor(
                    name=get_unique_node_tensor_name(onnx_model, swish_mul.output[0]+'_bot_shape'),
                    data_type=TensorProto.INT64,
                    dims=[len(inverse_bot_shape)],
                    vals=inverse_bot_shape)
                onnx_model.graph.initializer.append(inverse_bot_shape_tensor)
            inverse_top_reshape = onnx.helper.make_node(
                name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name+'_inverse'),
                op_type='Reshape',
                inputs=[swish_mul.output[0], inverse_top_shape_tensor.name],
                outputs=[swish_mul.output[0]+'_inverse_bot_shape'])
            inverse_transpose = onnx.helper.make_node(
                name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name+'_inverse'),
                op_type='Transpose',
                inputs=[inverse_top_reshape.output[0]],
                outputs=[swish_mul.output[0]+'_inverse'],
                perm=[0, 1, 3, 5, 2, 4])
            inverse_bot_reshape = onnx.helper.make_node(
                name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name+'_inverse'),
                op_type='Reshape',
                inputs=[inverse_transpose.output[0], inverse_bot_shape_tensor.name],
                outputs=[swish_mul.output[0]+'_inverse_bot_shape'])
            swish_node_id = get_node_id(onnx_model, swish_mul)
            onnx_model = insert_node_by_list(onnx_model, [inverse_bot_reshape, inverse_transpose, inverse_top_reshape], swish_node_id+1)
            for pre_next_node in pre_next_nodes_list:
                if pre_next_node.name == new_top_reshape.name:
                    continue
                for pre_next_in_idx, pre_next_input in enumerate(pre_next_node.input):
                    pre_next_node.input[pre_next_in_idx] = inverse_bot_reshape.output[0] \
                        if pre_next_input == swish_mul.output[0] else pre_next_input
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    def backward_move_conv(onnx_model, nodes_serial:list, pre_node:onnx.NodeProto, src_perm:list):
        conv_attribute_dict = attribute_to_dict(pre_node.attribute)
        dilations = conv_attribute_dict.get('dilations', [1, 1])
        kernel_shape = conv_attribute_dict.get('kernel_shape', [1, 1])
        pads = conv_attribute_dict.get('pads', [0, 0, 0, 0])
        strides = conv_attribute_dict.get('strides', [1, 1])
        conv_in_shape = get_shape_by_name(onnx_model, pre_node.input[0])
        conv_out_shape = get_shape_by_name(onnx_model, pre_node.output[0])        
        if dilations != [1, 1] or kernel_shape != [1, 1] or pads != [0, 0, 0, 0] \
            or strides != [1, 1] or conv_in_shape[-2:] != conv_out_shape[-2:]:
                return onnx_model, False
        
        next_nodes_list = get_node_by_input(onnx_model, nodes_serial[-1].output)
        pre_next_nodes_list = get_node_by_input(onnx_model, pre_node.output) 
        top_out_shape = get_shape_by_name(onnx_model, nodes_serial[0].output[0])
        blk_out_shape = get_shape_by_name(onnx_model, nodes_serial[-1].output[0])
        new_top_shape = copy.deepcopy(top_out_shape)
        new_top_shape[1] = conv_in_shape[1]
        new_bot_shape = copy.deepcopy(blk_out_shape)
        new_bot_shape[1] = conv_in_shape[1]
        new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_top_shape, dtype=np.int64))
        if new_top_shape_tensor is None:
            new_top_shape_name = nodes_serial[0].input[1] \
                if not find_other_node_by_input(onnx_model, nodes_serial[0], nodes_serial[0].input[1]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1])
            new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(new_top_shape)],
                                                           vals=new_top_shape)
            if new_top_shape_tensor.name == nodes_serial[0].input[1]:
                onnx_model = delete_initializer_by_name(onnx_model, new_top_shape_tensor.name)
            onnx_model.graph.initializer.append(new_top_shape_tensor)
        new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(new_bot_shape, dtype=np.int64))
        if new_bot_shape_tensor is None:
            new_bot_shape_name = nodes_serial[-1].input[1] \
                if not find_other_node_by_input(onnx_model, nodes_serial[-1], nodes_serial[-1].input[1]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1])
            new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(new_bot_shape)],
                                                           vals=new_bot_shape)
            if new_bot_shape_tensor.name == nodes_serial[-1].input[1]:
                onnx_model = delete_initializer_by_name(onnx_model, new_bot_shape_tensor.name)
            onnx_model.graph.initializer.append(new_bot_shape_tensor)
        new_top_reshape = onnx.helper.make_node(name=nodes_serial[0].name,
                                                op_type='Reshape',
                                                inputs=[pre_node.input[0], new_top_shape_tensor.name],
                                                outputs=[nodes_serial[0].output[0]])
        new_transpose = onnx.helper.make_node(name=nodes_serial[1].name,
                                              op_type='Transpose',
                                              inputs=[new_top_reshape.output[0]],
                                              outputs=[nodes_serial[1].output[0]],
                                              perm=src_perm)
        new_bot_reshape = onnx.helper.make_node(name=nodes_serial[-1].name,
                                                op_type='Reshape',
                                                inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                outputs=[nodes_serial[-1].output[0]])
        
        for src_node in nodes_serial:
            onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
            onnx_model.graph.node.remove(src_node)
            
        pre_node.input[0] = new_bot_reshape.output[0]
        pre_node_id = get_node_id(onnx_model, pre_node)
        onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], pre_node_id)
        
        for next_node in next_nodes_list:
            for next_id, next_input in enumerate(next_node.input):
                next_node.input[next_id] = pre_node.output[0] if next_input == nodes_serial[-1].output[0] else next_input
        
        if len(pre_next_nodes_list) > 1:
            inverse_top_shape = [new_top_shape[perm_id] for perm_id in src_perm]
            inverse_top_shape_tensor = get_initial_by_value(onnx_model, np.array(inverse_top_shape, dtype=np.int64))
            if inverse_top_shape_tensor is None:
                inverse_top_shape_tensor = onnx.helper.make_tensor(
                    name=get_unique_node_tensor_name(onnx_model, pre_node.output[0]+'_top_shape'),
                    data_type=TensorProto.INT64,
                    dims=[len(inverse_top_shape)],
                    vals=inverse_top_shape)
                onnx_model.graph.initializer.append(inverse_top_shape_tensor)
            blk_in_shape = get_shape_by_name(onnx_model, pre_node.output[0])
            inverse_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(blk_in_shape, dtype=np.int64))
            if inverse_bot_shape_tensor is None:
                inverse_bot_shape_tensor = onnx.helper.make_tensor(
                    name=get_unique_node_tensor_name(onnx_model, pre_node.output[0]+'_bot_shape'),
                    data_type=TensorProto.INT64,
                    dims=[len(blk_in_shape)],
                    vals=blk_in_shape
                )
                onnx_model.graph.initializer.append(inverse_bot_shape_tensor)
            inverse_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name+'_inverse'),
                                                        op_type='Reshape',
                                                        inputs=[pre_node.output[0], inverse_top_shape_tensor.name],
                                                        outputs=[pre_node.output[0]+'_inverse_top_shape'])
            inverse_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name+'_inverse'),
                                                      op_type='Transpose',
                                                      inputs=[inverse_top_reshape.output[0]],
                                                      outputs=[pre_node.output[0]+'_inverse'],
                                                      perm=[0, 1, 3, 5, 2, 4])
            inverse_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name+'_inverse'),
                                                        op_type='Reshape',
                                                        inputs=[inverse_transpose.output[0], inverse_bot_shape_tensor.name],
                                                        outputs=[pre_node.output[0]+'_inverse_bot_shape'])
            pre_node_id = get_node_id(onnx_model, pre_node)
            onnx_model = insert_node_by_list(onnx_model, [inverse_bot_reshape, inverse_transpose, inverse_top_reshape], pre_node_id+1)
            for pre_next_node in pre_next_nodes_list:
                if pre_next_node.name == nodes_serial[0].name:
                    continue
                for pre_next_in_idx, pre_next_input in enumerate(pre_next_node.input):
                    pre_next_node.input[pre_next_in_idx] = inverse_bot_reshape.output[0] \
                        if pre_next_input == pre_node.output[0] else pre_next_input
        onnx_model = delete_value_info_by_name(onnx_model, pre_node.output[0])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    def backward_move_addmulsubdiv(onnx_model, nodes_serial:list, pre_node:onnx.NodeProto, src_perm:list):
        blk_in_shape = get_shape_by_name(onnx_model, nodes_serial[0].input[0])
        new_serial_num = 0
        for pre_input in pre_node.input:
            if find_init_by_name(onnx_model, pre_input):
                factor_constant = get_tensor_from_initializer(onnx_model, pre_input)
                if (len(factor_constant.shape) >= 2 and list(factor_constant.shape[-2:]) not in [[1, 1], blk_in_shape[2:]]) \
                    or (len(factor_constant.shape) < 2 and factor_constant.size != 1):
                        return onnx_model, False
            else:
                pre_in_shape = get_shape_by_name(onnx_model, pre_input)
                if pre_in_shape != blk_in_shape and pre_in_shape[2:] != [1, 1]:
                    return onnx_model, False
                elif pre_in_shape == blk_in_shape:
                    new_serial_num += 1
                elif len(pre_in_shape) >= 2 and pre_in_shape[-2:] == blk_in_shape[-2:]:
                    new_serial_num += 1
                    
        blk_out_shape = get_shape_by_name(onnx_model, nodes_serial[-1].output[0])
        top_out_shape = get_shape_by_name(onnx_model, nodes_serial[0].output[0])
        bot_in_shape = get_shape_by_name(onnx_model, nodes_serial[1].output[0])
        next_nodes_list = get_node_by_input(onnx_model, nodes_serial[-1].output)
        pre_next_nodes_list = get_node_by_input(onnx_model, pre_node.output)
        
        onnx_model = delete_value_info_by_name(onnx_model, nodes_serial[0].input[0])
        for src_node in nodes_serial:
            onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
            onnx_model.graph.node.remove(src_node)
            
        for idx, pre_input in enumerate(pre_node.input):
            pre_input_shape = get_shape_by_name(onnx_model, pre_input)
            if len(pre_input_shape) >= 2 and pre_input_shape[-2:] == blk_in_shape[-2:]:
                new_top_out_shape = pre_input_shape[:-2] + top_out_shape[-4:]
                new_bot_out_shape = pre_input_shape[:-2] + blk_out_shape[-2:]
                new_perm = src_perm[-len(new_top_out_shape):]
                new_perm = [perm - (len(top_out_shape) - len(new_top_out_shape)) for perm in new_perm]
                if find_init_by_name(onnx_model, pre_input):
                    factor_constant = get_tensor_from_initializer(onnx_model, pre_input)
                    new_factor = np.reshape(factor_constant, tuple(new_top_out_shape))
                    new_factor = np.transpose(new_factor, tuple(new_perm))
                    new_factor = np.reshape(new_factor, tuple(new_bot_out_shape))
                    new_factor_tensor = get_initial_by_value(onnx_model, new_factor)
                    if new_factor_tensor is None:
                        new_factor_name = pre_input if not find_other_node_by_input(onnx_model, pre_node, pre_input) \
                            else get_unique_node_tensor_name(onnx_model, pre_input)
                        new_factor_tensor = onnx.helper.make_tensor(name=new_factor_name,
                                                                    data_type=NPDTYPE_2_ONNXDTYPE[new_factor.dtype],
                                                                    dims=new_perm.shape,
                                                                    vals=new_factor.flatten().tolist())
                        if new_factor_tensor.name == pre_input:
                            onnx_model = delete_initializer_by_name(onnx_model, pre_input)
                        onnx_model.graph.initializer.append(new_factor_tensor)
                    pre_node.input[idx] = new_factor_tensor.name
                elif (idx > 0 and pre_input != pre_node.input[0]) or idx == 0:
                    new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_top_out_shape, dtype=np.int64))
                    if new_top_shape_tensor is None:
                        new_top_shape_name = nodes_serial[0].input[1] \
                            if not find_other_node_by_input(onnx_model, nodes_serial[0], nodes_serial[0].input[1]) \
                                else get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1])
                        new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_name,
                                                                       data_type=TensorProto.INT64,
                                                                       dims=[len(new_top_out_shape)],
                                                                       vals=new_top_out_shape)
                        if new_top_shape_tensor.name == nodes_serial[0].input[1]:
                            onnx_model = delete_initializer_by_name(onnx_model, nodes_serial[0].input[1])
                        onnx_model.graph.initializer.append(new_top_shape_tensor)
                    new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(new_bot_out_shape, dtype=np.int64))
                    if new_bot_shape_tensor is None:
                        new_bot_shape_name = nodes_serial[-1].input[1] \
                            if not find_other_node_by_input(onnx_model, nodes_serial[-1], nodes_serial[-1].input[1]) \
                                else get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1])
                        new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_name,
                                                                       data_type=TensorProto.INT64,
                                                                       dims=[len(new_bot_out_shape)],
                                                                       vals=new_bot_out_shape)
                        if new_bot_shape_tensor.name == nodes_serial[-1].input[1]:
                            onnx_model = delete_initializer_by_name(onnx_model, nodes_serial[-1].input[1])
                        onnx_model.graph.initializer.append(new_bot_shape_tensor)
                    append_str = '_%d'%idx if new_serial_num > 1 else ''
                    new_top_reshape = onnx.helper.make_node(name=nodes_serial[0].name + append_str,
                                                            op_type='Reshape',
                                                            inputs=[pre_input, new_top_shape_tensor.name],
                                                            outputs=[nodes_serial[0].output[0] + append_str])
                    new_transpose = onnx.helper.make_node(name=nodes_serial[1].name + append_str,
                                                          op_type='Transpose',
                                                          inputs=[new_top_reshape.output[0]],
                                                          outputs=[nodes_serial[1].output[0] + append_str],
                                                          perm=new_perm)
                    new_bot_reshape = onnx.helper.make_node(name=nodes_serial[-1].name + append_str,
                                                            op_type='Reshape',
                                                            inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                            outputs=[nodes_serial[-1].output[0] + append_str])
                    pre_node_id = get_node_id(onnx_model, pre_node)
                    onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], pre_node_id)
                    pre_node.input[idx] = new_bot_reshape.output[0]
                else:
                    pre_node.input[idx] = pre_node.input[0]
        for next_node in next_nodes_list:
            for next_id, next_input in enumerate(next_node.input):
                next_node.input[next_id] = pre_node.output[0] if next_input == nodes_serial[-1].output[0] else next_input
        # pre_node.output[0] = nodes_serial[-1].output[0]
        if len(pre_next_nodes_list) > 1:
            pre_next_top_shape_tensor = get_initial_by_value(onnx_model, np.array(bot_in_shape, dtype=np.int64))
            if pre_next_top_shape_tensor is None:
                pre_next_top_shape_tensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, pre_node.output[0]+'_top_shape'),
                                                                    data_type=TensorProto.INT64,
                                                                    dims=[len(bot_in_shape)],
                                                                    vals=bot_in_shape)
                onnx_model.graph.initializer.append(pre_next_top_shape_tensor)
            pre_next_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(blk_in_shape, dtype=np.int32))
            if pre_next_bot_shape_tensor is None:
                pre_next_bot_shape_tensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, pre_node.output[0]+'_bot_shape'),
                                                                    data_type=TensorProto.INT64,
                                                                    dims=[len(blk_in_shape)],
                                                                    vals=blk_in_shape)
                onnx_model.graph.initializer.append(pre_next_bot_shape_tensor)
            pre_next_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name+'_inverse'),
                                                         op_type='Reshape',
                                                         inputs=[pre_node.output[0], pre_next_top_shape_tensor.name],
                                                         outputs=[pre_node.output[0]+'_inverse_top_shape'])
            pre_next_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name+'_inverse'),
                                                       op_type='Transpose',
                                                       inputs=[pre_next_top_reshape.output[0]],
                                                       outputs=[pre_node.output[0]+'_inverse'],
                                                       perm=[0, 1, 3, 5, 2, 4])
            pre_next_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name+'_inverse'),
                                                         op_type='Reshape',
                                                         inputs=[pre_next_transpose.output[0], pre_next_bot_shape_tensor.name],
                                                         outputs=[pre_node.output[0]+'_inverse_bot_shape'])
            pre_node_id = get_node_id(onnx_model, pre_node)
            onnx_model = insert_node_by_list(onnx_model, [pre_next_bot_reshape, pre_next_transpose, pre_next_top_reshape], pre_node_id+1)
            for pre_next_node in pre_next_nodes_list:
                if pre_next_node.name == nodes_serial[0].name:
                    continue
                for pre_next_in_idx, pre_next_input in enumerate(pre_next_node.input):
                    pre_next_node.input[pre_next_in_idx] = pre_next_bot_reshape.output[0] \
                        if pre_next_input == pre_node.output[0] else pre_next_input
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
                
    def block_move(onnx_model, node):
        def check_im2col_group(onnx_model, node):
            if node is None or node.op_type != 'Reshape':
                return False
            pre_node = get_node_by_output(onnx_model, node.input[0])
            if node is None or pre_node.op_type != 'Transpose':
                return False
            pre_pre_node = get_node_by_output(onnx_model, pre_node.input[0])
            if pre_pre_node.op_type != 'Reshape':
                return False
            top_in_shape = get_shape_by_name(onnx_model, pre_pre_node.input[0])
            bot_out_shape = get_shape_by_name(onnx_model, node.output[0])
            if len(top_in_shape) != len(bot_out_shape) or top_in_shape[:2] != bot_out_shape[:2]:
                return False
            bot_in_shape = get_shape_by_name(onnx_model, node.input[0])
            if len(bot_in_shape) != len(bot_out_shape)+2 or bot_in_shape[:2] != bot_out_shape[:2] \
                or bot_in_shape[-4]*bot_in_shape[-3] != bot_out_shape[-2] \
                    or bot_in_shape[-2]*bot_in_shape[-1] != bot_out_shape[-1]:
                        return False
            perm_list = attribute_to_dict(pre_node.attribute).get('perm', list(range(len(bot_in_shape))).reverse())
            if perm_list != [0, 1, 3, 5, 2, 4]:
                return False
            top_out_shape = get_shape_by_name(onnx_model, pre_pre_node.output[0])
            if top_out_shape[-4]*top_out_shape[-3] != top_in_shape[-2] \
                or top_out_shape[-2]*top_out_shape[-1] != top_in_shape[-1]:
                    return False
            return True
        
        def get_im2col_group(onnx_model, node):
            transpose_node = get_node_by_output(onnx_model, node.input[0])
            top_reshape_node = get_node_by_output(onnx_model, transpose_node.input[0])
            return [top_reshape_node, transpose_node, node]    
        
        def check_get_mobilevitv2kqv_group(onnx_model, node):
            if node.op_type != 'Mul' \
                or find_init_by_name(onnx_model, node.input[0]) or find_init_by_name(onnx_model, node.input[1]):
                    return False
            relu_node = get_node_by_output(onnx_model, node.input[0])
            reducesum_node = get_node_by_output(onnx_model, node.input[1])
            if relu_node is None or reducesum_node is None:
                return False
            relu_node, reducesum_node = [relu_node, reducesum_node] \
                if relu_node.op_type == 'Relu' else [reducesum_node, relu_node]
            if relu_node.op_type != 'Relu' or reducesum_node.op_type != 'ReduceMean':
                return False
            reducesum_in_shape = get_shape_by_name(onnx_model, reducesum_node.input[0])
            reducesum_attribute = attribute_to_dict(reducesum_node.attribute)
            reducesum_axes = reducesum_attribute.get('axes')
            if isinstance(reducesum_axes, list) and \
                (len(reducesum_axes) > 1 or reducesum_axes[0] not in [-1, len(reducesum_in_shape)-1]):
                    return False
            reducesum_keepdims = reducesum_attribute.get('keepdims', 1)
            if reducesum_keepdims != 1:
                return False
            top_mul_node = get_node_by_output(onnx_model, reducesum_node.input[0])
            if top_mul_node.op_type != 'Mul' or find_init_by_name(onnx_model, top_mul_node.input[0]) \
                or find_init_by_name(onnx_model, top_mul_node.input[1]):
                    return False
            split_node = get_node_by_output(onnx_model, top_mul_node.input[0])
            softmax_node = get_node_by_output(onnx_model, top_mul_node.input[1])
            if split_node is None or softmax_node is None:
                return False
            split_node, softmax_node = [split_node, softmax_node] \
                if softmax_node.op_type == 'Softmax' else [softmax_node, split_node]
            if split_node.op_type != 'Split' or softmax_node.op_type != 'Softmax':
                return False
            split_axis = attribute_to_dict(split_node.attribute).get('axis', 1)
            if split_axis not in [1, 1-len(reducesum_in_shape)]:
                return False
            softmax_in_node = get_node_by_output(onnx_model, softmax_node.input[0])
            if softmax_in_node is None or softmax_in_node.name != split_node.name:
                return False
            relu_in_node = get_node_by_output(onnx_model, relu_node.input[0])
            if relu_in_node is None or relu_in_node.name != split_node.name:
                return False
            softmax_axis = attribute_to_dict(softmax_node.attribute).get('axis', 1)
            if softmax_axis not in [-1, len(reducesum_in_shape)-1]:
                return False
            softmax_in_shape = get_shape_by_name(onnx_model, softmax_node.input[0])
            relu_in_shape = get_shape_by_name(onnx_model, relu_node.input[0])
            top_mul_split_input = top_mul_node.input[0] \
                if top_mul_node.input[0] in list(split_node.output) else top_mul_node.input[1]
            top_mul_in_shape = get_shape_by_name(onnx_model, top_mul_split_input)
            if softmax_in_shape[1] != 1 or relu_in_shape[1] != top_mul_in_shape[1]:
                return False
            split_outputs_list = list(split_node.output)
            if split_outputs_list.index(top_mul_split_input) != 2 \
                or split_outputs_list.index(softmax_node.input[0]) != 1 \
                    or split_outputs_list.index(relu_node.input[0]) != 3:
                        return False
            return [split_node, softmax_node, top_mul_node, reducesum_node, relu_node, node]
        
        def check_swish_group(onnx_model, node):
            if node.op_type != 'Mul' \
                or find_init_by_name(onnx_model, node.input[0]) or find_init_by_name(onnx_model, node.input[0]):
                    return False
            pre_node0 = get_node_by_output(onnx_model, node.input[0])
            pre_node1 = get_node_by_output(onnx_model, node.input[1])
            sigmoid_node, input_node = [pre_node0, pre_node1] if pre_node0.op_type == 'Sigmoid' else [pre_node1, pre_node0]
            if sigmoid_node is None or sigmoid_node.op_type != 'Sigmoid' or sigmoid_node.input[0] != input_node.output[0]:
                return False
            else:
                return True
        
        def get_swish_group(onnx_model, node):
            pre_node0 = get_node_by_output(onnx_model, node.input[0])
            pre_node1 = get_node_by_output(onnx_model, node.input[1])
            return [node, pre_node0] if pre_node0.op_type == 'Sigmoid' else [node, pre_node1]
            
        if check_node_serial_group(onnx_model, node, ["Reshape", "Transpose", "Reshape"]):
            nodes_serial = get_node_serial_group(onnx_model, node, ["Reshape", "Transpose", "Reshape"]) 
            top_reshape_node, transpose_node, bot_reshape_node = nodes_serial
            blk_in_shape = get_shape_by_name(onnx_model, top_reshape_node.input[0])
            blk_out_shape = get_shape_by_name(onnx_model, bot_reshape_node.output[0])
            if len(blk_in_shape) != 4 or len(blk_out_shape) != 4 or blk_in_shape[:2] != blk_out_shape[:2]:
                return onnx_model, False
            top_out_shape = get_shape_by_name(onnx_model, top_reshape_node.output[0])
            if len(top_out_shape) != 6 or top_out_shape[:2] != blk_in_shape[:2] or blk_in_shape[0] != 1 \
                or top_out_shape[2]*top_out_shape[3] != blk_in_shape[2] or top_out_shape[4]*top_out_shape[5] != blk_in_shape[3]:
                    return onnx_model, False
            transpose_perm = attribute_to_dict(transpose_node.attribute).get('perm', list(range(len(top_out_shape))).reverse())
            if transpose_perm != [0, 1, 4, 2, 5, 3]:
                return onnx_model, False
            
            pre_node = get_node_by_output(onnx_model, top_reshape_node.input[0])
            # mobilevitv2kqv_serial = check_get_mobilevitv2kqv_group(onnx_model, pre_node)
            # if mobilevitv2kqv_serial:
            #     onnx_model, state = backward_move_mobilevitv2kqv(onnx_model, nodes_serial, mobilevitv2kqv_serial, transpose_perm)
            if check_swish_group(onnx_model, pre_node):
                pre_nodes_serial = get_swish_group(onnx_model, pre_node)
                onnx_model, state = backward_move_swish(onnx_model, nodes_serial, pre_nodes_serial)
            elif check_im2col_group(onnx_model, pre_node):
                pre_nodes_serial = get_im2col_group(onnx_model, pre_node)
                onnx_model, state = backward_move_im2col(onnx_model, nodes_serial, pre_nodes_serial, transpose_perm)
            elif pre_node.op_type in ['Add', 'Mul', 'Sub', 'Div']:
                onnx_model, state = backward_move_addmulsubdiv(onnx_model, nodes_serial, pre_node, transpose_perm)
            elif pre_node.op_type == 'Conv':
                onnx_model, state = backward_move_conv(onnx_model, nodes_serial, pre_node, transpose_perm)
            else:
                return onnx_model, False
            return onnx_model, state
        return onnx_model, False

    for node in onnx_model.graph.node[::-1]:
        onnx_model, restart = block_move(onnx_model, node)
        if restart:
            onnx_model = infer_model_shape(onnx_model)
            return onnx_model, restart
    return onnx_model, False

@OnnxDebuggerMeet.opt_move_wrapper
def opt_moveForwardIm2ColReshapeTransposeReshape(onnx_model):
    def forward_move_reducemean(onnx_model, nodes_serial:list, next_node:onnx.NodeProto, src_perm:list):
        next_nodes_list = get_node_by_input(onnx_model, nodes_serial[-1].output)
        reducemean_in_shape = get_shape_by_name(onnx_model, next_node.input[0])
        reducemean_out_shape = get_shape_by_name(onnx_model, next_node.output[0])
        reducemean_attributes = attribute_to_dict(next_node.attribute)
        if onnx_model.opset_import[0].version < 18:
            reducemean_axes_list = reducemean_attributes.get('axes')
        else:
            reducemean_axes_list = get_tensor_from_initializer(onnx_model, next_node.input[1]).tolist()
        if not isinstance(reducemean_axes_list, list):
            reducemean_axes_list = [reducemean_axes_list]
        reducemean_axes_pos_list = []
        for reducemean_axis in reducemean_axes_list:
            reducemean_pos_axis = reducemean_axis if reducemean_axis >= 0 else len(reducemean_in_shape)+reducemean_axis
            reducemean_axes_pos_list.append(reducemean_pos_axis)
        reducemean_out_infer = copy.deepcopy(reducemean_out_shape)
        for reducemean_pos_axis in reducemean_axes_pos_list:
            reducemean_out_infer[reducemean_pos_axis] = 1
        if reducemean_out_infer != reducemean_in_shape and reducemean_out_infer[-2:] != [1, 1]:
            return onnx_model, False
        im2col_top_out_shape = get_shape_by_name(onnx_model, nodes_serial[0].output[0])
        im2col_bot_out_shape = get_shape_by_name(onnx_model, nodes_serial[-1].output[0])
        if len(next_nodes_list) <= 1:
            for src_node in nodes_serial:
                onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
                onnx_model.graph.node.remove(src_node)
        if reducemean_out_infer == reducemean_in_shape:
            new_top_shape = copy.deepcopy(im2col_top_out_shape)
            new_bot_shape = copy.deepcopy(im2col_bot_out_shape)
            new_top_shape_tensor_name = nodes_serial[0].input[1] \
                if not find_other_node_by_input(onnx_model, nodes_serial[0], nodes_serial[0].input[1]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1])
            new_bot_shape_tensor_name = nodes_serial[-1].input[1] \
                if not find_other_node_by_input(onnx_model, nodes_serial[-1], nodes_serial[-1].input[1]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1])
            if len(next_nodes_list) > 1:
                new_top_shape_tensor_name = nodes_serial[0].input[1] 
                new_bot_shape_tensor_name = nodes_serial[-1].input[1]
            new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_top_shape, dtype=np.int64))
            if new_top_shape_tensor is None:
                new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_tensor_name,
                                                            data_type=TensorProto.INT64,
                                                            dims=[len(new_top_shape)],
                                                            vals=new_top_shape)
                onnx_model.graph.initializer.append(new_top_shape_tensor)
            new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(new_bot_shape, dtype=np.int64))
            if new_bot_shape_tensor is None:
                new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_tensor_name,
                                                               data_type=TensorProto.INT64,
                                                               dims=[len(new_bot_shape)],
                                                               vals=new_bot_shape)
                onnx_model.graph.initializer.append(new_bot_shape_tensor)
            new_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name),
                                                    op_type='Reshape',
                                                    inputs=[next_node.output[0], new_top_shape_tensor.name],
                                                    outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[0].output[0])])
            new_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name),
                                                  op_type='Transpose',
                                                  inputs=[new_top_reshape.output[0]],
                                                  outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[1].output[0])],
                                                  perm=src_perm)
            new_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name),
                                                    op_type='Reshape',
                                                    inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                    outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[-1].output[0])])
            next_node.input[0] = nodes_serial[0].input[0]
            reducemean_next_nodes_list = get_node_by_input(onnx_model, next_node.output)
            for reducemean_next_node in reducemean_next_nodes_list:
                for cur_idx, cur_input in enumerate(reducemean_next_node.input):
                    reducemean_next_node.input[cur_idx] = new_bot_reshape.output[0] \
                        if cur_input == next_node.output[0] else cur_input
            reducemean_node_id = get_node_id(onnx_model, next_node)
            onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], reducemean_node_id+1)
        else:
            next_node.input[0] = nodes_serial[0].input[0]
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True 
    
    def forward_move_subaddmuldiv(onnx_model, nodes_serial:list, next_node:onnx.NodeProto, src_perm:list):
        next_nodes_list = get_node_by_input(onnx_model, nodes_serial[-1].output)
        next_other_input = next_node.input[0] if next_node.input[1] == nodes_serial[-1].output[0] else next_node.input[1]
        blk_in_shape = get_shape_by_name(onnx_model, nodes_serial[0].input[0])
        blk_out_shape = get_shape_by_name(onnx_model, nodes_serial[-1].output[0])
        next_other_in_shape = get_shape_by_name(onnx_model, next_other_input)
        next_out_shape = get_shape_by_name(onnx_model, next_node.output[0])
        bot_in_shape = get_shape_by_name(onnx_model, nodes_serial[-1].input[0])
        top_out_shape = get_shape_by_name(onnx_model, nodes_serial[0].output[0])
        if next_out_shape != blk_out_shape:
            return onnx_model, False
        second_next_nodes_list = get_node_by_input(onnx_model, next_node.output)
        if next_other_input != nodes_serial[-1].output[0]:
            if find_init_by_name(onnx_model, next_other_input):
                other_in_array = get_tensor_from_initializer(onnx_model, next_other_input)
                if len(other_in_array.shape) >= 2 and other_in_array.shape[-2:] == blk_out_shape[-2:]:
                    new_in_array = np.reshape(other_in_array, tuple(list(other_in_array.shape)[:-2]+bot_in_shape[-4:]))
                    new_in_array_init_perm = [0, 1, 4, 2, 5, 3][-len(new_in_array.shape):]
                    new_in_array_perm = [perm - (6 - len(new_in_array.shape)) for perm in new_in_array_init_perm]
                    new_in_array = np.transpose(new_in_array, tuple(new_in_array_perm))
                    new_in_tensor = get_initial_by_value(onnx_model, new_in_array)
                    if new_in_tensor is None:
                        new_in_tensor_name = next_other_input \
                            if not find_other_node_by_input(onnx_model, next_node, next_other_input) \
                                else get_unique_node_tensor_name(onnx_model, next_other_input)
                        new_in_tensor = onnx.helper.make_tensor(name=new_in_tensor_name,
                                                                data_type=NPDTYPE_2_ONNXDTYPE[new_in_array.dtype],
                                                                dims=new_in_array.shape,
                                                                vals=new_in_array.flatten().tolist())
                        onnx_model.graph.initializer.append(new_in_tensor)
                    next_node.input[list(next_node.input).index(next_other_input)] = new_in_tensor.name
                elif (len(other_in_array.shape) >= 2 and list(other_in_array.shape[-2:]) != [1, 1]) \
                    or (len(other_in_array.shape) == 1 and other_in_array.shape[0] != 1):
                        return onnx_model, False
            else:
                if len(next_other_in_shape) >= 2 and next_other_in_shape[-2:] == blk_out_shape[-2:]:
                    new_inverse_top_shape = next_other_in_shape[:-2] + bot_in_shape[-4:]
                    new_inverse_init_perm = [0, 1, 4, 2, 5, 3][-len(new_inverse_top_shape)]
                    new_inverse_perm = [perm - (6 - len(new_inverse_init_perm)) for perm in new_inverse_init_perm]
                    new_inverse_bot_shape = next_other_in_shape[:-2] + blk_in_shape[-2:]
                    new_inverse_top_shape_tensor_name = get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1]+'_inverse')
                    new_inverse_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_inverse_top_shape, dtype=np.int64))
                    if new_inverse_top_shape_tensor is None:
                        new_inverse_top_shape_tensor = onnx.helper.make_tensor(name=new_inverse_top_shape_tensor_name,
                                                                               data_type=TensorProto.INT64,
                                                                               dims=[len(new_inverse_top_shape)],
                                                                               vals=new_inverse_top_shape)
                        onnx_model.graph.initializer.append(new_inverse_top_shape_tensor)
                    new_inverse_bot_shape_tensor_name = get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1]+'_inverse')
                    new_inverse_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(new_inverse_bot_shape, dtype=np.int64))
                    if new_inverse_top_shape_tensor is None:
                        new_inverse_bot_shape_tensor = onnx.helper.make_tensor(name=new_inverse_bot_shape_tensor_name,
                                                                               data_type=TensorProto.INT64,
                                                                               dims=[len(new_inverse_bot_shape)],
                                                                               vals=new_inverse_bot_shape)
                        onnx_model.graph.initializer.append(new_inverse_bot_shape_tensor)
                    inverse_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name+'_inverse'),
                                                                op_type='Reshape',
                                                                inputs=[next_other_input, new_inverse_top_shape_tensor.name],
                                                                outputs=[next_other_input+'_inverse_top'])
                    inverse_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name+'_inverse'),
                                                              op_type='Transpose',
                                                              inputs=[inverse_top_reshape.output[0]],
                                                              outputs=[next_other_input+'_inverse'],
                                                              perm=new_inverse_perm)
                    inverse_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name+'_inverse'),
                                                                op_type='Reshape',
                                                                inputs=[inverse_transpose.output[0], new_inverse_bot_shape_tensor.name],
                                                                outputs=[next_other_input+'_inverse_bot'])
                    next_node_id = get_node_id(onnx_model, next_node)
                    next_node.input[list(next_node.input).index(next_other_input)] = inverse_bot_reshape.output[0]
                    onnx_model = insert_node_by_list(onnx_model, [inverse_bot_reshape, inverse_transpose, inverse_top_reshape], next_node_id)
                elif (len(next_other_in_shape) >= 2 and next_other_in_shape[-2:] != [1, 1]) \
                    or (len(next_other_in_shape) == 1 and next_other_in_shape[0] != 1):
                        return onnx_model, False
            if len(next_nodes_list) <= 1:
                for src_node in nodes_serial:
                    onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
                    onnx_model.graph.node.remove(src_node)
            new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(top_out_shape, dtype=np.int64))
            new_top_shape_tensor_name = nodes_serial[0].input[1] \
                if not find_other_node_by_input(onnx_model, nodes_serial[0], nodes_serial[0].input[0]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1])
            if new_top_shape_tensor is None:
                new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_tensor_name,
                                                               data_type=TensorProto.INT64,
                                                               dims=[len(top_out_shape)],
                                                               vals=top_out_shape)
                onnx_model.graph.initializer.append(new_top_shape_tensor)
            new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(blk_out_shape, dtype=np.int64))
            new_bot_shape_tensor_name = nodes_serial[-1].input[1] \
                if not find_other_node_by_input(onnx_model, nodes_serial[-1], nodes_serial[-1].input[0]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1])
            if new_bot_shape_tensor is None:
                new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_tensor_name,
                                                               data_type=TensorProto.INT64,
                                                               dims=[len(blk_out_shape)],
                                                               vals=blk_out_shape)
                onnx_model.graph.initializer.append(new_bot_shape_tensor)
            new_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name),
                                                    op_type='Reshape',
                                                    inputs=[next_node.output[0], new_top_shape_tensor.name],
                                                    outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[0].output[0])])
            new_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name),
                                                  op_type='Transpose',
                                                  inputs=[new_top_reshape.output[0]],
                                                  outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[1].output[0])],
                                                  perm=src_perm)
            new_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name),
                                                    op_type='Reshape',
                                                    inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                    outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[-1].output[0])])
            next_node.input[list(next_node.input).index(nodes_serial[-1].output[0])] = nodes_serial[0].input[0]
            onnx_model = delete_value_info_by_name(onnx_model, next_node.output[0])
            next_node_id = get_node_id(onnx_model, next_node)
            onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], next_node_id+1)
            for cur_node in second_next_nodes_list:
                for cur_idx, cur_input in enumerate(cur_node.input):
                    cur_node.input[cur_idx] = new_bot_reshape.output[0] if cur_input == next_node.output[0] else cur_input
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        else:
            if len(next_nodes_list) <= 1:
                for src_node in nodes_serial:
                    onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
                    onnx_model.graph.node.remove(src_node)
            new_top_shape_tensor_name = nodes_serial[0].input[1] \
                if find_other_node_by_input(onnx_model, nodes_serial[0], nodes_serial[0].input[1]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1])
            new_bot_shape_tensor_name = nodes_serial[-1].input[1] \
                if find_other_node_by_input(onnx_model, nodes_serial[-1], nodes_serial[-1].input[1]) \
                    else get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1])
            next_node.input[0] = nodes_serial[0].input[0]
            next_node.input[1] = nodes_serial[0].input[0]
            new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(top_out_shape, dtype=np.int64))
            if new_top_shape_tensor is None:
                new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_tensor_name,
                                                            data_type=TensorProto.INT64,
                                                            dims=[len(top_out_shape)],
                                                            vals=top_out_shape)
                onnx_model.graph.initializer.append(new_top_shape_tensor)
            new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(blk_out_shape, dtype=np.int64))
            if new_bot_shape_tensor is None:
                new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_tensor_name,
                                                               data_type=TensorProto.INT64,
                                                               dims=[len(blk_out_shape)],
                                                               vals=blk_out_shape)
                onnx_model.graph.initializer.append(new_bot_shape_tensor)
            new_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name),
                                                    op_type='Reshape',
                                                    inputs=[next_node.output[0], new_top_shape_tensor.name],
                                                    outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[0].output[0])])
            new_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name),
                                                  op_type='Transpose',
                                                  inputs=[new_top_reshape.output[0]],
                                                  outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[1].output[0])],
                                                  perm=src_perm)
            new_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name),
                                                    op_type='Reshape',
                                                    inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                    outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[-1].output[0])])
            for cur_node in second_next_nodes_list:
                for cur_idx, cur_input in enumerate(cur_node.input):
                    cur_node.input[cur_idx] = new_bot_reshape.output[0] if cur_input == next_node.output[0] else cur_input
            next_node_id = get_node_id(onnx_model, next_node)
            onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], next_node_id)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True 
        
    def forward_move_conv(onnx_model:onnx.ModelProto, nodes_serial:list, conv_node:onnx.NodeProto, src_perm:list):
        conv_attribute = attribute_to_dict(conv_node.attribute)
        dilations = conv_attribute.get('dilations', [1, 1])
        kernel_shape = conv_attribute.get('kernel_shape', [1, 1])
        pads = conv_attribute.get('pads', [0, 0, 0, 0])
        strides = conv_attribute.get('strides', [1, 1]) 
        conv_in_shape = get_shape_by_name(onnx_model, conv_node.input[0])
        conv_out_shape = get_shape_by_name(onnx_model, conv_node.output[0])
        if dilations != [1, 1] or kernel_shape != [1, 1] or pads != [0, 0, 0, 0] \
            or strides != [1, 1] or conv_in_shape[2:] != conv_out_shape[2:]:
                return onnx_model, False
        next_nodes_list = get_node_by_input(onnx_model, nodes_serial[-1].output)
        second_next_nodes_list = get_node_by_input(onnx_model, conv_node.output)
        top_out_shape = get_shape_by_name(onnx_model, nodes_serial[0].output[0])
        blk_out_shape = get_shape_by_name(onnx_model, nodes_serial[-1].output[0])
        
        if len(next_nodes_list) <= 1:
            for src_node in nodes_serial:
                onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
                onnx_model.graph.node.remove(src_node)
        conv_node.input[0] = nodes_serial[0].input[0]
        new_top_shape_tensor_name = nodes_serial[0].input[1] \
            if not find_other_node_by_input(onnx_model, nodes_serial[0], nodes_serial[0].input[1]) \
                else get_unique_node_tensor_name(onnx_model, nodes_serial[0].input[1])
        new_bot_shape_tensor_name = nodes_serial[-1].input[1] \
            if not find_other_node_by_input(onnx_model, nodes_serial[-1], nodes_serial[-1].input[1]) \
                else get_unique_node_tensor_name(onnx_model, nodes_serial[-1].input[1])
        new_top_shape = copy.deepcopy(top_out_shape)
        new_top_shape[1] = conv_out_shape[1]
        new_top_shape_tensor = get_initial_by_value(onnx_model, np.array(new_top_shape, dtype=np.int64))
        if new_top_shape_tensor is None:
            new_top_shape_tensor = onnx.helper.make_tensor(name=new_top_shape_tensor_name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(top_out_shape)],
                                                           vals=new_top_shape)
            onnx_model.graph.initializer.append(new_top_shape_tensor)
        new_bot_shape = copy.deepcopy(blk_out_shape)
        new_bot_shape[1] = conv_out_shape[1]
        new_bot_shape_tensor = get_initial_by_value(onnx_model, np.array(new_bot_shape, dtype=np.int64))
        if new_bot_shape_tensor is None:
            new_bot_shape_tensor = onnx.helper.make_tensor(name=new_bot_shape_tensor_name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(blk_out_shape)],
                                                           vals=new_bot_shape)
            onnx_model.graph.initializer.append(new_bot_shape_tensor)
        new_top_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[0].name),
                                                op_type='Reshape',
                                                inputs=[conv_node.output[0], new_top_shape_tensor.name],
                                                outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[0].output[0])])
        new_transpose = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[1].name),
                                              op_type='Transpose',
                                              inputs=[new_top_reshape.output[0]],
                                              outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[1].output[0])],
                                              perm=src_perm)
        new_bot_reshape = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, nodes_serial[-1].name),
                                                op_type='Reshape',
                                                inputs=[new_transpose.output[0], new_bot_shape_tensor.name],
                                                outputs=[get_unique_node_tensor_name(onnx_model, nodes_serial[-1].output[0])])
        conv_node_id = get_node_id(onnx_model, conv_node)
        for cur_node in second_next_nodes_list:
            for cur_idx, cur_input in enumerate(cur_node.input):
                cur_node.input[cur_idx] = new_bot_reshape.output[0] if cur_input == conv_node.output[0] else cur_input
        onnx_model = insert_node_by_list(onnx_model, [new_bot_reshape, new_transpose, new_top_reshape], conv_node_id+1)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True       
    
    def block_move(onnx_model, node):         
        if check_node_serial_group(onnx_model, node, ["Reshape", "Transpose", "Reshape"]):
            nodes_serial = get_node_serial_group(onnx_model, node, ["Reshape", "Transpose", "Reshape"]) 
            top_reshape_node, transpose_node, bot_reshape_node = nodes_serial
            blk_in_shape = get_shape_by_name(onnx_model, top_reshape_node.input[0])
            blk_out_shape = get_shape_by_name(onnx_model, bot_reshape_node.output[0])
            if len(blk_in_shape) != 4 or len(blk_out_shape) != 4 or blk_in_shape[:2] != blk_out_shape[:2]:
                return onnx_model, False
            top_out_shape = get_shape_by_name(onnx_model, top_reshape_node.output[0])
            if len(top_out_shape) != 6 or top_out_shape[:2] != blk_in_shape[:2] or blk_in_shape[0] != 1 \
                or top_out_shape[2]*top_out_shape[3] != blk_in_shape[2] or top_out_shape[4]*top_out_shape[5] != blk_in_shape[3]:
                    return onnx_model, False
            transpose_perm = attribute_to_dict(transpose_node.attribute).get('perm', list(range(len(top_out_shape))).reverse())
            if transpose_perm != [0, 1, 3, 5, 2, 4]:
                return onnx_model, False
            bot_in_shape = get_shape_by_name(onnx_model, bot_reshape_node.input[0])
            if bot_in_shape[2]*bot_in_shape[3] != blk_out_shape[2] or bot_in_shape[4]*bot_in_shape[5] != blk_out_shape[3]:
                return onnx_model, False
            
            next_nodes_list = get_node_by_input(onnx_model, bot_reshape_node.output)
            for next_node in next_nodes_list:
                if next_node.op_type == 'ReduceMean':
                    onnx_model, state = forward_move_reducemean(onnx_model, nodes_serial, next_node, transpose_perm)
                elif next_node.op_type in ['Sub', 'Add', 'Mul', 'Div']:
                    onnx_model, state = forward_move_subaddmuldiv(onnx_model, nodes_serial, next_node, transpose_perm)
                elif next_node.op_type == 'Conv':
                    onnx_model, state = forward_move_conv(onnx_model, nodes_serial, next_node, transpose_perm)
                else:
                    continue
                if state:
                    return onnx_model, state
        return onnx_model, False

    for node in onnx_model.graph.node:
        onnx_model, restart = block_move(onnx_model, node)
        if restart:
            onnx_model = infer_model_shape(onnx_model)
            return onnx_model, restart
    return onnx_model, False