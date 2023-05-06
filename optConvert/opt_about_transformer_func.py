from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convert3dimMultiAttentionKQVTo4dim(onnx_model, node, node_index):
    def transform_permute_dim(src_dim):
        if src_dim == [0, 2, 1, 3]:
            return [0, 1, 3, 2]
        elif src_dim == [0, 1, 3, 2]:
            return [0, 3, 2, 1]
        elif src_dim == [0, 3, 2, 1]:
            return [0, 2, 1, 3]
        elif src_dim == [0, 3, 1, 2]:
            return [0, 2, 3, 1]
        elif src_dim == [0, 2, 3, 1]:
            return [1]
        else:
            return None
    
    def set_transpose_perm(node, perm):
        node.attribute[0].ints[0] = perm[0]
        node.attribute[0].ints[1] = perm[1]
        node.attribute[0].ints[2] = perm[2]
        node.attribute[0].ints[3] = perm[3]
    
    if check_node_serial_group(onnx_model, node, ["MatMul", "Add", "Reshape", "Transpose"]):
        matMulNode, addNode, reshapeNode, transposeNode = get_node_serial_group(onnx_model, node, ["MatMul", "Add", "Reshape", "Transpose"])
        if not find_init_by_name(onnx_model, matMulNode.input[1]):
            return onnx_model, False
        addStaticInput, addDynamicInput = (addNode.input[0], addNode.input[1]) \
            if addNode.input[1] == matMulNode.output[0] else (addNode.input[1], addNode.input[0])
        if not find_init_by_name(onnx_model, addStaticInput):
            return onnx_model, False
        addStaticArr = get_tensor_from_initializer(onnx_model, addStaticInput)
        blkInShape = get_shape_by_name(onnx_model, matMulNode.input[0])
        matMulOutShape = get_shape_by_name(onnx_model, matMulNode.output[0])
        if len(blkInShape) != 3 or len(matMulOutShape) != 3 or blkInShape != matMulOutShape or addStaticArr.shape[-1] != matMulOutShape[-1]:
            return onnx_model, False
        addOutShape = get_shape_by_name(onnx_model, addNode.output[0])
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        transposeOutShape = get_shape_by_name(onnx_model, transposeNode.output[0])
        if addOutShape != matMulOutShape or len(reshapeOutShape) != 4 or len(transposeOutShape) != 4 or reshapeOutShape[:2] != addOutShape[:2]:
            return onnx_model, False
        transposePerm = attribute_to_dict(transposeNode.attribute).get("perm")
        newPreReshapeTensor = onnx.helper.make_tensor(name=matMulNode.name+"_insertReshapeShape",
                                                      data_type=TensorProto.INT64,
                                                      dims=[4],
                                                      vals=[blkInShape[0], blkInShape[1], 1, blkInShape[2]])
        newPreReshapeNode = onnx.helper.make_node(name=matMulNode.name+"_insertReshape",
                                                  op_type="Reshape",
                                                  inputs=[matMulNode.input[0], newPreReshapeTensor.name],
                                                  outputs=[matMulNode.name+"_insertReshapeOut"])
        newPreTransposeNode = onnx.helper.make_node(name=matMulNode.name+"_insertTranspose",
                                                    op_type="Transpose",
                                                    inputs=newPreReshapeNode.output,
                                                    outputs=[matMulNode.name+"_insertTransposeOut"],
                                                    perm=[0, 3, 2, 1])
        matMulInputArr = get_tensor_from_initializer(onnx_model, matMulNode.input[1])
        newMatMul2ConvWeightArr = matMulInputArr.transpose(1, 0)[:, :, np.newaxis, np.newaxis]
        newMatMul2ConvWeightTensor = onnx.helper.make_tensor(name=matMulNode.input[1]+"_2ConvWeight_"+matMulNode.name,
                                                             data_type=NPDTYPE_2_ONNXDTYPE[newMatMul2ConvWeightArr.dtype],
                                                             dims=newMatMul2ConvWeightArr.shape,
                                                             vals=newMatMul2ConvWeightArr.flatten().tolist())
        newMatMul2ConvAttr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newMatMul2ConvNode = onnx.helper.make_node(name=matMulNode.name+"_Conv",
                                                   op_type="Conv",
                                                   inputs=[newPreTransposeNode.output[0], newMatMul2ConvWeightTensor.name],
                                                   outputs=matMulNode.output,
                                                   **newMatMul2ConvAttr)
        if addStaticArr.shape == tuple(addOutShape):
            newAddStaticArr = np.expand_dims(addStaticArr.transpose(0, 2, 1), axis=2)
        elif addStaticArr.size == addOutShape[-1] and addStaticArr.shape[0] == addStaticArr.size:
            newAddStaticArr = addStaticArr[np.newaxis, :, np.newaxis, np.newaxis]
        else:
            return onnx_model, False
        newAddStaticTensor = onnx.helper.make_tensor(name=addStaticInput+"_newTensor_"+addNode.name,
                                                data_type=NPDTYPE_2_ONNXDTYPE[addStaticArr.dtype],
                                                dims=newAddStaticArr.shape,
                                                vals=newAddStaticArr.flatten().tolist())
        addNode.input[list(addNode.input).index(addStaticInput)] = newAddStaticTensor.name
        newAfterReshapeShapeTensor=onnx.helper.make_tensor(name=reshapeNode.input[1]+"_newShape_"+reshapeNode.name,
                                                           data_type=TensorProto.INT64,
                                                           dims=[4],
                                                           vals=[1, 8, int(addOutShape[-1]/8), addOutShape[1]])
        reshapeNode.input[1] = newAfterReshapeShapeTensor.name
        newTransposePerm = transform_permute_dim(transposePerm)
        if newTransposePerm is None:
            return onnx_model, False
        elif newTransposePerm == [1]:
            nodesListFromTransposeOut = get_node_by_input(onnx_model, transposeNode.output)
            for nodeFromTransposeOut in nodesListFromTransposeOut:
                nodeFromTransposeOut.input[list(nodeFromTransposeOut.input).index(transposeNode.output[0])] = reshapeNode.output[0]
            onnx_model.graph.node.remove(transposeNode)
        else:
            set_transpose_perm(transposeNode, newTransposePerm)
        onnx_model = delete_value_info_by_name(onnx_model, matMulNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, addNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, reshapeNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, transposeNode.output[0])
        onnx_model.graph.initializer.append(newPreReshapeTensor)
        onnx_model.graph.initializer.append(newMatMul2ConvWeightTensor)
        onnx_model.graph.initializer.append(newAddStaticTensor)
        onnx_model.graph.initializer.append(newAfterReshapeShapeTensor)
        onnx_model.graph.node.insert(node_index, newMatMul2ConvNode)
        onnx_model.graph.node.insert(node_index, newPreTransposeNode)
        onnx_model.graph.node.insert(node_index, newPreReshapeNode)
        onnx_model = delete_nodes(onnx_model, [matMulNode])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True        
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_splitMatMulQK2DynamicConv(onnx_model, node, node_index):
    def find_QxKT_block(onnx_model, matMulNode):
        if matMulNode.op_type != "MatMul" or \
            find_init_by_name(onnx_model, matMulNode.input[0]) or find_init_by_name(onnx_model, matMulNode.input[1]):
                return None
        QTransposeNode = get_node_by_output(onnx_model, matMulNode.input[0])
        KReshapeNode = get_node_by_output(onnx_model, matMulNode.input[1])
        QReshapeNode = get_node_by_output(onnx_model, QTransposeNode.input[0])
        if QTransposeNode.op_type != "Transpose" or KReshapeNode.op_type != "Reshape" or QReshapeNode.op_type != "Reshape":
            return None
        QReshapeInShape = get_shape_by_name(onnx_model, QReshapeNode.input[0])
        QReshapeOutShape = get_shape_by_name(onnx_model, QReshapeNode.output[0])
        KReshapeInShape = get_shape_by_name(onnx_model, KReshapeNode.input[0])
        KReshapeOutShape = get_shape_by_name(onnx_model, KReshapeNode.output[0])
        if QReshapeInShape != KReshapeInShape or QReshapeOutShape != KReshapeOutShape or len(QReshapeInShape) != 4 or \
            len(QReshapeOutShape) != 4 or len(KReshapeInShape) != 4 or len(KReshapeOutShape) != 4 or\
                QReshapeInShape[2] != 1 or QReshapeOutShape[1]*QReshapeOutShape[2] != QReshapeInShape[1]:
                return None
        transposePerm = attribute_to_dict(QTransposeNode.attribute).get("perm")
        if transposePerm != [0, 1, 3, 2]:
            return None
        return [matMulNode, QReshapeNode, QTransposeNode, KReshapeNode]
    
    QxKTNodes = find_QxKT_block(onnx_model, node)
    if QxKTNodes is None:
        return onnx_model, False
    matMulNode, QReshapeNode, QTransposeNode, KReshapeNode = QxKTNodes
    QMatMulOutShape = get_shape_by_name(onnx_model, matMulNode.output[0])
    QReshapeInShape = get_shape_by_name(onnx_model, QReshapeNode.input[0])
    concat_inputs = []
    slice_interval = QReshapeInShape[1] // QMatMulOutShape[1]
    slice_starts = 0
    slice_axes = 1
    slice_steps = 1
    sliceStartsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_position_0",
                                               data_type=TensorProto.INT64,
                                               dims=[1],
                                               vals=[slice_starts])
    sliceAxesTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_axes",
                                              data_type=TensorProto.INT64,
                                              dims=[1],
                                              vals=[slice_axes])
    sliceStepsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_steps",
                                               data_type=TensorProto.INT64,
                                               dims=[1],
                                               vals=[slice_steps])
    onnx_model.graph.initializer.append(sliceStartsTensor)
    onnx_model.graph.initializer.append(sliceAxesTensor)
    onnx_model.graph.initializer.append(sliceStepsTensor)
    insertNodesDict={}
    for slice_num in range(QMatMulOutShape[1]):
        slice_ends = slice_starts + slice_interval
        sliceEndsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_position_%d"%(slice_num+1),
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[slice_ends])
        onnx_model.graph.initializer.append(sliceEndsTensor)
        newQSliceInputs = [QReshapeNode.input[0], sliceStartsTensor.name, sliceEndsTensor.name, sliceAxesTensor.name, sliceStepsTensor.name]
        newQSliceNode = onnx.helper.make_node(name=matMulNode.name+"_QInsertSlice_%d"%slice_num,
                                              op_type="Slice",
                                              inputs=newQSliceInputs,
                                              outputs=[QReshapeNode.input[0]+"_sliceOut_%d"%slice_num])
        newKSliceInputs = [KReshapeNode.input[0], sliceStartsTensor.name, sliceEndsTensor.name, sliceAxesTensor.name, sliceStepsTensor.name]
        newKSliceNode = onnx.helper.make_node(name=matMulNode.name+"_KInsertSlice_%d"%slice_num,
                                        op_type="Slice",
                                        inputs=newKSliceInputs,
                                        outputs=[KReshapeNode.input[0]+"_sliceOut_%d"%slice_num])
        newQSliceReshapeTensor = onnx.helper.make_tensor(name=QReshapeNode.name+"_sliceShape_%d"%slice_num,
                                                         data_type=TensorProto.INT64,
                                                         dims=[4],
                                                         vals=[slice_interval, QReshapeInShape[-1], 1, 1])
        onnx_model.graph.initializer.append(newQSliceReshapeTensor)
        newQSliceReshapeNode = onnx.helper.make_node(name=QReshapeNode.name+"_slice_%d"%slice_num,
                                                     op_type="Reshape",
                                                     inputs=[newQSliceNode.output[0], newQSliceReshapeTensor.name],
                                                     outputs=[QReshapeNode.output[0]+"_slice_%d"%slice_num])
        newQSliceTransposeNode = onnx.helper.make_node(name=QTransposeNode.name+"_slice_%d"%slice_num,
                                                       op_type="Transpose",
                                                       inputs=newQSliceReshapeNode.output,
                                                       outputs=[QTransposeNode.output[0]+"_slice_%d"%slice_num],
                                                       perm=[1, 0, 2, 3])
        newConvAttribute = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newSliceConvNode = onnx.helper.make_node(name=matMulNode.name+"_slice_Conv_%d"%slice_num,
                                                 op_type="Conv",
                                                 inputs=[newKSliceNode.output[0], newQSliceTransposeNode.output[0]],
                                                 outputs=[matMulNode.output[0]+"_slice_conv_%d"%slice_num],
                                                 **newConvAttribute)
        insertNodesDict[slice_num]=[newSliceConvNode, newKSliceNode, newQSliceTransposeNode, newQSliceReshapeNode, newQSliceNode]
        concat_inputs.append(newSliceConvNode.output[0])
        slice_starts = slice_ends
        sliceStartsTensor = sliceEndsTensor
    newConcatNode=onnx.helper.make_node(name=matMulNode.name+"_Slice_Concat",
                                        op_type="Concat",
                                        inputs=concat_inputs,
                                        outputs=[matMulNode.name+"_slice_concat"],
                                        axis=2)
    newAfterTransposeNode = onnx.helper.make_node(name=matMulNode.name+"_Slice_After_Transpose",
                                                  op_type="Transpose",
                                                  inputs=newConcatNode.output,
                                                  outputs=matMulNode.output,
                                                  perm=[0, 2, 1, 3])
    onnx_model.graph.node.insert(node_index, newAfterTransposeNode)
    onnx_model.graph.node.insert(node_index, newConcatNode)
    for num in insertNodesDict:
        onnx_model = insert_node_by_list(onnx_model, insertNodesDict[num], node_index) 
    onnx_model = delete_nodes(onnx_model, [QReshapeNode, QTransposeNode, KReshapeNode, matMulNode])
    onnx_model = delete_useless_value_info(onnx_model)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    
    return onnx_model, True  