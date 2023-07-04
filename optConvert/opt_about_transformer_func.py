from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *
from extraFunc.about_transformer_extra import *

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
        inputNode = get_node_by_output(onnx_model, matMulNode.input[0])
        addOutShape = get_shape_by_name(onnx_model, addNode.output[0])
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        transposeOutShape = get_shape_by_name(onnx_model, transposeNode.output[0])
        if addOutShape != matMulOutShape or len(reshapeOutShape) != 4 or len(transposeOutShape) != 4 or reshapeOutShape[:2] != addOutShape[:2]:
            return onnx_model, False
        transposePerm = attribute_to_dict(transposeNode.attribute).get("perm")
        newPreReshapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, inputNode.name+"_insertReshapeShape"),
                                                      data_type=TensorProto.INT64,
                                                      dims=[4],
                                                      vals=[blkInShape[0], blkInShape[1], 1, blkInShape[2]])
        newPreReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, inputNode.name+"_Reshape"),
                                                  op_type="Reshape",
                                                  inputs=[matMulNode.input[0], newPreReshapeTensor.name],
                                                  outputs=[matMulNode.name+"_insertReshapeOut"])
        newPreTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, inputNode.name+"_Transpose"),
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
                                                   outputs=[matMulNode.output[0]+"_toConv"],
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
        #addNode.input[list(addNode.input).index(addStaticInput)] = newAddStaticTensor.name
        newAddNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, addNode.name+"_new"),
                                           op_type="Add",
                                           inputs=[newMatMul2ConvNode.output[0], newAddStaticTensor.name],
                                           outputs=[get_unique_node_tensor_name(onnx_model, addNode.output[0]+"_new")])
        newAfterReshapeShapeTensor=onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, reshapeNode.input[1]+"_newShape"),
                                                           data_type=TensorProto.INT64,
                                                           dims=[4],
                                                           vals=[1, 8, int(addOutShape[-1]/8), addOutShape[1]])
        #reshapeNode.input[1] = newAfterReshapeShapeTensor.name
        newReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, reshapeNode.name+"_new"),
                                               op_type="Reshape",
                                               inputs=[newAddNode.output[0], newAfterReshapeShapeTensor.name],
                                               outputs=[get_unique_node_tensor_name(onnx_model, reshapeNode.output[0]+"_new")])
        newTransposePerm = transform_permute_dim(transposePerm)
        if newTransposePerm is None:
            return onnx_model, False
        elif newTransposePerm == [1]:
            nodesListFromTransposeOut = get_node_by_input(onnx_model, transposeNode.output)
            for nodeFromTransposeOut in nodesListFromTransposeOut:
                nodeFromTransposeOut.input[list(nodeFromTransposeOut.input).index(transposeNode.output[0])] = newReshapeNode.output[0]
            onnx_model.graph.node.remove(transposeNode)
        else:
            set_transpose_perm(transposeNode, newTransposePerm)
            transposeNode.input[0] = newReshapeNode.output[0]
        onnx_model.graph.initializer.append(newPreReshapeTensor)
        onnx_model.graph.initializer.append(newMatMul2ConvWeightTensor)
        onnx_model.graph.initializer.append(newAddStaticTensor)
        onnx_model.graph.initializer.append(newAfterReshapeShapeTensor)
        onnx_model.graph.node.insert(node_index, newMatMul2ConvNode)
        onnx_model.graph.node.insert(node_index, newPreTransposeNode)
        onnx_model.graph.node.insert(node_index, newPreReshapeNode)
        onnx_model = insert_node_by_list(onnx_model, [newReshapeNode, newAddNode], get_node_id(onnx_model, reshapeNode))
        reshapeOutNodes = get_node_by_input(onnx_model, reshapeNode.output)
        if len(reshapeOutNodes) == 0:
            onnx_model = delete_nodes(onnx_model, [reshapeNode])
        addOutNodes = get_node_by_input(onnx_model, addNode.output)
        if len(addOutNodes) == 0:
            onnx_model = delete_nodes(onnx_model, [addNode])
        onnx_model = delete_nodes(onnx_model, [matMulNode])
        onnx_model = delete_useless_value_info(onnx_model)
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
        newKSliceTransposeNode = onnx.helper.make_node(name=QTransposeNode.name+"_slice_%d"%slice_num,
                                                       op_type="Transpose",
                                                       inputs=newKSliceNode.output,
                                                       outputs=[QTransposeNode.output[0]+"_slice_%d"%slice_num],
                                                       perm=[3, 1, 2, 0])
        newConvAttribute = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newSliceConvNode = onnx.helper.make_node(name=matMulNode.name+"_slice_Conv_%d"%slice_num,
                                                 op_type="Conv",
                                                 inputs=[newQSliceNode.output[0], newKSliceTransposeNode.output[0]],
                                                 outputs=[matMulNode.output[0]+"_slice_conv_%d"%slice_num],
                                                 **newConvAttribute)
        insertNodesDict[slice_num]=[newSliceConvNode, newKSliceTransposeNode, newKSliceNode, newQSliceNode]
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
                                                  perm=[0, 2, 3, 1])
    onnx_model.graph.node.insert(node_index, newAfterTransposeNode)
    onnx_model.graph.node.insert(node_index, newConcatNode)
    for num in insertNodesDict:
        onnx_model = insert_node_by_list(onnx_model, insertNodesDict[len(insertNodesDict)-num-1], node_index)
    delNodesList = [matMulNode]
    QTransposeOutNodes = get_node_by_input(onnx_model, QTransposeNode.output)
    QReshapeOutNodes = get_node_by_input(onnx_model, QReshapeNode.output)
    if len(QTransposeOutNodes) == 1:
        delNodesList.append(QTransposeNode)
        if len(QReshapeOutNodes) == 1:
            delNodesList.append(QReshapeNode)
    KReshapeOutNodes = get_node_by_input(onnx_model, KReshapeNode.output)
    if len(KReshapeOutNodes) == 1:
        delNodesList.append(KReshapeNode)
    onnx_model = delete_nodes(onnx_model, delNodesList)
    onnx_model = delete_useless_value_info(onnx_model)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    
    return onnx_model, True 

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_splitVxSoftmax2DynamicConv(onnx_model, node, node_index):
    def find_VxSoftmax_block(onnx_model, matMulNode):
        if matMulNode.op_type != "MatMul" or \
            find_init_by_name(onnx_model, matMulNode.input[0]) or find_init_by_name(onnx_model, matMulNode.input[1]):
                return None
        VTransposeNode = get_node_by_output(onnx_model, matMulNode.input[1])
        VReshapeNode = get_node_by_output(onnx_model, VTransposeNode.input[0])
        SoftmaxNode = get_node_by_output(onnx_model, matMulNode.input[0])
        if VTransposeNode.op_type != "Transpose" or VReshapeNode.op_type != "Reshape" or SoftmaxNode.op_type != "Softmax":
            return None
        VReshapeInShape = get_shape_by_name(onnx_model, VReshapeNode.input[0])
        VReshapeOutShape = get_shape_by_name(onnx_model, VReshapeNode.output[0])
        SoftmaxInShape = get_shape_by_name(onnx_model, SoftmaxNode.input[0])
        if len(VReshapeInShape) != 4 or len(VReshapeOutShape) != 4 or len(SoftmaxInShape) != 4 or \
            VReshapeInShape[1] != VReshapeOutShape[1]*VReshapeOutShape[2] or VReshapeOutShape[-1] != SoftmaxInShape[-1]:
                return None
        VTransposePerm = attribute_to_dict(VTransposeNode.attribute).get("perm")
        SoftmaxAxis = attribute_to_dict(SoftmaxNode.attribute).get("axis", 1)
        SoftmaxAxis = len(SoftmaxInShape) + SoftmaxAxis if SoftmaxAxis < 0 else SoftmaxAxis
        if VTransposePerm != [0, 1, 3, 2] or SoftmaxAxis != 3:
            return None
        return [matMulNode, VTransposeNode, VReshapeNode, SoftmaxNode]
    
    VxSoftmaxNodes = find_VxSoftmax_block(onnx_model, node)
    if VxSoftmaxNodes is None:
        return onnx_model, False
    matMulNode, VTransposeNode, VReshapeNode, SoftmaxNode = VxSoftmaxNodes
    
    softmaxPreTransposeNode = onnx.helper.make_node(name=SoftmaxNode.name+"_PerTranspose",
                                                    op_type="Transpose",
                                                    inputs=SoftmaxNode.input,
                                                    outputs=[SoftmaxNode.output[0]+"_PreTranspose"],
                                                    perm=[0, 3, 1, 2])
    SoftmaxNode.input[0] = softmaxPreTransposeNode.output[0]
    SoftmaxNode.attribute[0].i = 1
    SoftmaxOutNodes = get_node_by_input(onnx_model, SoftmaxNode.output[0])
    if len(SoftmaxOutNodes) > 1:
        SoftmaxOutNodes = SoftmaxOutNodes.remove(matMulNode) if matMulNode in SoftmaxOutNodes else SoftmaxOutNodes
        softmaxAfterTransposeNode = onnx.helper.make_node(name=SoftmaxNode.name+"_AfterTranspose",
                                                          op_type="Transpose",
                                                          inputs=SoftmaxNode.output,
                                                          outputs=[SoftmaxNode.output[0]+"_AfterTranspose"],
                                                          perm=[0, 2, 3, 1])
        for SoftmaxOutNode in SoftmaxOutNodes:
            SoftmaxOutNode.input[list(SoftmaxOutNode.input).index(SoftmaxNode.output[0])] = softmaxAfterTransposeNode.output[0]
    onnx_model = delete_value_info_by_name(onnx_model, SoftmaxNode.output[0])
    
    VReshapeOutShape = get_shape_by_name(onnx_model, VReshapeNode.input[0])
    matMulOutShape = get_shape_by_name(onnx_model, matMulNode.output[0])
    slice_interval = VReshapeOutShape[1] // matMulOutShape[1]
    slice_starts = 0
    softmaxSlice_starts = 0
    v_slice_axes = 1
    softmax_slice_axes = 2
    slice_steps = 1
    sliceStartsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_position_0",
                                               data_type=TensorProto.INT64,
                                               dims=[1],
                                               vals=[slice_starts])
    sliceVAxesTensor = onnx.helper.make_tensor(name=matMulNode.name+"_v_slice_axes",
                                              data_type=TensorProto.INT64,
                                              dims=[1],
                                              vals=[v_slice_axes])
    sliceSoftmaxAxesTensor = onnx.helper.make_tensor(name=matMulNode.name+"_softmax_slice_axes",
                                                     data_type=TensorProto.INT64,
                                                     dims=[1],
                                                     vals=[softmax_slice_axes])
    sliceStepsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_steps",
                                               data_type=TensorProto.INT64,
                                               dims=[1],
                                               vals=[slice_steps])
    onnx_model.graph.initializer.append(sliceStartsTensor)
    onnx_model.graph.initializer.append(sliceVAxesTensor)
    onnx_model.graph.initializer.append(sliceSoftmaxAxesTensor)
    onnx_model.graph.initializer.append(sliceStepsTensor)
    softmaxSliceStartsTensor = sliceStartsTensor
    insertNodesDict = {}
    concat_inputs = []
    for slice_num in range(matMulOutShape[1]):
        slice_ends = slice_starts + slice_interval
        softmax_slice_ends = softmaxSlice_starts + 1
        sliceEndsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_position_%d"%(slice_num+1),
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[slice_ends])
        softmaxSliceEndsTensor = onnx.helper.make_tensor(name=matMulNode.name+"_softmaxSlice_position_%d"%(slice_num+1),
                                                         data_type=TensorProto.INT64,
                                                         dims=[1],
                                                         vals=[softmax_slice_ends])
        onnx_model.graph.initializer.append(sliceEndsTensor)
        onnx_model.graph.initializer.append(softmaxSliceEndsTensor)
        newVSliceInputs = [VReshapeNode.input[0], sliceStartsTensor.name, sliceEndsTensor.name, sliceVAxesTensor.name, sliceStepsTensor.name]
        newVSliceNode = onnx.helper.make_node(name=matMulNode.name+"_VInsertSlice_%d"%slice_num,
                                              op_type="Slice",
                                              inputs=newVSliceInputs,
                                              outputs=[VReshapeNode.input[0]+"_sliceOut_%d"%slice_num])
        newSoftmaxSliceInputs = [SoftmaxNode.output[0], softmaxSliceStartsTensor.name, softmaxSliceEndsTensor.name, sliceSoftmaxAxesTensor.name, sliceStepsTensor.name]
        newSoftmaxSliceNode = onnx.helper.make_node(name=matMulNode.name+"_SoftmaxInsertSlice_%d"%slice_num,
                                        op_type="Slice",
                                        inputs=newSoftmaxSliceInputs,
                                        outputs=[SoftmaxNode.input[0]+"_sliceOut_%d"%slice_num])
        newVSliceReshapeShapeTensor = onnx.helper.make_tensor(name=VReshapeNode.name+"_sliceShape_%d"%slice_num,
                                                              data_type=TensorProto.INT64,
                                                              dims=[4],
                                                              vals=[slice_interval, VReshapeOutShape[-1], 1, -1])
        onnx_model.graph.initializer.append(newVSliceReshapeShapeTensor)
        newVSliceReshapeNode = onnx.helper.make_node(name=VReshapeNode.name+"_slice_%d"%slice_num,
                                                       op_type="Reshape",
                                                       inputs=[newVSliceNode.output[0], newVSliceReshapeShapeTensor.name],
                                                       outputs=[VReshapeNode.output[0]+"_slice_%d"%slice_num])
        newConvAttribute = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newSliceConvNode = onnx.helper.make_node(name=matMulNode.name+"_slice_Conv_%d"%slice_num,
                                                 op_type="Conv",
                                                 inputs=[newSoftmaxSliceNode.output[0], newVSliceReshapeNode.output[0]],
                                                 outputs=[matMulNode.output[0]+"_slice_conv_%d"%slice_num],
                                                 **newConvAttribute)
        insertNodesDict[slice_num]=[newSliceConvNode, newVSliceReshapeNode, newVSliceNode, newSoftmaxSliceNode]
        concat_inputs.append(newSliceConvNode.output[0])
        slice_starts = slice_ends
        softmaxSlice_starts = softmax_slice_ends
        sliceStartsTensor = sliceEndsTensor
        softmaxSliceStartsTensor = softmaxSliceEndsTensor
    newConcatNode=onnx.helper.make_node(name=matMulNode.name+"_Slice_Concat",
                                        op_type="Concat",
                                        inputs=concat_inputs,
                                        outputs=[matMulNode.name+"_slice_concat"],
                                        axis=2)
    newAfterTransposeNode = onnx.helper.make_node(name=matMulNode.name+"_Slice_After_Transpose",
                                                  op_type="Transpose",
                                                  inputs=newConcatNode.output,
                                                  outputs=matMulNode.output,
                                                  perm=[0, 2, 3, 1])
    onnx_model.graph.node.insert(node_index, newAfterTransposeNode)
    onnx_model.graph.node.insert(node_index, newConcatNode)
    for num in insertNodesDict:
        onnx_model = insert_node_by_list(onnx_model, insertNodesDict[len(insertNodesDict)-num-1], node_index)
    softmaxNodeId = get_node_id(onnx_model, SoftmaxNode)
    onnx_model.graph.node.insert(softmaxNodeId, softmaxPreTransposeNode)
    delNodesList = [matMulNode]
    VTransposeOutNodes = get_node_by_input(onnx_model, VTransposeNode.output)
    VReshapeOutNodes = get_node_by_input(onnx_model, VReshapeNode.output)
    if len(VTransposeOutNodes) == 1:
        delNodesList.append(VTransposeNode)
        if len(VReshapeOutNodes) == 1:
            delNodesList.append(VReshapeNode)
    onnx_model = delete_nodes(onnx_model, delNodesList)
    onnx_model = delete_useless_value_info(onnx_model)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
      
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_3dimMultiAttentionxWto4dimConv(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Reshape", "MatMul", "Add"]):
        reshapeNode, matMulNode, addNode = get_node_serial_group(onnx_model, node, ["Reshape", "MatMul", "Add"]) 
        reshapeInShape = get_shape_by_name(onnx_model, reshapeNode.input[0])
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        matMulOutShape = get_shape_by_name(onnx_model, matMulNode.output[0])
        addOutShape = get_shape_by_name(onnx_model, addNode.output[0])
        if len(reshapeInShape) != 4 or len(reshapeOutShape) != 3 or len(matMulOutShape) != 3 or len(addOutShape) != 3 or \
            reshapeOutShape[:2] != reshapeInShape[:2] or reshapeOutShape[-1] != reshapeInShape[2]*reshapeInShape[3] or \
                matMulOutShape != addOutShape or not find_init_by_name(onnx_model, matMulNode.input[1]):
                    return onnx_model, False
        addOtherInput = addNode.input[0] if matMulNode.output[0] == addNode.input[1] else addNode.input[1]
        addOtherInputShape = get_shape_by_name(onnx_model, addOtherInput)
        if len(addOtherInputShape) != 1 and len(addOtherInputShape) != 3:
            return onnx_model, False
        sliceNumIndex = 2 + list(reshapeInShape[2:]).index(min(reshapeInShape[2:]))
        sliceNum = reshapeInShape[sliceNumIndex]
        newPrePerm = [0, 3, 2, 1] if sliceNumIndex == 2 else [0, 2, 1, 3]
        newPreTransposeNode = onnx.helper.make_node(name=reshapeNode.name+"_toTranspose",
                                                    op_type="Transpose",
                                                    inputs=[reshapeNode.input[0]],
                                                    outputs=[reshapeNode.output[0]+"_toTranspose"],
                                                    perm=newPrePerm)
        slice_starts = 0
        slice_axes = sliceNumIndex
        slice_steps = 1
        sliceStartTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_position_0",
                                                   data_type=TensorProto.INT64,
                                                   dims=[1],
                                                   vals=[slice_starts])
        sliceAxesTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_axes",
                                                  data_type=TensorProto.INT64,
                                                  dims=[1],
                                                  vals=[slice_axes])
        sliceStepTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_step",
                                                  data_type=TensorProto.INT64,
                                                  dims=[1],
                                                  vals=[slice_steps])
        onnx_model.graph.initializer.append(sliceStartTensor)
        onnx_model.graph.initializer.append(sliceAxesTensor)
        onnx_model.graph.initializer.append(sliceStepTensor)
        matMulStaticArr = get_tensor_from_initializer(onnx_model, matMulNode.input[1])
        if len(matMulStaticArr.shape) != 2:
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, False
        matMulStaticArr = matMulStaticArr.reshape(reshapeInShape[2], reshapeInShape[3], matMulStaticArr.shape[-1])
        matMulSliceIndex = 0 if sliceNumIndex == 2 else 1
        newConvAttribute = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        insertSliceNodesList = []
        insertConvNodesList = []
        insertAddNodesList = []
        for slice_id in range(sliceNum):
            slice_ends = slice_starts + 1
            sliceEndTensor = onnx.helper.make_tensor(name=matMulNode.name+"_slice_position_%d"%(slice_id+1),
                                                     data_type=TensorProto.INT64,
                                                     dims=[1],
                                                     vals=[slice_ends])
            onnx_model.graph.initializer.append(sliceEndTensor)
            newSliceInputs = [newPreTransposeNode.output[0], sliceStartTensor.name, sliceEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
            newSliceNode = onnx.helper.make_node(name=matMulNode.name+"_insertSlice_%d"%slice_id,
                                                 op_type="Slice",
                                                 inputs=newSliceInputs,
                                                 outputs=[matMulNode.input[0]+"_sliceOut_%d"%slice_id])
            insertSliceNodesList.append(newSliceNode)
            if matMulSliceIndex:
                matMulSliceStaticArr = matMulStaticArr[:, slice_id, :].reshape(-1, matMulStaticArr.shape[-1])
            else:
                matMulSliceStaticArr = matMulStaticArr[slice_id, :, :].reshape(-1, matMulStaticArr.shape[-1])
            matMulSliceStaticArr = np.transpose(matMulSliceStaticArr, (1, 0))[:, :, np.newaxis, np.newaxis]
            matMulSliceStaticTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, matMulNode.input[1]+"_slice_%d"%slice_id),
                                                              data_type=NPDTYPE_2_ONNXDTYPE[str(matMulSliceStaticArr.dtype)],
                                                              dims=matMulSliceStaticArr.shape,
                                                              vals=matMulSliceStaticArr.flatten().tolist())
            onnx_model.graph.initializer.append(matMulSliceStaticTensor)
            newConvNode = onnx.helper.make_node(name=matMulNode.name+"_toConv_%d"%slice_id,
                                                op_type="Conv",
                                                inputs=[newSliceNode.output[0], matMulSliceStaticTensor.name],
                                                outputs=[matMulNode.output[0]+"_slice_%d"%slice_id],
                                                **newConvAttribute)
            insertConvNodesList.append(newConvNode)
            slice_starts = slice_ends
            sliceStartTensor = sliceEndTensor
            if slice_id == 1:
                newAddInputs = [insertConvNodesList[0].output[0], newConvNode.output[0]]
            elif slice_id > 1:
                newAddInputs = [insertAddNodesList[slice_id-2].output[0], newConvNode.output[0]]
            else:
                continue
            newAddNode = onnx.helper.make_node(name=matMulNode.name+"_add_%d"%slice_id,
                                    op_type="Add",
                                    inputs=newAddInputs,
                                    outputs=[matMulNode.output[0]+"_addOut_%d"%slice_id])
            insertAddNodesList.append(newAddNode)
        #insertAddNodesList[-1].output[0] = matMulNode.output[0]
        addOtherInputArr = get_tensor_from_initializer(onnx_model, addOtherInput)
        if addOtherInputArr.size:
            newAddOtherInputArr = addOtherInputArr.reshape(1, -1, 1, 1) if len(addOtherInputArr.shape) == 1 \
                else np.expand_dims(np.transpose(addOtherInputArr, (0, 2, 1)), sliceNumIndex-1)
            newAddOtherInputTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, addOtherInput+"_new"),
                                                             data_type=NPDTYPE_2_ONNXDTYPE[newAddOtherInputArr.dtype],
                                                             dims=newAddOtherInputArr.shape,
                                                             vals=newAddOtherInputArr.flatten().tolist())
            onnx_model.graph.initializer.append(newAddOtherInputTensor)
            newAddOtherInput = newAddOtherInputTensor.name
        elif len(addOtherInputShape) == 1:
            newAddOtherReshapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, addOtherInput+"_newShape"),
                                                               data_type=TensorProto.INT64,
                                                               dims=[4],
                                                               vals=[1, addOtherInputShape[0], 1, 1])
            onnx_model.graph.initializer.append(newAddOtherReshapeTensor)
            newAddOtherReshapeNode = onnx.helper.make_node(name=addNode.name+"_newReshape",
                                                           op_type="Reshape",
                                                           inputs=[addOtherInput, newAddOtherReshapeTensor.name],
                                                           outputs=[addOtherInput+"_newReshape"])
            addNode_id = get_node_id(onnx_model, addNode)
            onnx_model.graph.node.insert(addNode_id, newAddOtherReshapeNode)
            newAddOtherInput = newAddOtherReshapeNode.output[0]
        else:
            newReshapeShape = np.expand_dims(np.array(addOtherInputShape, dtype=np.int64), sliceNumIndex - 1)
            newTransposePerm = [0, 2, 1, 3] if matMulSliceIndex else [0, 3, 2, 1]
            newAddOtherReshapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, addOtherInput+"_newShape"),
                                                               data_type=TensorProto.INT64,
                                                               dims=[4],
                                                               vals=newReshapeShape.tolist())
            onnx_model.graph.initializer.append(newAddOtherReshapeTensor)
            newAddOtherReshapeNode = onnx.helper.make_node(name=addNode.name+"_input_newReshape",
                                                           op_type="Reshape",
                                                           inputs=[addOtherInput, newAddOtherReshapeTensor.name],
                                                           outputs=[addOtherInput+"_newReshape"])
            newAddOtherTransposeNode = onnx.helper.make_node(name=addNode.name+"_input_newTranspose",
                                                             op_type="Transpose",
                                                             inputs=newAddOtherReshapeNode.output,
                                                             outputs=[addOtherInput+"_newTranspose"],
                                                             perm=newTransposePerm)
            addNode_id = get_node_id(onnx_model, addNode)
            onnx_model.graph.node.insert(addNode_id, newAddOtherTransposeNode)
            onnx_model.graph.node.insert(addNode_id, newAddOtherReshapeNode)
            newAddOtherInput = newAddOtherTransposeNode.output[0]
        newAddNode = onnx.helper.make_node(name=addNode.name+"_new",
                                           op_type="Add",
                                           inputs=[insertAddNodesList[-1].output[0], newAddOtherInput],
                                           outputs=[addNode.output[0]+"_4dimNew"])
        newAfterTransposePerm = [0, 2, 1, 3] if matMulSliceIndex else [0, 3, 2, 1]
        newAfterTransposeNode = onnx.helper.make_node(name=addNode.name+"_after_transpose",
                                                      op_type="Transpose",
                                                      inputs=newAddNode.output,
                                                      outputs=[addNode.output[0]+"_after_transpose_out"],
                                                      perm=newAfterTransposePerm)
        newAfterReshapeTensor = onnx.helper.make_tensor(name=addNode.name+"_after_reshapeShape",
                                                        data_type=TensorProto.INT64,
                                                        dims=[len(addOutShape)],
                                                        vals=addOutShape)
        onnx_model.graph.initializer.append(newAfterReshapeTensor)
        newAfterReshapeNode = onnx.helper.make_node(name=addNode.name+"_after_reshape",
                                                    op_type="Reshape",
                                                    inputs=[newAfterTransposeNode.output[0], newAfterReshapeTensor.name],
                                                    outputs=addNode.output)
        addNode_id = get_node_id(onnx_model, addNode)
        onnx_model = insert_node_by_list(onnx_model, [newAfterReshapeNode, newAfterTransposeNode, newAddNode], addNode_id)
        insertAddNodesList.reverse()
        #insertAddNodesList = insertAddNodesList[::-1]
        onnx_model = insert_node_by_list(onnx_model, insertAddNodesList, node_index)
        onnx_model = insert_node_by_list(onnx_model, insertConvNodesList, node_index)
        onnx_model = insert_node_by_list(onnx_model, insertSliceNodesList, node_index)
        onnx_model.graph.node.insert(node_index, newPreTransposeNode)
        delNodesList = [addNode]
        if len(get_node_by_input(onnx_model, matMulNode.output)) == 1:
            delNodesList.append(matMulNode)
            if len(get_node_by_input(onnx_model, reshapeNode.output)) == 1:
                delNodesList.append(reshapeNode)
        onnx_model = delete_nodes(onnx_model, delNodesList)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        onnx_model = delete_useless_value_info(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_3dimResidualAddTo4dim(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Add"]):
        bottomAddNode = get_node_serial_group(onnx_model, node, ["Add"])[0]
        if find_init_by_name(onnx_model, bottomAddNode.input[0]) or find_init_by_name(onnx_model, bottomAddNode.input[1]):
            return onnx_model, False
        bottomAddOutShape = get_shape_by_name(onnx_model, bottomAddNode.output[0])
        if len(bottomAddOutShape) != 3:
            return onnx_model, False
        topAddNode = get_node_by_output(onnx_model, bottomAddNode.input[0])
        reshapeNode = get_node_by_output(onnx_model, bottomAddNode.input[1])
        topAddNode = topAddNode if topAddNode.op_type == "Add" else reshapeNode
        reshapeNode = reshapeNode if reshapeNode.op_type == "Reshape" else topAddNode
        if topAddNode.op_type != "Add" or reshapeNode.op_type != "Reshape":
            return onnx_model, False
        topAddOutShape = get_shape_by_name(onnx_model, topAddNode.output[0])
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        if topAddOutShape != reshapeOutShape:
            return onnx_model, False
        reshapeInNode = get_node_by_output(onnx_model, reshapeNode.input[0])
        reshapeInShape = get_shape_by_name(onnx_model, reshapeNode.input[0])
        if len(reshapeInShape) != 4 or reshapeInShape[:2] != reshapeOutShape[:2] or reshapeInShape[0] != 1:
            return onnx_model, False
        topAddNodeId = get_node_id(onnx_model, topAddNode) 
        topAddOutNodesList = get_node_by_input(onnx_model, topAddNode.output)
        bottomAddOutNodesList = get_node_by_input(onnx_model, bottomAddNode.output)
        transposeAttr = attribute_to_dict(reshapeInNode.attribute)
        transposePerm = transposeAttr.get("perm", None)
        for topAddInput in topAddNode.input:
            topAddInShape = get_shape_by_name(onnx_model, topAddInput)
            if len(topAddInShape) == 1 and topAddInShape[0] == topAddOutShape[-1]:
                newReshapeShape = [1, 1, reshapeInShape[2], reshapeInShape[3]]
            elif len(topAddInShape) == 3 and topAddInShape == topAddOutShape:
                newReshapeShape = reshapeInShape
            elif len(topAddInShape) == 3 and topAddInShape[-1] == 1:
                newReshapeShape = [1, topAddInShape[1], 1, 1]
            elif len(topAddInShape) == 3 and topAddInShape[1] == 1:
                newReshapeShape = [1, 1, reshapeInShape[2], reshapeInShape[3]]
            else:
                return onnx_model, False
            if not find_init_by_name(onnx_model, topAddInput):
                newTopPreReshapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, topAddInput+"_shape"),
                                                            data_type=TensorProto.INT64,
                                                            dims=[4],
                                                            vals=newReshapeShape)
                onnx_model.graph.initializer.append(newTopPreReshapeTensor)
                newTopPreReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, topAddNode.name+"_preReshape"),
                                                        op_type="Reshape",
                                                        inputs=[topAddInput, newTopPreReshapeTensor.name],
                                                        outputs=[get_unique_node_tensor_name(onnx_model, topAddInput+"_preReshapeOut")])
                topAddNodeNewInput = newTopPreReshapeNode.output[0]
                if transposePerm is not None:
                    newTopPreTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, topAddNode.name+"_preTranspose"),
                                                                   op_type="Transpose",
                                                                   inputs=newTopPreReshapeNode.output,
                                                                   outputs=[get_unique_node_tensor_name(onnx_model, topAddInput+"_preTransposeOut")],
                                                                   perm=transposePerm)
                    topAddNodeNewInput = newTopPreTransposeNode.output[0]
                    onnx_model.graph.node.insert(topAddNodeId, newTopPreTransposeNode)
                onnx_model.graph.node.insert(topAddNodeId, newTopPreReshapeNode)
                topAddNode.input[list(topAddNode.input).index(topAddInput)] = topAddNodeNewInput
            else:
                topAddInArr = get_tensor_from_initializer(onnx_model, topAddInput)
                newTopAddInArr = np.reshape(topAddInArr, tuple(newReshapeShape))
                if transposePerm is not None:
                    newTopAddInArr = np.transpose(newTopAddInArr, tuple(transposePerm))
                newAddInputTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, topAddInput+"_new"),
                                                            data_type=NPDTYPE_2_ONNXDTYPE[newTopAddInArr.dtype],
                                                            dims=newTopAddInArr.shape,
                                                            vals=newTopAddInArr.flatten().tolist())
                onnx_model.graph.initializer.append(newAddInputTensor)
                topAddNode.input[list(topAddNode.input).index(topAddInput)] = newAddInputTensor.name
        newTopAfterShapeTensor = onnx.helper.make_tensor(name=topAddNode.output[0]+"_shape",
                                                    data_type=TensorProto.INT64,
                                                    dims=[3],
                                                    vals=topAddOutShape)
        onnx_model.graph.initializer.append(newTopAfterShapeTensor)
        newTopAfterReshapeNode = onnx.helper.make_node(name=topAddNode.name+"_afterReshape",
                                                       op_type="Reshape",
                                                       inputs=[topAddNode.output[0], newTopAfterShapeTensor.name],
                                                       outputs=[topAddNode.output[0]+"_afterReshapeOut"])
        topAddNodeId = get_node_id(onnx_model, topAddNode)
        if transposePerm is not None:
            newTopAfterTransposeNode = onnx.helper.make_node(name=topAddNode.name+"_afterTranspose",
                                                             op_type="Transpose",
                                                             inputs=topAddNode.output,
                                                             outputs=[topAddNode.output[0]+"_afterTransposeOut"],
                                                             perm=transposePerm)
            newTopAfterReshapeNode.input[0] = newTopAfterTransposeNode.output[0]
            onnx_model.graph.node.insert(topAddNodeId+1, newTopAfterReshapeNode)
            onnx_model.graph.node.insert(topAddNodeId+1, newTopAfterTransposeNode)
        else:
            onnx_model.graph.node.insert(topAddNodeId+1, newTopAfterReshapeNode)
        for topAddOutNode in topAddOutNodesList:
            if topAddOutNode.name == bottomAddNode.name:
                continue
            for out_id, topAddOutNodeInput in enumerate(topAddOutNode.input):
                topAddOutNode.input[out_id] = newTopAfterReshapeNode.output[0] \
                    if topAddOutNodeInput == topAddNode.output[0] else topAddOutNodeInput
        bottomAddNode.input[list(bottomAddNode.input).index(reshapeNode.output[0])] = reshapeNode.input[0] \
            if reshapeInNode.op_type != "Transpose" else reshapeInNode.input[0]
        node_index = get_node_id(onnx_model, bottomAddNode)
        newBottomAddAfterShapeTensor = onnx.helper.make_tensor(name=bottomAddNode.output[0]+"_shape",
                                                               data_type=TensorProto.INT64,
                                                               dims=[len(bottomAddOutShape)],
                                                               vals=bottomAddOutShape)
        newBottomAddAfterReshapeNode = onnx.helper.make_node(name=bottomAddNode.name+"_afterReshape",
                                                             op_type="Reshape",
                                                             inputs=[bottomAddNode.output[0], newBottomAddAfterShapeTensor.name],
                                                             outputs=[bottomAddNode.output[0]+"_afterReshapeOut"])
        blkOutReshapeValueInfo = onnx.helper.make_tensor_value_info(name=newBottomAddAfterReshapeNode.output[0],
                                                                    elem_type=1,
                                                                    shape=bottomAddOutShape)
        onnx_model.graph.initializer.append(newBottomAddAfterShapeTensor)
        onnx_model.graph.value_info.append(blkOutReshapeValueInfo)
        if transposePerm is not None:
            newBottomAddAfterTransposeNode = onnx.helper.make_node(name=bottomAddNode.name+"_afterTranspose",
                                                            op_type="Transpose",
                                                            inputs=bottomAddNode.output,
                                                            outputs=[bottomAddNode.output[0]+"_afterTransposeOut"],
                                                            perm=transposePerm)
            newBottomAddAfterReshapeNode.input[0] = newBottomAddAfterTransposeNode.output[0]
            blkOutTransposeValueInfo = onnx.helper.make_tensor_value_info(name=newBottomAddAfterTransposeNode.output[0],
                                                                    elem_type=1,
                                                                    shape=reshapeInShape)
            onnx_model.graph.value_info.append(blkOutTransposeValueInfo)
            onnx_model.graph.node.insert(node_index+1, newBottomAddAfterReshapeNode)
            onnx_model.graph.node.insert(node_index+1, newBottomAddAfterTransposeNode)
            onnx_model.graph.node.remove(reshapeInNode)
        else:
            onnx_model.graph.node.insert(node_index+1, newBottomAddAfterReshapeNode)
        onnx_model.graph.node.remove(reshapeNode)
        for bottomAddOutNode in bottomAddOutNodesList:
            for out_id, bottomAddOutNodeInput in enumerate(bottomAddOutNode.input):
                bottomAddOutNode.input[out_id] = newBottomAddAfterReshapeNode.output[0] \
                    if bottomAddOutNodeInput == bottomAddNode.output[0] else bottomAddOutNodeInput
        onnx_model = delete_value_info_by_name(onnx_model, topAddNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, bottomAddNode.output[0])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        bottomReshapeOutNodesList = get_node_by_input(onnx_model, newBottomAddAfterReshapeNode.output)
        for bottomReshapeOutNode in bottomReshapeOutNodesList:
            if bottomReshapeOutNode.op_type == "Add":
                onnx_model = check_Continue3dimResidual(onnx_model, bottomReshapeOutNode, newBottomAddAfterReshapeNode)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_3dimLayerNormalTo4dim(onnx_model, node, node_index):
    def find_tranposeInput_axes_for_reduce(topRMAxes: int, inShape: list, trOutShape: list, trPerm: list):
        topAxes = None
        if topRMAxes in [-1, 2]:
            if inShape[topRMAxes] == trOutShape[-1]:
                topAxes = trPerm.index(3)
            elif inShape[topRMAxes] == trOutShape[2] and trOutShape[-1] == 1:
                topAxes = trPerm.index(2)
        elif topRMAxes in [-2, 1]:
            if inShape[topRMAxes] == trOutShape[1]:
                topAxes = trPerm.index(1)
            elif inShape[topRMAxes] == trOutShape[2] and trOutShape[1] == 1:
                topAxes = trPerm.index(2)
        return topAxes
    
    def create_newarray_from_permute(stSrcArr, blkInShape: list, topOutShape: list, tPerm: list): 
        if len(stSrcArr.shape) == 1 or stSrcArr.shape[1] == 1:
            newArr = np.transpose(stSrcArr.reshape(1, 1, blkInShape[-1], 1), tuple(tPerm)) \
                if topOutShape[-1] == 1 else np.transpose(stSrcArr.reshape(1, 1, 1, blkInShape[-1]), tuple(tPerm))
        elif stSrcArr.shape == tuple(blkInShape):
            newArr = np.reshape(stSrcArr, tuple(topOutShape)).transpose(tPerm)
        else:
            newArr = np.transpose(stSrcArr.reshape(1, blkInShape[1], 1, 1), tuple(tPerm)) \
                if topOutShape[1] == 1 else np.transpose(stSrcArr.reshape(1, 1, blkInShape[1], 1), tuple(tPerm))
        return newArr
    
    lnNodesDicts = get_layernormal_node_dict(onnx_model, node)
    if lnNodesDicts is None:
        return onnx_model, False
    lnInputShape = get_shape_by_name(onnx_model, lnNodesDicts['input'].output[0])
    lnOutputShape = get_shape_by_name(onnx_model, lnNodesDicts['output'].output[0])
    if len(lnInputShape) != 3 or lnInputShape != lnOutputShape or lnInputShape[0] != 1:
        return onnx_model, False
    if lnNodesDicts['input'].op_type == "Reshape":
        topReshapeNode = lnNodesDicts['input']
        topTransposeNode = get_node_by_output(onnx_model, topReshapeNode.input[0])
        if topTransposeNode.op_type != 'Transpose':
            return onnx_model, False
        topTransposeOutShape = get_shape_by_name(onnx_model, topTransposeNode.output[0])
        if len(topTransposeOutShape) != 4:           
            return onnx_model, False
        topReduceMeanAttr = attribute_to_dict(lnNodesDicts['topReduceMean'].attribute)
        topReduceMeanAxes = topReduceMeanAttr.get("axes", None)
        if isinstance(topReduceMeanAxes, list):
            topReduceMeanAxes = topReduceMeanAxes[0]
        rightReduceMeanAttr = attribute_to_dict(lnNodesDicts['rightReduceMean'].attribute)
        rightReduceMeanAxes = rightReduceMeanAttr.get("axes", None)
        if isinstance(rightReduceMeanAxes, list):
            rightReduceMeanAxes = rightReduceMeanAxes[0]
        topTransposePerm = attribute_to_dict(topTransposeNode.attribute).get('perm', [3, 2, 1, 0])
        topRMTrInAxes = find_tranposeInput_axes_for_reduce(topReduceMeanAxes, lnInputShape, topTransposeOutShape, topTransposePerm)
        rightRMTrInAxes = find_tranposeInput_axes_for_reduce(rightReduceMeanAxes, lnInputShape, topTransposeOutShape, topTransposePerm)
        if topRMTrInAxes is None or rightRMTrInAxes is None:    
            return onnx_model, False
        topReduceMeanAxesAttr = onnx.helper.make_attribute('axes', [topRMTrInAxes])
        topReduceMeanKeepdims = topReduceMeanAttr.get('keepdims', 1)
        if isinstance(topReduceMeanKeepdims, list):
            topReduceMeanKeepdims = topReduceMeanKeepdims[0]
        topReduceMeanKeepdimAttr = onnx.helper.make_attribute('keepdims', topReduceMeanKeepdims)
        del lnNodesDicts['topReduceMean'].attribute[:]
        lnNodesDicts['topReduceMean'].attribute.append(topReduceMeanAxesAttr)
        lnNodesDicts['topReduceMean'].attribute.append(topReduceMeanKeepdimAttr)
        stLeftMulInput = lnNodesDicts['leftMul'].input[0] if find_init_by_name(onnx_model, lnNodesDicts['leftMul'].input[0]) \
            else lnNodesDicts['leftMul'].input[1]
        stLeftMulArr = get_tensor_from_initializer(onnx_model, stLeftMulInput)
        if stLeftMulArr.size != 1:  
            newLeftMulArr = create_newarray_from_permute(stLeftMulArr, lnInputShape, topTransposeOutShape, topTransposePerm)
            newLeftMulTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stLeftMulInput+"_new"),
                                                    data_type=NPDTYPE_2_ONNXDTYPE[newLeftMulArr.dtype],
                                                    dims=newLeftMulArr.shape,
                                                    vals=newLeftMulArr.flatten().tolist())
            onnx_model.graph.initializer.append(newLeftMulTensor)
            lnNodesDicts['leftMul'].input[list(lnNodesDicts['leftMul'].input).index(stLeftMulInput)] = newLeftMulTensor.name
        rightReduceMeanAxesAttr = onnx.helper.make_attribute('axes', [rightRMTrInAxes])
        rightReduceMeanKeepdims = rightReduceMeanAttr.get('keepdims', 1)
        if isinstance(rightReduceMeanKeepdims, list):
            rightReduceMeanKeepdims = rightReduceMeanKeepdims[0]
        rightReduceMeanKeepdimAttr = onnx.helper.make_attribute('keepdims', rightReduceMeanKeepdims)
        del lnNodesDicts['rightReduceMean'].attribute[:]
        lnNodesDicts['rightReduceMean'].attribute.append(rightReduceMeanAxesAttr)
        lnNodesDicts['rightReduceMean'].attribute.append(rightReduceMeanKeepdimAttr)
        stRightMulInput = lnNodesDicts['rightSecMul'].input[1] \
            if find_init_by_name(onnx_model, lnNodesDicts['rightSecMul'].input[1]) else lnNodesDicts['rightSecMul'].input[0]
        stRightMulArr = get_tensor_from_initializer(onnx_model, stRightMulInput)
        if stRightMulArr.size != 1:
            newRightMulArr = create_newarray_from_permute(stRightMulArr, lnInputShape, topTransposeOutShape, topTransposePerm)
            newRightMulTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stRightMulInput+"_new"),
                                                    data_type=NPDTYPE_2_ONNXDTYPE[newRightMulArr.dtype],
                                                    dims=newRightMulArr.shape,
                                                    vals=newRightMulArr.flatten().tolist())
            onnx_model.graph.initializer.append(newRightMulTensor)
            lnNodesDicts['rightSecMul'].input[list(lnNodesDicts['rightSecMul'].input).index(stRightMulInput)] = newRightMulTensor.name
        stRightAddInput = lnNodesDicts['rightAdd'].input[1] \
            if find_init_by_name(onnx_model, lnNodesDicts['rightAdd'].input[1]) else lnNodesDicts['rightAdd'].input[0]
        stRightAddArr = get_tensor_from_initializer(onnx_model, stRightAddInput)
        if stRightAddArr.size != 1:
            newRightAddArr = create_newarray_from_permute(stRightAddArr, lnInputShape, topTransposeOutShape, topTransposePerm)
            newRightAddTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stRightAddInput+"_new"),
                                                        data_type=NPDTYPE_2_ONNXDTYPE[newRightAddArr.dtype],
                                                        dims=newRightAddArr.shape,
                                                        vals=newRightAddArr.flatten().tolist())
            onnx_model.graph.initializer.append(newRightAddTensor)
            lnNodesDicts['rightAdd'].input[list(lnNodesDicts['rightAdd'].input).index(stRightAddInput)] = newRightAddTensor.name
        stOutAddInput = lnNodesDicts['output'].input[1] \
            if find_init_by_name(onnx_model, lnNodesDicts['output'].input[1]) else lnNodesDicts['output'].input[0]
        stOutAddArr = get_tensor_from_initializer(onnx_model, stOutAddInput)
        if stOutAddArr.size != 1:
            newOutAddArr = create_newarray_from_permute(stOutAddArr, lnInputShape, topTransposeOutShape, topTransposePerm)
            newOutAddTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stOutAddInput+"_new"),
                                                      data_type=NPDTYPE_2_ONNXDTYPE[newOutAddArr.dtype],
                                                      dims=newOutAddArr.shape,
                                                      vals=newOutAddArr.flatten().tolist())
            onnx_model.graph.initializer.append(newOutAddTensor)
            lnNodesDicts['output'].input[list(lnNodesDicts['output'].input).index(stOutAddInput)] = newOutAddTensor.name
        srcLNOutput = node.output[0]
        node.output[0] = node.output[0]+'_new'
        newTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, node.name+"_newTranspose"),
                                                 op_type="Transpose",
                                                 inputs=node.output,
                                                 outputs=[get_unique_node_tensor_name(onnx_model, srcLNOutput+"_newTransposeOut")],
                                                 perm=topTransposePerm)
        newReshapeShape = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, node.name+"_newShape"),
                                                data_type=TensorProto.INT64,
                                                dims=[3],
                                                vals=lnOutputShape)
        onnx_model.graph.initializer.append(newReshapeShape)
        newReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, node.name+"_newReshape"),
                                               op_type="Reshape",
                                               inputs=[newTransposeNode.output[0], newReshapeShape.name],
                                               outputs=[srcLNOutput])
        # newReshapeValue = onnx.helper.make_tensor_value_info(newReshapeNode.output[0], 1, lnOutputShape)
        # onnx_model.graph.value_info.append(newReshapeValue)
        topTransposeInShape = get_shape_by_name(onnx_model, topTransposeNode.input[0])
        newTransposeValue = onnx.helper.make_tensor_value_info(newTransposeNode.output[0], 1, topTransposeInShape)
        onnx_model.graph.value_info.append(newTransposeValue)
        lnNodesDicts['topReduceMean'].input[0] = topTransposeNode.input[0]
        lnNodesDicts['sub'].input[0] = topTransposeNode.input[0]
        for key in lnNodesDicts.keys():
            onnx_model = delete_value_info_by_name(onnx_model, lnNodesDicts[key].output[0])
        lnOutNodesList = get_node_by_input(onnx_model, node.output)
        for lnOutNode in lnOutNodesList:
            for inId, lnOutNodeInput in enumerate(lnOutNode.input):
                lnOutNode.input[inId] = newReshapeNode.output[0] if lnOutNodeInput == node.output[0] else lnOutNodeInput
        onnx_model = insert_node_by_list(onnx_model, [newReshapeNode, newTransposeNode], node_index+1)
        if not get_node_by_input(onnx_model, topReshapeNode.output):
            onnx_model.graph.node.remove(topReshapeNode)
            if not get_node_by_input(onnx_model, topTransposeNode.output):
                onnx_model.graph.node.remove(topTransposeNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_3dimFeedForwardTo4dim(onnx_model, node, node_index):
    searchOpsList = ['MatMul', 'Add']
    if check_node_serial_group(onnx_model, node, searchOpsList):
        matMulNode, addNode = get_node_serial_group(onnx_model, node, searchOpsList)
        addNextNodesList = get_node_by_input(onnx_model, addNode.output)
        reluNodesList = [addNextNode for addNextNode in addNextNodesList if addNextNode.op_type == 'Relu']
        matMulOutShape = get_shape_by_name(onnx_model, matMulNode.output[0])
        addOutShape = get_shape_by_name(onnx_model, addNode.output[0])
        if len(matMulOutShape) != 3 or len(addOutShape) != 3 or addOutShape[0] != 1:
            return onnx_model, False
        staticAddInput, dynamicAddInput = [addNode.input[0], addNode.input[1]] \
            if find_init_by_name(onnx_model, addNode.input[0]) else [addNode.input[1], addNode.input[0]]
        if not find_init_by_name(onnx_model, staticAddInput):
            return onnx_model, False
        insertNodesList = []
        matMulInLst = find_init_by_name(onnx_model, matMulNode.input[0])
        matMulInRst = find_init_by_name(onnx_model, matMulNode.input[1])
        if not matMulInLst and not matMulInRst:
            return onnx_model, False 
        stMatMulIn, dyMatMulIn = [matMulNode.input[1], matMulNode.input[0]] \
            if matMulInRst else [matMulNode.input[0], matMulNode.input[1]]    
        dyMatMulInShape = get_shape_by_name(onnx_model, dyMatMulIn)
        newShape = [dyMatMulInShape[0], dyMatMulInShape[1], 1, dyMatMulInShape[2]]
        newShapeTensor = onnx.helper.make_tensor(name=dyMatMulIn+'_newShape',
                                                    data_type=TensorProto.INT64,
                                                    dims=[4],
                                                    vals=newShape)
        onnx_model.graph.initializer.append(newShapeTensor)
        newReshapeNode = onnx.helper.make_node(name=dyMatMulIn+'_reshape',
                                                    op_type='Reshape',
                                                    inputs=[dyMatMulIn, newShapeTensor.name],
                                                    outputs=[dyMatMulIn+'_reshapeOut'])
        insertNodesList.append(newReshapeNode)
        dyMatMulMatix = get_tensor_from_initializer(onnx_model, stMatMulIn)
        newConvIn = newReshapeNode.output[0]
        if matMulInRst:
            newTransposeNode = onnx.helper.make_node(name=matMulNode.input[0]+'_transpose',
                                                        op_type='Transpose',
                                                        inputs=newReshapeNode.output,
                                                        outputs=[matMulNode.input[0]+'_transposeOut'],
                                                        perm=[0, 3, 2, 1])
            insertNodesList.append(newTransposeNode)
            newConvIn = newTransposeNode.output[0]
            dyMatMulMatix = np.transpose(dyMatMulMatix, (1, 0))[:, :, np.newaxis, np.newaxis]
        newCWTensor = onnx.helper.make_tensor(name=matMulNode.input[1]+'_weight',
                                                data_type=NPDTYPE_2_ONNXDTYPE[dyMatMulMatix.dtype],
                                                dims=dyMatMulMatix.shape,
                                                vals=dyMatMulMatix.flatten().tolist())
        onnx_model.graph.initializer.append(newCWTensor)
        newConvAttr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newConv = onnx.helper.make_node(name=matMulNode.name+'_Conv',
                                            op_type='Conv',
                                            inputs=[newConvIn, newCWTensor.name],
                                            outputs=[matMulNode.output[0]+'_convOut'],
                                            **newConvAttr)
        insertNodesList.append(newConv)
        newAddNode = onnx.helper.make_node(name=addNode.name+'_new',
                                            op_type='Add',
                                            inputs=[newConv.output[0], staticAddInput],
                                            outputs=[addNode.output[0]+'_newOut'])
        addInMatixArr = get_tensor_from_initializer(onnx_model, staticAddInput)
        if addInMatixArr.size != 1:
            newAddMatixArr = addInMatixArr[np.newaxis, np.newaxis, np.newaxis, :] \
                if len(addInMatixArr.shape) == 1 else addInMatixArr[:, :, np.newaxis, :]
            if matMulInRst:
                newAddMatixArr = newAddMatixArr.transpose(0, 3, 2, 1)
            newAddMatixTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, staticAddInput+'_new'),
                                                        data_type=NPDTYPE_2_ONNXDTYPE[newAddMatixArr.dtype],
                                                        dims=newAddMatixArr.shape,
                                                        vals=newAddMatixArr.flatten().tolist())
            onnx_model.graph.initializer.append(newAddMatixTensor)
            newAddNode.input[1] = newAddMatixTensor.name
        insertNodesList.append(newAddNode)
        newAddOutShape = [addOutShape[0], addOutShape[2], 1, addOutShape[1]] \
            if matMulInRst else [addOutShape[0], addOutShape[1], 1, addOutShape[2]]
        newAddValue = onnx.helper.make_tensor_value_info(newAddNode.output[0], 1, newAddOutShape)
        onnx_model.graph.value_info.append(newAddValue)  
        for relu_id, reluNode in enumerate(reluNodesList):
            reluNode.input[0] = newAddNode.output[0]
            reluOutNodesList = get_node_by_input(onnx_model, reluNode.output)
            reluNodeId = get_node_id(onnx_model, reluNode)
            afterReshapeShape = onnx.helper.make_tensor(name=reluNode.output[0]+"_newShape_%d"%relu_id,
                                                        data_type=TensorProto.INT64,
                                                        dims=[3],
                                                        vals=addOutShape)
            onnx_model.graph.initializer.append(afterReshapeShape)
            afterReshapeNode = onnx.helper.make_node(name=reluNode.name+"_afterReshape_%d"%relu_id,
                                                     op_type="Reshape",
                                                     inputs=[reluNode.output[0], afterReshapeShape.name],
                                                     outputs=[reluNode.output[0]+"_afterReshapeOut_%d"%relu_id])
            if matMulInRst:
                afterTransposeNode = onnx.helper.make_node(name=reluNode.name+"_afterTranspose_%d"%relu_id,
                                                           op_type="Transpose",
                                                           inputs=reluNode.output,
                                                           outputs=[reluNode.output[0]+"_afterTransposeOut_%d"%relu_id],
                                                           perm=[0, 3, 2, 1])
                afterReshapeNode.input[0] = afterTransposeNode.output[0]
                onnx_model.graph.node.insert(reluNodeId+1, afterReshapeNode)
                onnx_model.graph.node.insert(reluNodeId+1, afterTransposeNode)
                afterTransposeValue = onnx.helper.make_tensor_value_info(afterTransposeNode.output[0], 1, 
                                                        [addOutShape[0], addOutShape[2], 1, addOutShape[1]])
                onnx_model.graph.value_info.append(afterTransposeValue)
            else:
                onnx_model.graph.node.insert(reluNodeId+1, afterReshapeNode)
            for reluOutNode in reluOutNodesList:
                for inId, reluOutNodeIn in enumerate(reluOutNode.input):
                    reluOutNode.input[inId] = afterReshapeNode.output[0] \
                        if reluOutNodeIn == reluNode.output[0] else reluOutNodeIn
            afterReshapeValue = onnx.helper.make_tensor_value_info(afterReshapeNode.output[0], 1, addOutShape)
            onnx_model.graph.value_info.append(afterReshapeValue)
        if not reluNodesList:
            addNodeId = get_node_id(onnx_model, addNode)
            afterReshapeShape = onnx.helper.make_tensor(name=addNode.output[0]+"_newShape",
                                            data_type=TensorProto.INT64,
                                            dims=[3],
                                            vals=addOutShape)
            onnx_model.graph.initializer.append(afterReshapeShape)
            afterReshapeNode = onnx.helper.make_node(name=addNode.name+"_afterReshape",
                                                     op_type="Reshape",
                                                     inputs=[newAddNode.output[0], afterReshapeShape.name],
                                                     outputs=addNode.output)
            if matMulInRst:
                afterTransposeNode = onnx.helper.make_node(name=addNode.name+"_afterTranspose",
                                                           op_type="Transpose",
                                                           inputs=newAddNode.output,
                                                           outputs=[addNode.output[0]+"_afterTransposeOut"],
                                                           perm=[0, 3, 2, 1])
                afterReshapeNode.input[0] = afterTransposeNode.output[0]
                onnx_model.graph.node.insert(addNodeId+1, afterReshapeNode)
                onnx_model.graph.node.insert(addNodeId+1, afterTransposeNode)
                afterTransposeValue = onnx.helper.make_tensor_value_info(afterTransposeNode.output[0], 1, 
                                                        [addOutShape[0], addOutShape[2], 1, addOutShape[1]])
            else:
                onnx_model.graph.node.insert(addNodeId+1, afterReshapeNode)
        onnx_model = delete_value_info_by_name(onnx_model, addNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, dynamicAddInput)
        onnx_model = delete_value_info_by_name(onnx_model, dyMatMulIn)               
        insertNodesList.reverse()
        onnx_model = insert_node_by_list(onnx_model, insertNodesList, node_index)
        if reluNodesList:
            addOutNodesList = get_node_by_input(onnx_model, addNode.output)
            if not addOutNodesList:
                onnx_model.graph.node.remove(addNode)
                matMulOutNodesList = get_node_by_input(onnx_model, matMulNode.output)
                if not matMulOutNodesList:
                    onnx_model.graph.node.remove(matMulNode)
        else:
            onnx_model.graph.node.remove(addNode)
            matMulOutNodesList = get_node_by_input(onnx_model, matMulNode.output)
            if not matMulOutNodesList:
                onnx_model.graph.node.remove(matMulNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_transposeReshape3dimAddTo4dimAdd(onnx_model, node, node_index):
    if node.op_type != 'Add':
        return onnx_model, False
    if find_init_by_name(onnx_model, node.input[0]) or find_init_by_name(onnx_model, node.input[1]):
        return onnx_model, False
    leftReshapeNode = get_node_by_output(onnx_model, node.input[0])
    rightReshapeNode = get_node_by_output(onnx_model, node.input[1])
    if leftReshapeNode.op_type != 'Reshape' or rightReshapeNode.op_type != 'Reshape':
        return onnx_model, False
    leftTransposeNode = get_node_by_output(onnx_model, leftReshapeNode.input[0])
    rightTransposeNode = get_node_by_output(onnx_model, rightReshapeNode.input[0])
    if leftTransposeNode.op_type != 'Transpose' or rightTransposeNode.op_type != 'Transpose':
        return onnx_model, False
    leftReshapeOutShape = get_shape_by_name(onnx_model, leftReshapeNode.output[0])
    rightReshapeOutShape = get_shape_by_name(onnx_model, rightReshapeNode.output[0])
    leftTransposeOutShape = get_shape_by_name(onnx_model, leftTransposeNode.output[0])
    rightTransposeOutShape = get_shape_by_name(onnx_model, rightTransposeNode.output[0])
    addOutShape = get_shape_by_name(onnx_model, node.output[0])
    if len(leftReshapeOutShape) != 3 or len(rightReshapeOutShape) != 3 \
        or len(leftTransposeOutShape) != 4 or len(rightTransposeOutShape) != 4:
        return onnx_model, False
    leftTransposePerm = attribute_to_dict(leftTransposeNode.attribute).get('perm', [3, 2, 1, 0])
    rightTransposePerm = attribute_to_dict(rightTransposeNode.attribute).get('perm', [3, 2, 1, 0])
    if leftTransposePerm != rightTransposePerm:
        return onnx_model, False
    addOutNodesList = get_node_by_input(onnx_model, node.output)
    node.input[0] = leftTransposeNode.input[0]
    node.input[1] = rightTransposeNode.input[0]
    afterTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, node.name+"_afterTranspose"),
                                               op_type='Transpose',
                                               inputs=node.output,
                                               outputs=[node.output[0]+'_afterTransposeOut'],
                                               perm=leftTransposePerm)
    afterReshapeShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, node.output[0]+'_newShape'),
                                                      data_type=TensorProto.INT64,
                                                      dims=[3],
                                                      vals=addOutShape)
    onnx_model.graph.initializer.append(afterReshapeShapeTensor)
    afterReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, node.name+"_afterReshape"),
                                             op_type='Reshape',
                                             inputs=[afterTransposeNode.output[0], afterReshapeShapeTensor.name],
                                             outputs=[node.output[0]+'_afterReshapeOut'])
    onnx_model = insert_node_by_list(onnx_model, [afterReshapeNode, afterTransposeNode], node_index+1)
    afterReshapeValue = onnx.helper.make_tensor_value_info(afterReshapeNode.output[0], 1, addOutShape)
    onnx_model.graph.value_info.append(afterReshapeValue)
    leftAddRandomArr = np.array(np.random.random(tuple(leftTransposeOutShape)), dtype=np.float32)
    rightAddRandomArr =np.array(np.random.random(tuple(rightTransposeOutShape)), dtype=np.float32)
    addRandomOutArr = leftAddRandomArr + rightAddRandomArr
    afterTransposeValue = onnx.helper.make_tensor_value_info(afterTransposeNode.output[0], 1, addRandomOutArr.shape)
    onnx_model.graph.value_info.append(afterTransposeValue)
    for addOutNode in addOutNodesList:
        for inId, addOutNodeInput in enumerate(addOutNode.input):
            addOutNode.input[inId] = afterReshapeNode.output[0] \
                if addOutNodeInput == node.output[0] else addOutNodeInput
    leftReshapeOutsList = get_node_by_input(onnx_model, leftReshapeNode.output)
    if not leftReshapeOutsList:
        onnx_model = delete_value_info_by_name(onnx_model, leftReshapeNode.output[0])
        onnx_model.graph.node.remove(leftReshapeNode)
        leftTransposeOutsList = get_node_by_input(onnx_model, leftTransposeNode.output)
        if not leftTransposeOutsList:
            onnx_model = delete_value_info_by_name(onnx_model, leftTransposeNode.output[0])
            onnx_model.graph.node.remove(leftTransposeNode)
    rightReshapeOutsList = get_node_by_input(onnx_model, rightReshapeNode.output)
    if not rightReshapeOutsList:
        onnx_model = delete_value_info_by_name(onnx_model, rightReshapeNode.output[0])
        onnx_model.graph.node.remove(rightReshapeNode)
        rightTransposeOutsList = get_node_by_input(onnx_model, rightTransposeNode.output)
        if not rightTransposeOutsList:
            onnx_model = delete_value_info_by_name(onnx_model, rightTransposeNode.output[0])
            onnx_model.graph.node.remove(rightTransposeNode)
    onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionMaskMulTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Mul', 'Mul', 'Transpose']):
        serialNodesList = get_node_serial_group(onnx_model, node, ['Transpose', 'Mul', 'Mul', 'Transpose'])
        topTransposeNode, topMulNode, botMulNode, botTransposeNode = serialNodesList
        topOutShape = get_shape_by_name(onnx_model, topTransposeNode.input[0])
        botOutShape = get_shape_by_name(onnx_model, botTransposeNode.output[0])
        if len(topOutShape) != 4 or topOutShape != botOutShape:
            return onnx_model, False
        topTransposePerm = attribute_to_dict(topTransposeNode.attribute).get('perm', [3, 2, 1, 0])
        botTransposePerm = attribute_to_dict(botTransposeNode.attribute).get('perm', [3, 2, 1, 0])
        topStMulIn = topMulNode.input[1] if topTransposeNode.output[0] == topMulNode.input[0] else topMulNode.input[1]
        if not find_init_by_name(onnx_model, topStMulIn):
            return onnx_model, False
        botOtherMulIn = botMulNode.input[0] if topMulNode.output[0] == botMulNode.input[1] else botMulNode.input[1]
        if find_init_by_name(onnx_model, botOtherMulIn):
            return onnx_model, False
        newPerm = [topTransposePerm[perm_id] for perm_id in topTransposePerm]
        newPerm = topTransposePerm if newPerm == [0, 1, 2, 3] else newPerm
        if newPerm != botTransposePerm:
            return onnx_model, False
        botOtherMulInShape = get_shape_by_name(onnx_model, botOtherMulIn)
        newTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, botOtherMulIn+'_newTranspose'),
                                                 op_type='Transpose',
                                                 inputs=[botOtherMulIn],
                                                 outputs=[get_unique_node_tensor_name(onnx_model, botOtherMulIn+'_newTransposeOut')],
                                                 perm=newPerm)
        if len(botOtherMulInShape) == 1:
            newShapeVals = [1, 1, 1, botOtherMulInShape[0]]
            newShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, botOtherMulIn+'_newShape'),
                                                     data_type=TensorProto.INT64,
                                                     dims=[4],
                                                     vals=newShapeVals)
            onnx_model.graph.initializer.append(newShapeTensor)
            newReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, botOtherMulIn+'_newReshape'),
                                                   op_type='Reshape',
                                                   inputs=[botOtherMulIn, newShapeTensor],
                                                   outputs=[get_unique_node_tensor_name(onnx_model, botOtherMulIn+'_newReshapeOut')])
            newTransposeNode.input[0] = newReshapeNode.output[0]
            onnx_model.graph.node.insert(node_index, newTransposeNode)
            onnx_model.graph.node.insert(node_index, newReshapeNode)
            newReshapeValue = onnx.helper.make_tensor_value_info(newReshapeNode.output[0], 1, newShapeVals)
            onnx_model.graph.value_info.append(newReshapeValue)
            newTransposeOutShape = [newShapeVals[perm_id] for perm_id in newPerm]
        else:
            onnx_model.graph.node.insert(node_index, newTransposeNode)
            newTransposeOutShape = [botOtherMulInShape[perm_id] for perm_id in newPerm]
        newTransposeValue = onnx.helper.make_tensor_value_info(newTransposeNode.output[0], 1, newTransposeOutShape)
        onnx_model.graph.value_info.append(newTransposeValue)
        
        botMulNode.input[list(botMulNode.input).index(botOtherMulIn)] = newTransposeNode.output[0]
        topMulInArr = get_tensor_from_initializer(onnx_model, topStMulIn)
        if topMulInArr.size != 1:
            newTopMulInArr = topMulInArr.reshape(1, 1, 1, topMulInArr.shape[0]) if len(topMulInArr.shape) == 1 else topMulInArr
            newTopMulInArr = np.transpose(newTopMulInArr, tuple(newPerm))
            newTopMulInTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, topStMulIn+'_new'),
                                                        data_type=NPDTYPE_2_ONNXDTYPE[newTopMulInArr.dtype],
                                                        dims=newTopMulInArr.shape,
                                                        vals=newTopMulInArr.flatten().tolist())
            onnx_model.graph.initializer.append(newTopMulInTensor)
            topMulNode.input[list(topMulNode.input).index(topStMulIn)] = newTopMulInTensor.name
        botTpOutNodesList = get_node_by_input(onnx_model, botTransposeNode.output)
        for botTpOutNode in botTpOutNodesList:
            for inId, botTpOutNodeIn in enumerate(botTpOutNode.input):
                botTpOutNode.input[inId] = botMulNode.output[0] if botTpOutNodeIn == botTransposeNode.output[0] else botTpOutNodeIn
        topMulNode.input[list(topMulNode.input).index(topTransposeNode.output[0])] = topTransposeNode.input[0]
        onnx_model = delete_value_info_by_name(onnx_model, topTransposeNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, botTransposeNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, topMulNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, botMulNode.output[0])
        topTransposeInShape = get_shape_by_name(onnx_model, topTransposeNode.input[0])
        newBotMulOutValue = onnx.helper.make_tensor_value_info(botMulNode.output[0], 1, topTransposeInShape)
        onnx_model.graph.value_info.append(newBotMulOutValue)
        onnx_model = delete_nodes(onnx_model, [topTransposeNode, botTransposeNode])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertViT_attention(onnx_model, node, node_index):
    def convert_multi_batch(onnx_model, node, nodeid, serial_nodes):
        kConvNode = serial_nodes['k_serial'][0]
        qReshapeNode = serial_nodes['q_serial'][0]
        vReshapeNode = serial_nodes['v_serial'][0]
        kqMatMulNode = serial_nodes['kq_serial'][0]
        kqMulNode = serial_nodes['kq_serial'][1]
        kqAddNode = serial_nodes['kq_serial'][2]
        kqSoftmaxNode = serial_nodes['kq_serial'][3]
        kInShape = get_shape_by_name(onnx_model, kConvNode.input[0])
        qInShape = get_shape_by_name(onnx_model, qReshapeNode.input[0])
        vInShape = get_shape_by_name(onnx_model, vReshapeNode.input[0])
        newKInShape = [1, kInShape[0]*kInShape[1], kInShape[2], kInShape[3]]
        newQInShape = [1, qInShape[0]*qInShape[1], qInShape[2], qInShape[3]]
        newVInShape = [1, qInShape[0]*vInShape[1], vInShape[2], vInShape[3]]
        newKInShapeTensor = get_initial_by_value(onnx_model, np.array(newKInShape, dtype=np.int64))
        if newKInShapeTensor is None:
            newKInShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, kConvNode.input[0]+'_newShape'),
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(newKInShape)],
                                                    vals=newKInShape)
        newQInShapeTensor = newKInShapeTensor if newKInShape == newQInShape else None
        newQInShapeTensor = get_initial_by_value(onnx_model, np.array(newQInShape, dtype=np.int64)) if newQInShapeTensor is None else newQInShapeTensor
        if newQInShapeTensor is None:
            newQInShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, qReshapeNode.name+'_newShape'),
                                        data_type=TensorProto.INT64,
                                        dims=[len(newQInShape)],
                                        vals=newQInShape)
        newVInShapeTensor = newKInShapeTensor if newKInShape == newVInShape else None
        newVInShapeTensor = newQInShapeTensor if newVInShapeTensor is None and newQInShape == newVInShape else newVInShapeTensor
        newVInShapeTensor = get_initial_by_value(onnx_model, np.array(newVInShape, dtype=np.int64)) if newVInShapeTensor is None else newVInShapeTensor
        if newVInShapeTensor is None:
            newVInShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, vReshapeNode.name+'_newShape'),
                                        data_type=TensorProto.INT64,
                                        dims=[len(newVInShape)],
                                        vals=newVInShape)
        newQSliceShapeTensor = get_initial_by_value(onnx_model, np.array([1, qInShape[1], 1, qInShape[2]*qInShape[3]], dtype=np.int64))
        if newQSliceShapeTensor is None:
            newQSliceShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, qReshapeNode.input[1]+'_newQShape'),
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(newQInShape)],
                                                    vals=[1, qInShape[1], 1, qInShape[2]*qInShape[3]])
        newVSliceShapeTensor = get_initial_by_value(onnx_model, np.array([vInShape[1], vInShape[2]*vInShape[3], 1, 1], dtype=np.int64))
        if newVSliceShapeTensor is None:
            newVSliceShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, vReshapeNode.input[1]+'_newVShape'),
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(newVInShape)],
                                                    vals=[vInShape[1], vInShape[2]*vInShape[3], 1, 1])
        kqAddStaticInput = kqAddNode.input[1] if find_init_by_name(onnx_model, kqAddNode.input[1]) else kqAddNode.input[0]
        kqAddStArr = get_tensor_from_initializer(onnx_model, kqAddStaticInput)
        newKQAddStArr = copy.deepcopy(kqAddStArr)
        newKQAddStTensorList = []
        if kqAddStArr.size != 1:
            while True:
                if len(newKQAddStArr.shape) > 4:
                    return onnx_model, False
                elif len(newKQAddStArr.shape) == 4:
                    break
                newKQAddStArr = np.expand_dims(newKQAddStArr, axis=0)
            if newKQAddStArr.shape[1] not in [kConvInShape[0], 1]:
                return onnx_model, False
            for vid in range(newKQAddStArr.shape[1]):
                newKQAddStSubArr = newKQAddStArr[:, vid:(vid+1), :, :]
                if newKQAddStSubArr.shape[2] == kInShape[2] * kInShape[3]:
                    newKQAddStSubArr = np.reshape(newKQAddStSubArr, (1, kInShape[2], kConvInShape[3], newKQAddStSubArr.shape[-1]))
                elif newKQAddStSubArr.shape[2] != 1:
                    return onnx_model, False
                newKQAddStSubArr = np.transpose(newKQAddStSubArr, (0, 3, 1, 2))
                newKQAddStTensor = get_initial_by_value(onnx_model, newKQAddStSubArr)
                if newKQAddStTensor is None:
                    newKQAddStTensor = onnx.helper.make_tensor(name=kqAddStaticInput+('_new' if newKQAddStArr.shape[1] == 1 else '_new%d'%vid),
                                                           data_type=NPDTYPE_2_ONNXDTYPE[newKQAddStSubArr.dtype],
                                                           dims=newKQAddStSubArr.shape,
                                                           vals=newKQAddStSubArr.flatten().tolist())
                newKQAddStTensorList.append(newKQAddStTensor)
        
        kqMulStaticInput = kqMulNode.input[1] if find_init_by_name(onnx_model, kqMulNode.input[1]) else kqMulNode.input[0]
        kqMulStArr = get_tensor_from_initializer(onnx_model, kqMulStaticInput)
        newKQMulStArr = copy.deepcopy(kqMulStArr)
        newKQMulStTensorList = []
        if kqMulStArr.size != 1:
            while True:
                if len(newKQMulStArr.shape) > 4:
                    return onnx_model, False
                elif len(newKQMulStArr.shape) == 4:
                    break
                newKQMulStArr = np.expand_dims(newKQMulStArr, axis=0)
            if newKQMulStArr.shape[1] not in [kConvInShape[0], 1]:
                return onnx_model, False
            for vid in range(newKQAddStArr.shape[1]):
                newKQMulStSubArr = newKQAddStArr[:, vid:(vid+1), :, :]
                if newKQMulStSubArr.shape[2] == kInShape[2] * kInShape[3]:
                    newKQMulStSubArr = np.reshape(newKQMulStSubArr, (1, kInShape[2], kInShape[3], newKQMulStSubArr.shape[-1]))
                elif newKQMulStSubArr.shape[2] != 1:
                    return onnx_model, False
                newKQMulStSubArr = np.tanspose(newKQMulStSubArr, (0, 3, 1, 2))
                newKQMulStTensor = get_initial_by_value(onnx_model, newKQMulStSubArr)
                if newKQMulStTensor is None:
                    newKQMulStTensor = onnx.helper.make_tensor(name=kqMulStaticInput+('_new' if newKQMulStArr.shape[1] == 1 else '_new%d'%vid),
                                                           data_type=NPDTYPE_2_ONNXDTYPE[newKQMulStSubArr.dtype],
                                                           dims=newKQMulStSubArr.shape,
                                                           vals=newKQMulStSubArr.flatten().tolist())
                newKQMulStTensorList.append(newKQMulStTensor)
        kqvShapeTensorList = [newKInShapeTensor, newQInShapeTensor, newVInShapeTensor, newQSliceShapeTensor, newVSliceShapeTensor]
        for kqvShapeTensor in kqvShapeTensorList:
            if not find_init_by_name(onnx_model, kqvShapeTensor.name):
                onnx_model.graph.initializer.append(kqvShapeTensor)
        for newKQAddTensor in newKQAddStTensorList:
            if not find_init_by_name(onnx_model, newKQAddTensor.name):
                onnx_model.graph.initializer.append(newKQAddTensor)
        for newKQMulTensor in newKQMulStTensorList:
            if not find_init_by_name(onnx_model, newKQMulTensor.name):
                onnx_model.graph.initializer.append(newKQMulTensor)
                            
        newKRSNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, kConvNode.input[0]+'_reshape'),
                                        op_type='Reshape',
                                        inputs=[kConvNode.input[0], newKInShapeTensor.name],
                                        outputs=[kConvNode.input[0]+'_newShapeOut'])
        newQRSNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, qReshapeNode.input[0]+'_reshape'),
                                           op_type='Reshape',
                                           inputs=[qReshapeNode.input[0], newQInShapeTensor.name],
                                           outputs=[qReshapeNode.input[0]+'_newShapeOut'])
        newVRSNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, vReshapeNode.input[0]+'_reshape'),
                                           op_type='Reshape',
                                           inputs=[vReshapeNode.input[0], newVInShapeTensor.name],
                                           outputs=[vReshapeNode.input[0]+'_newShapeOut'])
        sliceStart = 0
        sliceStartTensor = get_initial_by_value(onnx_model, np.array(sliceStart, dtype=np.int64))
        if sliceStartTensor is None:
            sliceStartTensor = onnx.helper.make_tensor(name=node.name+'_kqv_slice_loc0',
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[sliceStart])
        sliceAxesTensor = get_initial_by_value(onnx_model, np.array(1, dtype=np.int64))
        if sliceAxesTensor is None:
            sliceAxesTensor = onnx.helper.make_tensor(name=node.name+'_kqv_slice_axes',
                                                   data_type=TensorProto.INT64,
                                                   dims=[1],
                                                   vals=[1])
        sliceStepTensor = get_initial_by_value(onnx_model, np.array(1, dtype=np.int64))
        if sliceStepTensor is None:
            sliceStepTensor = onnx.helper.make_tensor(name=node.name+'_kqv_slice_step',
                                                   data_type=TensorProto.INT64,
                                                   dims=[1],
                                                   vals=[1])
        for sliceInTensor in [sliceStartTensor, sliceAxesTensor, sliceStepTensor]:
            if not find_init_by_name(onnx_model, sliceInTensor.name):
                onnx_model.graph.initializer.append(sliceInTensor)
                
        kqSoftmaxAxis = attribute_to_dict(kqSoftmaxNode.attribute).get('axis', 1)   
        kqSoftmaxAxisW4D = kqSoftmaxAxis + 1 if kqSoftmaxAxis >= 0 else kqSoftmaxAxis + 4 
        newKQSoftmaxAxis = [0, 3, 1, 2].index(kqSoftmaxAxisW4D)
        
        sliceConvKNodes = []
        sliceReshapeTransposeQNodes = []
        sliceReshapeVNodes = []
        sliceNewKQNodes = []
        sliceNewKQVNodes = []
        newConcatInputs = []
        sliceKStart = sliceStart
        sliceQStart = sliceStart
        sliceVStart = sliceStart
        sliceKStartTensor = sliceStartTensor
        sliceQStartTensor = sliceStartTensor
        sliceVStartTensor = sliceStartTensor
        for sliceNum in range(kInShape[0]):
            sliceKEnd = sliceKStart + kInShape[1]
            sliceQEnd = sliceQStart + qInShape[1]
            sliceVEnd = sliceVStart + vInShape[1]
            sliceKEndTensor = get_initial_by_value(onnx_model, np.array(sliceKEnd, dtype=np.int64))
            if sliceKEndTensor is None:
                sliceKEndTensor = onnx.helper.make_tensor(name=kConvNode.input[0]+'_slice_loc%d'%(sliceNum+1),
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[sliceKEnd])
                onnx_model.graph.initializer.append(sliceKEndTensor)
            sliceQEndTensor = sliceKEndTensor if sliceQEnd == sliceKEnd else get_initial_by_value(onnx_model, np.array(sliceQEnd, dtype=np.int64))
            if sliceQEndTensor is None:
                sliceKEndTensor = onnx.helper.make_tensor(name=qReshapeNode.input[0]+'_slice_loc%d'%(sliceNum+1),
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[sliceQEnd])
                onnx_model.graph.initializer.append(sliceQEndTensor)
            sliceVEndTensor = sliceKEndTensor if sliceVEnd == sliceKEnd else None
            sliceVEndTensor = sliceQEndTensor if sliceVEnd == sliceQEnd and sliceVEndTensor is None \
                else get_initial_by_value(onnx_model, np.array(sliceVEnd, dtype=np.int64))
            if sliceVEndTensor is None:
                sliceVEndTensor = onnx.helper.make_tensor(name=vReshapeNode.input[0]+'_slice_loc%d'%(sliceNum+1),
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[sliceVEnd])
                onnx_model.graph.initializer.append(sliceVEndTensor)
            sliceKInputs = [newKRSNode.output[0], sliceKStartTensor.name, sliceKEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
            sliceKNode = onnx.helper.make_node(name=kConvNode.input[0]+'_slice%d'%sliceNum,
                                               op_type='Slice',
                                               inputs=sliceKInputs,
                                               outputs=[kConvNode.input[0]+'_slice%d'%sliceNum+'_out'])
            sliceQInputs = [newQRSNode.output[0], sliceQStartTensor.name, sliceQEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
            sliceQNode = onnx.helper.make_node(name=qReshapeNode.input[0]+'_slice%d'%sliceNum,
                                               op_type='Slice',
                                               inputs=sliceQInputs,
                                               outputs=[qReshapeNode.input[0]+'_slice%d'%sliceNum+'_out'])
            sliceVInputs = [newVRSNode.output[0], sliceVStartTensor.name, sliceVEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
            sliceVNode = onnx.helper.make_node(name=vReshapeNode.input[0]+'_slice%d'%sliceNum,
                                               op_type='Slice',
                                               inputs=sliceVInputs,
                                               outputs=[vReshapeNode.input[0]+'_slice%d'%sliceNum+'_out'])
            sliceConvKNodes.append(sliceKNode)
            sliceReshapeTransposeQNodes.append(sliceQNode)
            sliceReshapeVNodes.append(sliceVNode)

            newKConvNode = copy.deepcopy(kConvNode)
            newKConvNode.input[0] = sliceKNode.output[0]
            newKConvNode.output[0] = kConvNode.output[0]+'_%d'%sliceNum
            newKConvNode.name = kConvNode.name+'_%d'%sliceNum
            sliceConvKNodes.append(newKConvNode)
            
            newQReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, qReshapeNode.name+'_slice%d'%sliceNum),
                                                    op_type='Reshape',
                                                    inputs=[sliceQNode.output[0], newQSliceShapeTensor.name],
                                                    outputs=[qReshapeNode.output[0]+'_slice%d'%sliceNum])
            sliceReshapeTransposeQNodes.append(newQReshapeNode)
            newQTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, qReshapeNode.name+'_slice%d'%sliceNum+'_toTranspose'),
                                                      op_type='Transpose',
                                                      inputs=newQReshapeNode.output,
                                                      outputs=[qReshapeNode.output[0]+'_tpOut_slice%d'%sliceNum],
                                                      perm=[3, 1, 0, 2])
            sliceReshapeTransposeQNodes.append(newQTransposeNode)
            
            newVReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, vReshapeNode.name+'_slice%d'%sliceNum),
                                                    op_type='Reshape',
                                                    inputs=[sliceVNode.output[0], newVSliceShapeTensor.name],
                                                    outputs=[vReshapeNode.output[0]+'_slice%d'%sliceNum])
            sliceReshapeVNodes.append(newVReshapeNode)
            
            newKQConvAttr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            newKQConvNode = onnx.helper.make_node(name=kqMatMulNode.name+'_slice%d'%sliceNum,
                                                  op_type='Conv',
                                                  inputs=[newKConvNode.output[0], newQTransposeNode.output[0]],
                                                  outputs=[kqMatMulNode.output[0]+'_slice%d'%sliceNum],
                                                  **newKQConvAttr)
            sliceNewKQNodes.append(newKQConvNode)
            newKQMulInputs = [newKQConvNode.output[0]]
            if len(newKQMulStTensorList) == 0:
                newKQMulInputs.append(kqMulStaticInput)
            elif len(newKQMulStTensorList) == 1:
                newKQMulInputs.append(newKQMulStTensorList[0].name)
            else:
                newKQMulInputs.append(newKQMulStTensorList[sliceNum].name)
            newKQMulNode = onnx.helper.make_node(name=kqMulNode.name+'_slice%d'%sliceNum,
                                                op_type='Mul',
                                                inputs=newKQMulInputs,
                                                outputs=[kqMulNode.output[0]+'_slice%d'%sliceNum])
            sliceNewKQNodes.append(newKQMulNode)
            newKQAddInputs = [newKQMulNode.output[0]]
            if len(newKQAddStTensorList) == 0:
                newKQAddInputs.append(kqAddStaticInput)
            elif len(newKQAddStTensorList) == 1:
                newKQAddInputs.append(newKQAddStTensorList[0].name)
            else:
                newKQAddInputs.append(newKQAddStTensorList[sliceNum].name)
            newKQAddNode = onnx.helper.make_node(name=kqAddNode.name+'_slice%d'%sliceNum,
                                                 op_type='Add',
                                                 inputs=newKQAddInputs,
                                                 outputs=[kqAddNode.output[0]+'_slice%d'%sliceNum])
            sliceNewKQNodes.append(newKQAddNode)
            newKQSoftmaxNode = onnx.helper.make_node(name=kqSoftmaxNode.name+'_slice%d'%sliceNum,
                                                 op_type='Softmax',
                                                 inputs=newKQAddNode.output,
                                                 outputs=[kqSoftmaxNode.output[0]+'_slice%d'%sliceNum],
                                                 axis=newKQSoftmaxAxis)
            sliceNewKQNodes.append(newKQSoftmaxNode)
            
            newKQVConvAttr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            newKQVConvNode = onnx.helper.make_node(name=node.name+'_slice%d'%sliceNum,
                                                   op_type='Conv',
                                                   inputs=[newKQSoftmaxNode.output[0], newVReshapeNode.output[0]],
                                                   outputs=[node.output[0]+'_slice%d'%sliceNum],
                                                   **newKQVConvAttr)
            sliceNewKQVNodes.append(newKQVConvNode)
            newConcatInputs.append(newKQVConvNode.output[0])
            sliceKStartTensor = sliceKEndTensor
            sliceQStartTensor = sliceQEndTensor
            sliceVStartTensor = sliceVEndTensor
            sliceKStart = sliceKEnd
            sliceQStart = sliceQEnd
            sliceVStart = sliceVEnd
        
        #newConcatInputs.reverse()
        newConcatNode = onnx.helper.make_node(name=node.name+'_newConcat',
                                              op_type='Concat',
                                              inputs=newConcatInputs,
                                              outputs=[node.output[0]+'_newConcatOut'],
                                              axis=1)
        
        kqvMatMulOutShape = get_shape_by_name(onnx_model, node.output[0])
        tfOutNodes = get_node_by_input(onnx_model, node.output)
        if not (len(tfOutNodes) == 1 and tfOutNodes[0].op_type == 'Reshape'):
            outRSShapeTensor = get_initial_by_value(onnx_model, np.array(kqvMatMulOutShape, dtype=np.int64))
            if outRSShapeTensor is None:
                outRSShapeTensor = onnx.helper.make_tensor(name=node.output[0]+'_cvtShape',
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(kqvMatMulOutShape)],
                                                       vals=kqvMatMulOutShape)
                onnx_model.graph.initializer.append(outRSShapeTensor)
            outRSNode = onnx.helper.make_node(name=node.name+'_reshape',
                                              op_type='Reshape',
                                              inputs=[newConcatNode.output[0], outRSShapeTensor.name],
                                              outputs=node.output)
            onnx_model.graph.node.insert(nodeid, outRSNode)
        else:
            newConcatNode.output[0] = node.output[0]
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            newConcatOutShape = [1, vInShape[0]*vInShape[1], kInShape[2], kInShape[3]]
            tfNextOutShape = get_shape_by_name(onnx_model, tfOutNodes[0].output[0])
            if newConcatOutShape == tfNextOutShape:
                newConcatNode.output[0] = tfOutNodes[0].output[0]
                onnx_model.graph.node.remove(tfOutNodes[0])
            else:
                tfOutValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newConcatOutShape)
                onnx_model.graph.value_info.append(tfOutValueInfo)
        onnx_model.graph.node.insert(nodeid, newConcatNode)
        sliceNewKQVNodes.reverse()
        onnx_model = insert_node_by_list(onnx_model, sliceNewKQVNodes, nodeid)
        sliceKQTile = int(len(sliceNewKQNodes) / kInShape[0])
        sliceKTile = int(len(sliceConvKNodes) / kInShape[0])
        sliceQTile = int(len(sliceReshapeTransposeQNodes) / kInShape[0])
        sliceVTile = int(len(sliceReshapeVNodes) / kInShape[0])
        for sliceNum in range(kInShape[0]):
            tileKQNodes = sliceNewKQNodes[(sliceNum*sliceKQTile):((sliceNum+1)*sliceKQTile)]
            tileKQNodes.reverse()
            onnx_model = insert_node_by_list(onnx_model, tileKQNodes, nodeid)
            tileKNodes = sliceConvKNodes[(sliceNum*sliceKTile):((sliceNum+1)*sliceKTile)]
            tileKNodes.reverse()
            onnx_model = insert_node_by_list(onnx_model, tileKNodes, nodeid)
            tileQNodes = sliceReshapeTransposeQNodes[(sliceNum*sliceQTile):((sliceNum+1)*sliceQTile)]
            tileQNodes.reverse()
            onnx_model = insert_node_by_list(onnx_model, tileQNodes, nodeid)
            tileVNodes = sliceReshapeVNodes[(sliceNum*sliceVTile):((sliceNum+1)*sliceVTile)]
            tileVNodes.reverse()
            onnx_model = insert_node_by_list(onnx_model, tileVNodes, nodeid)
        onnx_model.graph.node.insert(nodeid, newKRSNode)
        onnx_model.graph.node.insert(nodeid, newQRSNode)
        onnx_model.graph.node.insert(nodeid, newVRSNode)
        for serial_name in list(serial_nodes.keys()):
            key_nodes = serial_nodes[serial_name]
            for key_node in key_nodes:
                onnx_model = delete_value_info_by_name(onnx_model, key_node.output[0])
            onnx_model = delete_nodes(onnx_model, key_nodes)
        onnx_model.graph.node.remove(node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    def convert_one_batch(onnx_model, node, nodeid, serial_nodes):
        kConvNode = serial_nodes['k_serial'][0]
        qReshapeNode = serial_nodes['q_serial'][0]
        vReshapeNode = serial_nodes['v_serial'][0]
        kqMatMulNode = serial_nodes['kq_serial'][0]
        kqMulNode = serial_nodes['kq_serial'][1]
        kqAddNode = serial_nodes['kq_serial'][2]
        kqSoftmaxNode = serial_nodes['kq_serial'][3]
        delValInfosList = [vReshapeNode.output[0], kqTransposeNode.output[0], kqSoftmaxNode.output[0], kqAddNode.output[0], 
                           kqMulNode.output[0], kqMatMulNode.output[0], kTransposeNode.output[0], kReshapeNode.output[0],
                           qReshapeNode.output[0]]
        qInShape = get_shape_by_name(onnx_model, qReshapeNode.input[0])
        vInShape = get_shape_by_name(onnx_model, vReshapeNode.input[0])
        kInShape = get_shape_by_name(onnx_model, kConvNode.input[0])
        
        kqAddStaticInput = kqAddNode.input[1] if find_init_by_name(onnx_model, kqAddNode.input[1]) else kqAddNode.input[0]
        kqAddStArr = get_tensor_from_initializer(onnx_model, kqAddStaticInput)
        newKQAddStArr = copy.deepcopy(kqAddStArr)
        newKQAddStTensor = get_initial_by_name(onnx_model, kqAddStaticInput)
        if kqAddStArr.size != 1:
            while True:
                if len(newKQAddStArr.shape) > 4:
                    return onnx_model, False
                elif len(newKQAddStArr.shape) == 4:
                    break
                newKQAddStArr = np.expand_dims(newKQAddStArr, axis=0)
            if newKQAddStArr.shape[2] == kInShape[-1]*kInShape[-2]:
                newKQAddStArr = np.reshape(newKQAddStArr, (1, kInShape[-2], kInShape[-1], newKQAddStArr.shape[-1]))
            elif newKQAddStArr.shape[2] != 1:
                return onnx_model, False
            newKQAddStArr = np.transpose(newKQAddStArr, (0, 3, 1, 2))
            newKQAddStTensor = get_initial_by_value(onnx_model, newKQAddStArr)
            if newKQAddStTensor is None:
                newKQAddStTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, kqAddStaticInput+'_new'),
                                                           data_type=NPDTYPE_2_ONNXDTYPE[newKQAddStArr.dtype],
                                                           dims=newKQAddStArr.shape,
                                                           vals=newKQAddStArr.flatten().tolist())
        
        kqMulStaticInput = kqMulNode.input[1] if find_init_by_name(onnx_model, kqMulNode.input[1]) else kqMulNode.input[0]
        kqMulStArr = get_tensor_from_initializer(onnx_model, kqMulStaticInput)
        newKQMulStArr = copy.deepcopy(kqMulStArr)
        newKQMulStTensor = get_initial_by_name(onnx_model, kqMulStaticInput)
        if kqMulStArr.size != 1:
            while True:
                if len(newKQMulStArr.shape) > 4:
                    return onnx_model, False
                elif len(newKQMulStArr.shape) == 4:
                    break
                newKQMulStArr = np.expand_dims(newKQMulStArr, axis=0)
            if newKQMulStArr.shape[2] == kInShape[-1]*kInShape[-2]:
                newKQMulStArr = np.reshape(newKQMulStArr, (1, kInShape[-2], kInShape[-1], newKQMulStArr.shape[-1]))
            elif newKQMulStArr.shape[2] != 1:
                return onnx_model, False
            newKQMulStArr = np.transpose(newKQMulStArr, (0, 3, 1, 2))
            newKQMulStTensor = get_initial_by_value(onnx_model, newKQMulStArr)
            if newKQMulStTensor is None:
                newKQMulStTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, kqMulStaticInput+'_new'),
                                                           data_type=NPDTYPE_2_ONNXDTYPE[newKQMulStArr.dtype],
                                                           dims=newKQMulStArr.shape,
                                                           vals=newKQMulStArr.flatten().tolist())
        kqMulNode.input[list(kqMulNode.input).index(kqMulStaticInput)] = newKQMulStTensor.name
        kqAddNode.input[list(kqAddNode.input).index(kqAddStaticInput)] = newKQAddStTensor.name
        
        for kqStTensor in [newKQMulStTensor, newKQAddStTensor]:
            if not find_init_by_name(onnx_model, kqStTensor.name):
                onnx_model.graph.initializer.append(kqStTensor)
        
        newQShape = [1, qInShape[1], 1, qInShape[2]*qInShape[3]]
        newQShapeTensor = get_initial_by_value(onnx_model, np.array(newQShape, dtype=np.int64))
        if newQShapeTensor is None:
            newQShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, qReshapeNode.input[1]+'_new'),
                                                     data_type=TensorProto.INT64,
                                                     dims=[len(newQShape)],
                                                     vals=newQShape)
            onnx_model.graph.initializer.append(newQShapeTensor)
        qReshapeNode.input[1] = newQShapeTensor.name
        newQTPNode = onnx.helper.make_node(name=qReshapeNode.name+'_TP',
                                           op_type='Transpose',
                                           inputs=[qReshapeNode.output[0]+'_new'],
                                           outputs=[qReshapeNode.output[0]],
                                           perm=[3, 1, 0, 2])
        qReshapeNode.output[0] = newQTPNode.input[0]
        kqMatMulNode.input[1] = kConvNode.output[0]
        kqMatMulNode.input[0] = newQTPNode.output[0]
        
        newKQConvAttr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newKQConvNode = onnx.helper.make_node(name=kqMatMulNode.name+'_toConv',
                                              op_type='Conv',
                                              inputs=[kConvNode.output[0], newQTPNode.output[0]],
                                              outputs=kqMatMulNode.output,
                                              **newKQConvAttr)
        
        newVShape = [vInShape[1], vInShape[2]*vInShape[3], 1, 1]
        newVShapeTensor = get_initial_by_value(onnx_model, np.array(newVShape, dtype=np.int64))
        if newVShapeTensor is None:
            newVShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, vReshapeNode.input[1]+'_newVShape'),
                                                      data_type=TensorProto.INT64,
                                                      dims=[len(newVShape)],
                                                      vals=newVShape)
            onnx_model.graph.initializer.append(newVShapeTensor)
        vReshapeNode.input[1] = newVShapeTensor.name

        kqSoftmaxAxis = attribute_to_dict(kqSoftmaxNode.attribute).get('axis', 1)   
        kqSoftmaxAxisW4D = kqSoftmaxAxis + 1 if kqSoftmaxAxis >= 0 else kqSoftmaxAxis + 4 
        newKQSoftmaxAxis = [0, 3, 1, 2].index(kqSoftmaxAxisW4D)
        kqSoftmaxNodeAttr = onnx.helper.make_attribute('axis', newKQSoftmaxAxis)
        del kqSoftmaxNode.attribute[:]
        kqSoftmaxNode.attribute.append(kqSoftmaxNodeAttr)
        
        newKQVConvAttr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        newKQVConvNode = onnx.helper.make_node(name=node.name+'_toConv',
                                              op_type='Conv',
                                              inputs=[kqSoftmaxNode.output[0], vReshapeNode.output[0]],
                                              outputs=node.output,
                                              **newKQVConvAttr)
        
        kqvMatMulOutShape = get_shape_by_name(onnx_model, node.output[0])
        tfOutNodes = get_node_by_input(onnx_model, node.output)
        if not (len(tfOutNodes) == 1 and tfOutNodes[0].op_type == 'Reshape'):
            outRSShapeTensor = get_initial_by_value(onnx_model, np.array(kqvMatMulOutShape))
            if outRSShapeTensor is None:
                outRSShapeTensor = onnx.helper.make_tensor(name=node.output[0]+'_cvtShape',
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(kqvMatMulOutShape)],
                                                       vals=kqvMatMulOutShape)
                onnx_model.graph.initializer.append(outRSShapeTensor)
            outRSNode = onnx.helper.make_node(name=node.name+'_reshape',
                                              op_type='Reshape',
                                              inputs=[node.output[0]+'_new', outRSShapeTensor.name],
                                              outputs=node.output)
            onnx_model.graph.node.insert(nodeid+1, outRSNode)
            newKQVConvNode.output[0] = outRSNode.input[0]
        else:
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            newOutShape = [1, vInShape[1], kInShape[2], kInShape[3]]
            tfNextOutShape = get_shape_by_name(onnx_model, tfOutNodes[0].output[0])
            if newOutShape == tfNextOutShape:
                newKQVConvNode.output[0] = tfOutNodes[0].output[0]
                onnx_model.graph.node.remove(tfOutNodes[0])
            else:
                tfOutValueInfo = onnx.helper.make_tensor_value_info(newKQVConvNode.output[0], 1, newOutShape)
                onnx_model.graph.value_info.append(tfOutValueInfo)
        
        for delValInfo in delValInfosList:
            onnx_model = delete_value_info_by_name(onnx_model, delValInfo)
        onnx_model.graph.node.insert(nodeid, newKQVConvNode)
        delNodesList = [kTransposeNode, kReshapeNode, kqTransposeNode, kqMatMulNode, node]
        onnx_model = delete_nodes(onnx_model, delNodesList)
        newKConvNode = copy.deepcopy(kConvNode)
        newQReshapeNode = copy.deepcopy(qReshapeNode)
        onnx_model.graph.node.remove(kConvNode)
        onnx_model.graph.node.remove(qReshapeNode)
        kqMulId = get_node_id(onnx_model, kqMulNode)
        onnx_model = insert_node_by_list(onnx_model, [newKQConvNode, newKConvNode, newQTPNode, newQReshapeNode], kqMulId)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    if node.op_type == 'MatMul':
        tfNodeSerial = get_vit_kqv_block_nodes(onnx_model, node)
        if tfNodeSerial is None:
            return onnx_model, False
        kConvNode = tfNodeSerial['k_serial'][0]
        kConvInShape = get_shape_by_name(onnx_model, kConvNode.input[0])
        qReshapeNode = tfNodeSerial['q_serial'][0]
        qReshapeInShape = get_shape_by_name(onnx_model, qReshapeNode.input[0])
        vReshapeNode = tfNodeSerial['v_serial'][0]
        vReshapeInShape = get_shape_by_name(onnx_model, vReshapeNode.input[0])
        if len(kConvInShape) != 4 or len(qReshapeInShape) != 4 or len(vReshapeInShape) != 4\
            or not (kConvInShape[0] == qReshapeInShape[0] == vReshapeInShape[0]):
            return onnx_model, False
        qReshapeOutShape = get_shape_by_name(onnx_model, qReshapeNode.output[0])
        if len(qReshapeOutShape) not in [3, 4] or qReshapeOutShape[-1] != qReshapeInShape[-2]*qReshapeInShape[-1] \
            or qReshapeOutShape[-3:-2] != qReshapeInShape[-4:-3]:
                return onnx_model, False
        kReshapeNode = tfNodeSerial['k_serial'][1]
        kReshapeOutShape = get_shape_by_name(onnx_model, kReshapeNode.output[0])
        if len(kReshapeOutShape) != len(qReshapeOutShape) or kReshapeOutShape[:-1] != qReshapeOutShape[:-1]:
            return onnx_model, False
        vReshapeOutShape = get_shape_by_name(onnx_model, vReshapeNode.output[0])
        if len(vReshapeOutShape) != len(kReshapeOutShape) or vReshapeOutShape[-1] != kReshapeOutShape[-1]:
            return onnx_model, False
        if not (vReshapeOutShape[:-2] == qReshapeOutShape[:-2] == kReshapeOutShape[:-2]):
            return onnx_model, False
        kTransposeNode = tfNodeSerial['k_serial'][2]
        kTransposePerm = attribute_to_dict(kTransposeNode.attribute).get('perm', list(range(len(kReshapeOutShape))).reverse())
        if kTransposePerm not in [[0, 2, 1], [0, 1, 3, 2]]:
            return onnx_model, False
        kqTransposeNode = tfNodeSerial['kq_serial'][-1]
        kqTransposeInShape = get_shape_by_name(onnx_model, kqTransposeNode.input[0])
        kqTransposePerm = attribute_to_dict(kqTransposeNode.attribute).get('perm', list(range(len(kqTransposeInShape))).reverse())
        if kqTransposePerm != kTransposePerm:
            return onnx_model, False
        if kConvInShape[0] != 1:
            return convert_multi_batch(onnx_model, node, node_index, tfNodeSerial)
        else:
            return convert_one_batch(onnx_model, node, node_index, tfNodeSerial)   
    return onnx_model, False