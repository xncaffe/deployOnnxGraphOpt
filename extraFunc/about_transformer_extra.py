from basicUtil.baseUtil import *

def check_Continue3dimResidual(onnx_model, addNode, reshapeNode):
    reshapeInNode = get_node_by_output(onnx_model, reshapeNode.input[0])
    addNodeOtherInput = addNode.input[1] if addNode.input[0] == reshapeNode.output[0] else addNode.input[0]
    addOutShape = get_shape_by_name(onnx_model, addNode.output[0])
    reshapeInShape = get_shape_by_name(onnx_model, reshapeNode.input[0])
    reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
    if reshapeOutShape != addOutShape:
        return onnx_model
    addOutNodesList = get_node_by_input(onnx_model, addNode.output)
    addNodeId = get_node_id(onnx_model, addNode)
    addOtherInShape = get_shape_by_name(onnx_model, addNodeOtherInput)
    if len(addOtherInShape) == 1 and addOtherInShape[0] == addOutShape[-1]:
        newReshapeShape = [1, 1, reshapeInShape[2], reshapeInShape[3]]
    elif len(addOtherInShape) == 3 and addOtherInShape == addOutShape:
        newReshapeShape = reshapeInShape
    elif len(addOtherInShape) == 3 and addOtherInShape[-1] == 1:
        newReshapeShape = [1, addOtherInShape[1], 1, 1]
    elif len(addOtherInShape) == 3 and addOtherInShape[1] == 1:
        newReshapeShape = [1, 1, reshapeInShape[2], reshapeInShape[3]]
    else:
        return onnx_model
    if not find_init_by_name(onnx_model, addNodeOtherInput):
        addOtherInNode = get_node_by_output(onnx_model, addNodeOtherInput)
        if addOtherInNode.op_type != "Reshape":
            return onnx_model
        addOtherInShapeTensor = onnx.helper.make_tensor(name=addNodeOtherInput+"_newReshape",
                                                        data_type=TensorProto.INT64,
                                                        dims=[4],
                                                        vals=newReshapeShape)
        addOtherReshapeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, addNode.name+"_Reshape"),
                                                    op_type="Reshape",
                                                    inputs=[addNodeOtherInput, addOtherInShapeTensor.name],
                                                    outputs=[addNodeOtherInput+"_reshapeOut"])
        onnx_model.graph.node.insert(addNodeId, addOtherReshapeNode)
        onnx_model.graph.initializer.append(addOtherInShapeTensor)
        addNode.input[list(addNode.input).index(addNodeOtherInput)] = addOtherReshapeNode.output[0]
        if reshapeInNode.op_type == "Transpose":
            addOtherTransposeNode = onnx.helper.make_node(name=get_unique_node_tensor_name(onnx_model, addNode.name+"_Transpose"),
                                                          op_type="Transpose",
                                                          inputs=addOtherReshapeNode.output,
                                                          outputs=[addNodeOtherInput+"_transposeOut"],
                                                          perm=attribute_to_dict(reshapeInNode.attribute).get("perm", []))
            onnx_model.graph.node.insert(addNodeId+1, addOtherTransposeNode)
            addNode.input[list(addNode.input).index(addOtherReshapeNode.output[0])] = addOtherTransposeNode.output[0]
    else:
        addOtherArr = get_tensor_from_initializer(onnx_model, addNodeOtherInput)
        newAddOtherArr = np.reshape(addOtherArr, tuple(newReshapeShape))
        if reshapeInNode.op_type == "Transpose":
            perm = attribute_to_dict(reshapeInNode.attribute).get("perm", [3, 2, 1, 0])
            newAddOtherArr = np.transpose(newAddOtherArr, tuple(perm))
        newAddOtherTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(addNodeOtherInput+"_new"),
                                                    data_type=NPDTYPE_2_ONNXDTYPE[newAddOtherArr.dtype],
                                                    dims=newAddOtherArr.shape,
                                                    vals=newAddOtherArr.flatten().tolist())
        onnx_model.graph.initializer.append(newAddOtherTensor)
        addNode.input[list(addNode.input).index(addNodeOtherInput)] = newAddOtherTensor.name
    addNode.input[list(addNode.input).index(reshapeNode.output[0])] = reshapeInNode.input[0] \
        if reshapeInNode.op_type == "Transpose" else reshapeNode.input[0]
    logger = logging.getLogger("[OptExtraProcess]")
    logger.info("Dealing with continuous transformer residual structures --> node_name: "+addNode.name)
    addNodeId = get_node_id(onnx_model, addNode)
    newAddAfterShapeTensor = onnx.helper.make_tensor(name=addNode.output[0]+"_Shape",
                                                     data_type=TensorProto.INT64,
                                                     dims=[3],
                                                     vals=addOutShape)
    onnx_model.graph.initializer.append(newAddAfterShapeTensor)
    newAddAfterReshapeNode = onnx.helper.make_node(name=addNode.name+"_afterReshape",
                                                   op_type="Reshape",
                                                   inputs=[addNode.output[0], newAddAfterShapeTensor.name],
                                                   outputs=[addNode.output[0]+"_afterReshapeOut"])
    if reshapeInNode.op_type == "Transpose":
        newAddAfterTransposeNode = onnx.helper.make_node(name=addNode.name+"_afterTranspose",
                                                         op_type="Transpose",
                                                         inputs=addNode.output,
                                                         outputs=[addNode.output[0]+"_afterTransposeOut"],
                                                         perm=attribute_to_dict(reshapeInNode.attribute).get("perm", []))
        newAddAfterReshapeNode.input[0] = newAddAfterTransposeNode.output[0]
        onnx_model.graph.node.insert(addNodeId+1, newAddAfterReshapeNode)
        onnx_model.graph.node.insert(addNodeId+1, newAddAfterTransposeNode)
        blkOutTransposeValue = onnx.helper.make_tensor_value_info(newAddAfterTransposeNode.output[0], 1, reshapeInShape)
        onnx_model.graph.value_info.append(blkOutTransposeValue)
    else:
        onnx_model.graph.node.insert(addNodeId+1, newAddAfterReshapeNode)
    blkOutReshapeValue = onnx.helper.make_tensor_value_info(newAddAfterReshapeNode.output[0], 1, addOutShape)
    onnx_model.graph.value_info.append(blkOutReshapeValue)
    onnx_model = delete_value_info_by_name(onnx_model, addNode.output[0])
    for addOutNode in addOutNodesList:
        for input_id, addOutNodeInput in enumerate(addOutNode.input):
            addOutNode.input[input_id] = newAddAfterReshapeNode.output[0] if addOutNodeInput == addNode.output[0] else addOutNodeInput
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    addAfterReshapeOutNodesList = get_node_by_input(onnx_model, newAddAfterReshapeNode.output)
    for addAfterReshapeOutNode in addAfterReshapeOutNodesList:
        if addAfterReshapeOutNode.op_type == "Add":
            onnx_model = check_Continue3dimResidual(onnx_model, addAfterReshapeOutNode, newAddAfterReshapeNode)
    return onnx_model

def get_layernormal_node_dict(onnx_model, addNode):
    if addNode.op_type != "Add":
        return None
    staticAddInput, dynamicAddInput = [addNode.input[1], addNode.input[0]] \
        if find_init_by_name(onnx_model, addNode.input[1]) else [addNode.input[0], addNode.input[1]]
    if not find_init_by_name(onnx_model, staticAddInput):
        return None
    divNode = get_node_by_output(onnx_model, dynamicAddInput)
    if divNode.op_type != "Div":
        return None
    leftMulNode = get_node_by_output(onnx_model, divNode.input[0])
    rightAddNode = get_node_by_output(onnx_model, divNode.input[1])
    if leftMulNode.op_type != "Mul" or rightAddNode.op_type != "Add":
        return None
    staticLeftMulInput, dynamicLeftMulInput = [leftMulNode.input[0], leftMulNode.input[1]] \
        if find_init_by_name(onnx_model, leftMulNode.input[0]) else [leftMulNode.input[1], leftMulNode.input[0]]
    if not find_init_by_name(onnx_model, staticLeftMulInput):
        return None
    staticRightAddInput, dynamicRightAddInput = [rightAddNode.input[1], rightAddNode.input[0]] \
        if find_init_by_name(onnx_model, rightAddNode.input[1]) else [rightAddNode.input[0], rightAddNode.input[1]]
    if not find_init_by_name(onnx_model, staticRightAddInput):
        return None
    rightSqrtNode = get_node_by_output(onnx_model, dynamicRightAddInput)
    if rightSqrtNode.op_type != "Sqrt":
        return None
    rightSecMulNode = get_node_by_output(onnx_model, rightSqrtNode.input[0])
    if rightSecMulNode.op_type != "Mul":
        return None
    staticRightSecMulInput, dynamicRightSecMulInput = [rightSecMulNode.input[1], rightSecMulNode.input[0]] \
        if find_init_by_name(onnx_model, rightSecMulNode.input[1]) else [rightSecMulNode.input[0], rightSecMulNode.input[1]]
    if not find_init_by_name(onnx_model, staticRightSecMulInput):
        return None
    rightReduceMeanNode = get_node_by_output(onnx_model, dynamicRightSecMulInput)
    if rightReduceMeanNode.op_type != "ReduceMean":
        return None
    rightTopMulNode = get_node_by_output(onnx_model, rightReduceMeanNode.input[0])
    if rightTopMulNode.op_type == "Mul":
        if rightTopMulNode.input[0] != rightTopMulNode.input[1]:
            return None
    elif rightTopMulNode.op_type == "Pow":
        powNumArr = get_tensor_from_initializer(onnx_model, rightTopMulNode.input[1])
        if powNumArr.size != 1:
            return None
        elif powNumArr[0] != 2:
            return None
    leftSubNode = get_node_by_output(onnx_model, dynamicLeftMulInput)
    rightSubNode = get_node_by_output(onnx_model, rightTopMulNode.input[0])
    if leftSubNode != rightSubNode or \
        find_init_by_name(onnx_model, leftSubNode.input[0]) or find_init_by_name(onnx_model, leftSubNode.input[1]):
            return None
    leftLNInNode = get_node_by_output(onnx_model, leftSubNode.input[0])
    topReduceMeanNode = get_node_by_output(onnx_model, leftSubNode.input[1])
    if topReduceMeanNode.op_type != "ReduceMean":
        return None
    rightLNInNode = get_node_by_output(onnx_model, topReduceMeanNode.input[0])
    if rightLNInNode != leftLNInNode:
        return None
    reNodesDict = {'input': leftLNInNode,
                   'topReduceMean': topReduceMeanNode,
                   'sub': leftSubNode,
                   'leftMul': leftMulNode,
                   'rightTopMul': rightTopMulNode,
                   'rightReduceMean': rightReduceMeanNode,
                   'rightSecMul': rightSecMulNode,
                   'Sqrt': rightSqrtNode,
                   'rightAdd': rightAddNode,
                   'div': divNode,
                   'output': addNode}
    return reNodesDict

def get_vit_kqv_block_nodes(onnx_model, node):
    if node.op_type != 'MatMul':
        return None
    vReshape = get_node_by_output(onnx_model, node.input[0])
    if vReshape is None or vReshape.op_type != 'Reshape':
        return None
    kqTranspose = get_node_by_output(onnx_model, node.input[1])
    if kqTranspose is None or kqTranspose.op_type != 'Transpose':
        return None
    kqSoftmax = get_node_by_output(onnx_model, kqTranspose.input[0])
    if kqSoftmax is None or kqSoftmax.op_type != 'Softmax':
        return None
    kqAdd = get_node_by_output(onnx_model, kqSoftmax.input[0])
    if kqAdd is None or kqAdd.op_type != 'Add':
        return None
    if not find_init_by_name(onnx_model, kqAdd.input[0]) and not find_init_by_name(onnx_model, kqAdd.input[1]):
        return None
    dynamicKQAddIn = kqAdd.input[0] if find_init_by_name(onnx_model, kqAdd.input[1]) else kqAdd.input[1]
    if find_init_by_name(onnx_model, dynamicKQAddIn):
        return None
    kqMul = get_node_by_output(onnx_model, dynamicKQAddIn)
    if kqMul is None or kqMul.op_type != 'Mul':
        return None
    if not find_init_by_name(onnx_model, kqMul.input[0]) and not find_init_by_name(onnx_model, kqMul.input[1]):
        return None
    dynamicKQMulIn = kqMul.input[0] if find_init_by_name(onnx_model, kqMul.input[1]) else kqMul.input[1]
    if find_init_by_name(onnx_model, dynamicKQMulIn):
        return None
    kqMatMul = get_node_by_output(onnx_model, dynamicKQMulIn)
    if kqMatMul is None or kqMatMul.op_type != 'MatMul':
        return None
    qReshape = get_node_by_output(onnx_model, kqMatMul.input[1])
    if qReshape is None or qReshape.op_type != 'Reshape':
        return None
    kTranspose = get_node_by_output(onnx_model, kqMatMul.input[0])
    if kTranspose is None or kTranspose.op_type != 'Transpose':
        return None
    kReshape = get_node_by_output(onnx_model, kTranspose.input[0])
    if kReshape is None or kReshape.op_type != 'Reshape':
        return None
    kConv = get_node_by_output(onnx_model, kReshape.input[0])
    if kConv is None or kConv.op_type != 'Conv':
        return None
    # kSplit = get_node_by_output(onnx_model, kConv.input[0])
    # if kConv is None or kSplit.op_type != 'Split':
    #     return None
    # qSplit = get_node_by_output(onnx_model, qReshape.input[0])
    # if qSplit is None or qSplit.op_type != 'Split':
    #     return None
    # vSplit = get_node_by_output(onnx_model, vReshape.input[0])
    # if vSplit is None or vSplit.op_type != 'Split':
    #     return None
    # if kSplit != qSplit != vSplit:
    #     return None
    # topConv = get_node_by_output(onnx_model, kSplit.input[0])
    # if topConv is None or topConv.op_type != 'Conv':
    #     return None
    # topAdd = get_node_by_output(onnx_model, topConv.input[0])
    # if topAdd is None or topAdd.op_type is 'Add':
    #     return None
    reNodesDict = {
        'k_serial': [kConv, kReshape, kTranspose],
        'q_serial': [qReshape],
        'kq_serial': [kqMatMul, kqMul, kqAdd, kqSoftmax, kqTranspose],
        'v_serial': [vReshape]
    }
    return reNodesDict

def get_custom_three_conv_kqv_block_nodes(onnx_model, node):
    if node.op_type != 'MatMul':
        return None
    vReshape = get_node_by_output(onnx_model, node.input[0])
    if vReshape is None or vReshape.op_type != 'Reshape':
        return None
    vConv = get_node_by_output(onnx_model, vReshape.input[0])
    if vConv is None or vConv.op_type != 'Conv':
        return None
    kqTranspose = get_node_by_output(onnx_model, node.input[1])
    if kqTranspose is None or kqTranspose.op_type != 'Transpose':
        return None
    kqSoftmax = get_node_by_output(onnx_model, kqTranspose.input[0])
    if kqSoftmax is None or kqSoftmax.op_type != 'Softmax':
        return None
    kqMatMul = get_node_by_output(onnx_model, kqSoftmax.input[0])
    if kqMatMul is None or kqMatMul.op_type != 'MatMul':
        return None
    kTranspose = get_node_by_output(onnx_model, kqMatMul.input[0])
    if kTranspose is None or kTranspose.op_type != 'Transpose':
        return None
    kReshape = get_node_by_output(onnx_model, kTranspose.input[0])
    if kReshape is None or kReshape.op_type != 'Reshape':
        return None
    kConv = get_node_by_output(onnx_model, kReshape.input[0])
    if kConv is None or kConv.op_type != 'Conv':
        return None
    qReshape = get_node_by_output(onnx_model, kqMatMul.input[1])
    if qReshape is None or qReshape.op_type != 'Reshape':
        return None
    qConv = get_node_by_output(onnx_model, qReshape.input[0])
    if qConv is None or qConv.op_type != 'Conv':
        return None
    reNodesDict = {
        'k_serial': [kConv, kReshape, kTranspose],
        'q_serial': [qConv, qReshape],
        'kq_serial': [kqMatMul, kqSoftmax, kqTranspose],
        'v_serial': [vConv, vReshape]
    }
    return reNodesDict