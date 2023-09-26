from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionSeparatedLayerNormal(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Div", "Add"]):
        nodes_list = get_node_serial_group(onnx_model, node, ["Div", "Add"])        
        lastDivNode = nodes_list[0]
        lastAddNode = nodes_list[1]
        lastDivNode_outTensorShape = get_shape_by_name(onnx_model, lastDivNode.output[0])
        lastAddNode_baisShape = get_shape_by_name(onnx_model, lastAddNode.input[1])
        if len(lastDivNode_outTensorShape) != 3 or len(lastAddNode_baisShape) != 1:
            return onnx_model, False
        '''
        find left branch node list
        '''
        leftMulNode = get_node_by_output(onnx_model, lastDivNode.input[0])
        if leftMulNode.op_type != "Mul":
            return onnx_model, False
        ln_weight_tensor = get_tensor_from_initializer(onnx_model, leftMulNode.input[1])
        leftSubOutName = leftMulNode.input[0]
        ln_weight_name = leftMulNode.input[1]
        if not ln_weight_tensor.size:
            ln_weight_tensor = get_tensor_from_initializer(onnx_model, leftMulNode.input[0])
            if not ln_weight_tensor.size:
                return onnx_model, False
            leftSubOutName = leftMulNode.input[1]
            ln_weight_name = leftMulNode.input[0]
        leftSubNode = get_node_by_output(onnx_model, leftSubOutName)
        if leftSubNode.op_type != "Sub":
            return onnx_model, False
        if get_tensor_from_initializer(onnx_model, leftSubNode.input[0]) or \
            get_tensor_from_initializer(onnx_model, leftSubNode.input[1]):
                return onnx_model, False
        firstInNode = get_node_by_output(onnx_model, leftSubNode.input[0])
        leftReduceMeanNode = get_node_by_output(onnx_model, leftSubNode.input[1])
        if leftReduceMeanNode.op_type != "ReduceMean" or leftReduceMeanNode.input[0] != firstInNode.output[0]:
            return onnx_model, False
        
        
        leftReduceMeanAttr = attribute_to_dict(leftReduceMeanNode.attribute)
        axis = leftReduceMeanAttr["axes"]
        axis = axis[0] if isinstance(axis, list) else axis
        if lastAddNode_baisShape[0] != lastDivNode_outTensorShape[axis]:
            return onnx_model, False
        '''
        find right branch node list
        '''
        rightAddNode = get_node_by_output(onnx_model, lastDivNode.input[1])
        if rightAddNode.op_type != "Add":
            return onnx_model, False
        ln_eps_tensor = get_tensor_from_initializer(onnx_model, rightAddNode.input[1])
        if not ln_eps_tensor:
            return onnx_model, False
        try:
            eps_value = ln_eps_tensor[0]
        except:
            eps_value = ln_eps_tensor
        rightSqrtNode = get_node_by_output(onnx_model, rightAddNode.input[0])
        if rightSqrtNode.op_type != "Sqrt":
            return onnx_model, False
        rightDivNode = get_node_by_output(onnx_model, rightSqrtNode.input[0])
        if rightDivNode.op_type != "Div":
            return onnx_model, False
        varDiv_tensor = get_tensor_from_initializer(onnx_model, rightDivNode.input[1])
        if not varDiv_tensor or varDiv_tensor.size != 1:
            return onnx_model, False
        try:
            varDiv_value = varDiv_tensor[0]
        except:
            varDiv_value = varDiv_tensor
        if int(varDiv_value) != lastAddNode_baisShape[0] - 1:
            return onnx_model, False
        rightSumMulNode = get_node_by_output(onnx_model, rightDivNode.input[0])
        if rightSumMulNode.op_type != "Mul":
            return onnx_model, False
        numMul_tensor = get_tensor_from_initializer(onnx_model, rightSumMulNode.input[1])
        try:
            numMul_value = numMul_tensor[0]
        except:
            numMul_value = numMul_tensor
        if int(numMul_value) != lastAddNode_baisShape[0]:
            return onnx_model, False
        rightSumReduceMeanNode = get_node_by_output(onnx_model, rightSumMulNode.input[0])
        if rightSumReduceMeanNode.op_type != "ReduceMean":
            return onnx_model, False
        rightSumReduceMeanAttr = attribute_to_dict(rightSumReduceMeanNode.attribute)
        if rightSumReduceMeanAttr["axes"] != leftReduceMeanAttr["axes"]:
            return onnx_model, False
        right2PowMulNode = get_node_by_output(onnx_model, rightSumReduceMeanNode.input[0])
        if right2PowMulNode.op_type != "Mul" or right2PowMulNode.input[0] != right2PowMulNode.input[1]:
            return onnx_model, False
        rightSubNode = get_node_by_output(onnx_model, right2PowMulNode.input[0])
        if rightSubNode.op_type != "Sub" or rightSubNode.input[0] != firstInNode.output[0]:
            return onnx_model, False
        rightReduceMeanNode = get_node_by_output(onnx_model, rightSubNode.input[1])
        if rightReduceMeanNode.op_type != "ReduceMean" or rightReduceMeanNode.input[0] != firstInNode.output[0]:
            return onnx_model, False
        rightReduceMeanAttr = attribute_to_dict(rightReduceMeanNode.attribute)
        if rightReduceMeanAttr["axes"] != leftReduceMeanAttr["axes"]:
            return onnx_model, False
        
        del_node_lists = [lastAddNode, lastDivNode, leftMulNode, leftSubNode, leftReduceMeanNode, rightAddNode,
                          rightSqrtNode, rightDivNode, rightSumMulNode, rightSumReduceMeanNode, right2PowMulNode]
        for del_node in [rightSubNode, rightReduceMeanNode]:
            if del_node not in del_node_lists:
                del_node_lists.append(del_node)
        layerNormalAttr = {"axis": axis, "epsilon": float(eps_value)}
        layerNormalNode = onnx.helper.make_node(name=firstInNode.name+"_insertLayerNormal",
                                                inputs=[firstInNode.input[0], ln_weight_name, lastAddNode.input[1]],
                                                outputs = lastAddNode.output,
                                                op_type = "LayerNormalization",
                                                **layerNormalAttr)
        onnx_model.graph.node.insert(node_index, layerNormalNode)
        onnx_model = delete_nodes(onnx_model, del_node_lists)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceDivByMul(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Div"]):
        divNode = get_node_serial_group(onnx_model, node, ["Div"])[0]
        fixInput = divNode.input[1] if find_init_by_name(onnx_model, divNode.input[1]) else divNode.input[0]
        if fixInput != divNode.input[1] or not find_init_by_name(onnx_model, fixInput):
            return onnx_model, False
        fixInputValue = get_tensor_from_initializer(onnx_model, fixInput)
        fixInputValue = 1.0 / fixInputValue
        newInitial = onnx.helper.make_tensor(name=fixInput,
                                             data_type=NPDTYPE_2_ONNXDTYPE[fixInputValue.dtype],
                                             dims=fixInputValue.shape,
                                             vals=fixInputValue.reshape(-1).tolist())
        if not find_other_node_by_input(onnx_model, divNode, fixInput):
            onnx_model = delete_initializer_by_name(onnx_model, fixInput)
        else:
            newInitial.name += "_{}ToMul_init".format(divNode.name)
        onnx_model.append(newInitial)
        newMulNode = onnx.helper.make_node(name=divNode.name+"ToMul",
                                           op_type="Mul",
                                           inputs=[divNode.input[0], newInitial.name],
                                           outputs=divNode.output)
        onnx_model.graph.node.remove(divNode)
        onnx_model.graph.node.insert(node_index, newMulNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionMultiMulDiv(onnx_model, node, node_index):
    def find_nodes_list_static_muldiv(onnx_model, node, node_index):
        if check_node_serial_group(onnx_model, node, ["Div"]) or check_node_serial_group(onnx_model, node, ["Mul"]):
            re_nodes_list = []
            effectNode = get_node_serial_group(onnx_model, node, ["Div"])
            if not effectNode:
                effectNode = get_node_serial_group(onnx_model, node, ["Mul"])
            effectNode = effectNode[0] 
            staticInput, dynamicInput = (effectNode.input[1], effectNode.input[0]) \
                if find_init_by_name(onnx_model, effectNode.input[1]) else (effectNode.input[0], effectNode.input[1])
            if (not find_init_by_name(onnx_model, staticInput)) or (effectNode.op_type=="Div" and staticInput != effectNode.input[1]):
                return []
            re_nodes_list.append(effectNode)
            staticValueArr = get_tensor_from_initializer(onnx_model, staticInput)
            newStaticValueArr = 1.0 / staticValueArr if effectNode.op_type == "Div" else staticValueArr
            while True:
                currentNode = get_node_by_output(onnx_model, dynamicInput)
                if currentNode is None or currentNode.op_type not in ["Div", "Mul"]:
                    break
                staticInput, dynamicInput = (currentNode.input[1], currentNode.input[0]) \
                    if find_init_by_name(onnx_model, currentNode.input[1]) else (currentNode.input[0], currentNode.input[1])
                if (not find_init_by_name(onnx_model, staticInput)) or (currentNode.op_type == "Div" and staticInput != currentNode.input[1]):
                    break
                initValueArr = get_tensor_from_initializer(onnx_model, staticInput)
                newStaticValueArr = newStaticValueArr*(1.0/initValueArr) if currentNode.op_type == "Div" else newStaticValueArr*initValueArr
                re_nodes_list.append(currentNode)
            if len(re_nodes_list) == 1 and re_nodes_list[0].op_type == "Mul":
                return []
            newStaticValueArr = newStaticValueArr.astype(np.float32)
            return [re_nodes_list, staticInput, dynamicInput, newStaticValueArr]
        return []
    
    validInfos = find_nodes_list_static_muldiv(onnx_model, node, node_index)
    if not validInfos:
        return onnx_model, False
    newInitTensorName = validInfos[1]+"_{}_newTensor".format(validInfos[0][-1].name) \
        if find_other_node_by_input(onnx_model, validInfos[0][-1], validInfos[1]) else validInfos[1]
    newInitTensor = onnx.helper.make_tensor(name=newInitTensorName,
                                            data_type=NPDTYPE_2_ONNXDTYPE[validInfos[-1].dtype],
                                            dims=validInfos[-1].shape,
                                            vals=validInfos[-1].flatten().tolist())
    if newInitTensorName == validInfos[1]:
        onnx_model = delete_initializer_by_name(onnx_model, validInfos[1])
    newMulNode = onnx.helper.make_node(name=validInfos[0][-1].name+"_fusionMul",
                                       op_type="Mul",
                                       inputs=[validInfos[2], newInitTensor.name],
                                       outputs=validInfos[0][0].output)
    onnx_model.graph.initializer.append(newInitTensor)
    onnx_model.graph.node.insert(node_index, newMulNode)
    onnx_model = delete_nodes(onnx_model, validInfos[0])
    onnx_model = delete_useless_input_in_initializer(onnx_model)

    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionMultiSubReduceMean(onnx_model, node, node_index):
    def split_reduceMean_sub_node(nodes_list):
        reduceMean_nodes = []
        sub_nodes = []
        for x_node in nodes_list:
            if x_node.op_type == "ReduceMean":
                reduceMean_nodes.append(x_node)
            elif x_node.op_type == "Sub":
                sub_nodes.append(x_node)
        return reduceMean_nodes, sub_nodes
    
    def match_reduceMean_sub_node(reduceMeanNodes, subNodes):
        matchDicts = {}
        for sub_node in subNodes:
            for reduceMean_node in reduceMeanNodes:
                if reduceMean_node.output[0] in sub_node.input and reduceMean_node.input[0] in sub_node.input:
                    matchDicts[sub_node.name]=[sub_node, reduceMean_node]
        return matchDicts
    
    if node.op_type != "Sub" or \
        find_init_by_name(onnx_model, node.input[0]) or find_init_by_name(onnx_model, node.input[1]):
            return onnx_model, False
    leftInputNode = get_node_by_output(onnx_model, node.input[0])
    rightInputNode = get_node_by_output(onnx_model, node.input[1])
    if leftInputNode.op_type != "ReduceMean" and rightInputNode.op_type != "ReduceMean":
        return onnx_model, False
    reduceMeanNode, preNode = (rightInputNode, leftInputNode) \
        if rightInputNode.op_type == "ReduceMean" else (leftInputNode, rightInputNode)       
    if reduceMeanNode.input[0] not in node.input:
        return onnx_model, False
    parallNodes = get_node_by_input(onnx_model, [preNode.output[0]])
    if len(parallNodes) < 3:
        return onnx_model, False
    parallNodes.remove(reduceMeanNode)
    parallNodes.remove(node)
    reduceMeanNodeAttrDict = attribute_to_dict(reduceMeanNode.attribute)
    reduceMeanNodeAxes = reduceMeanNodeAttrDict.get("axes", 1)
    reduceMeanNodeAxes = reduceMeanNodeAxes[0] if isinstance(reduceMeanNodeAxes, list) else reduceMeanNodeAxes
    reduceMeanNodeInputShape = get_shape_by_name(onnx_model, reduceMeanNode.input[0])
    reduceMeanNodeAxes = reduceMeanNodeAxes if reduceMeanNodeAxes >= 0 else len(reduceMeanNodeInputShape)+reduceMeanNodeAxes
    reduceMeanNodeKeepdims = reduceMeanNodeAttrDict.get("keepdims", 1)
    
    otherReduceMeanNodes, otherSubNodes = split_reduceMean_sub_node(parallNodes)
    matchDict = match_reduceMean_sub_node(otherReduceMeanNodes, otherSubNodes)
    delNodes_list = []
    for subNodeName in matchDict:
        otherSubNode, otherReduceMeanNode = matchDict[subNodeName][0], matchDict[subNodeName][1]
        if reduceMeanNode.input[0] != otherReduceMeanNode.input[0] or reduceMeanNode.input[0] not in otherSubNode.input:
            continue
        if list(otherSubNode.input).index(reduceMeanNode.input[0]) != list(node.input).index(reduceMeanNode.input[0]):
            continue
        otherReduceMeanNodeAttrDict = attribute_to_dict(otherReduceMeanNode.attribute)
        if reduceMeanNodeKeepdims != otherReduceMeanNodeAttrDict.get("keepdims", 1):
            continue
        otherReduceMeanNodeAxes = otherReduceMeanNodeAttrDict.get("axes", 1)
        otherReduceMeanNodeAxes = otherReduceMeanNodeAxes[0] if isinstance(otherReduceMeanNodeAxes, list) else otherReduceMeanNodeAxes
        otherReduceMeanNodeAxes = otherReduceMeanNodeAxes if otherReduceMeanNodeAxes >= 0 else len(reduceMeanNodeInputShape)+otherReduceMeanNodeAxes
        if otherReduceMeanNodeAxes != reduceMeanNodeAxes:
            continue
        otherSubOutNodes = get_node_by_input(onnx_model, otherSubNode.output)
        for otherSubOutNode in otherSubOutNodes:
            otherSubOutNode.input[list(otherSubOutNode.input).index(otherSubNode.output[0])] = node.output[0]
        if otherSubNode not in delNodes_list:
            delNodes_list.append(otherSubNode)
        if otherReduceMeanNode not in delNodes_list:
            delNodes_list.append(otherReduceMeanNode)
    if not delNodes_list:
        return onnx_model, False
    onnx_model = delete_nodes(onnx_model, delNodes_list)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Transpose"]):
        transposeNode = get_node_serial_group(onnx_model, node, ["Transpose"])[0]
        nextNodesList = get_node_by_input(onnx_model, transposeNode.output)
        if not nextNodesList:
            return onnx_model, False
        nextTransposeNodesList = [nextNode for nextNode in nextNodesList if nextNode.op_type == "Transpose"]
        if not nextTransposeNodesList:
            return onnx_model, False
        netOutputNames = [netOutput.name for netOutput in onnx_model.graph.output]
        transposeInShape = get_shape_by_name(onnx_model, transposeNode.input[0])
        transposeAttrDict = attribute_to_dict(transposeNode.attribute)
        transposePerm = transposeAttrDict.get("perm", list(range(len(transposeInShape))).reverse())
        for nextTransposeNode in nextTransposeNodesList:
            nextTransposeAttrDict = attribute_to_dict(nextTransposeNode.attribute)
            nextTransposePerm = nextTransposeAttrDict.get("perm", list(range(len(transposeInShape))).reverse())
            newFusionPerm = [transposePerm[permId] for permId in nextTransposePerm]
            if newFusionPerm == list(range(len(transposeInShape))):
                secondNodesList = get_node_by_input(onnx_model, nextTransposeNode.output)
                if nextTransposeNode.output[0] in netOutputNames:
                    del onnx_model.graph.output[netOutputNames.index(nextTransposeNode.output[0])]
                    onnx_model.graph.output.extend(get_value_info_by_name(onnx_model, transposeNode.input[0]))
                for secondNode in secondNodesList:
                    secondNode.input[list(secondNode.input).index(nextTransposeNode.output[0])] = transposeNode.input[0]
                onnx_model.graph.node.remove(nextTransposeNode)
            else:
                nextTransposeNode.input[0] = transposeNode.input[0]
                del nextTransposeNode.attribute[:]
                newFusionPermAttribute = onnx.helper.make_attribute("perm", newFusionPerm)
                nextTransposeNode.attribute.append(newFusionPermAttribute)
        if len(get_node_by_input(onnx_model, transposeNode.output)) == 0:
            del onnx_model.graph.node[node_index]
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        onnx_model = delete_useless_value_info(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionMultiBranchReshapeTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ["Reshape", "Transpose"]):
        reshapeNode, transposeNode = get_node_serial_group(onnx_model, node, ["Reshape", "Transpose"])
        blkOutNodesList = get_node_by_input(onnx_model, [reshapeNode.input[0]])
        if not blkOutNodesList:
            return onnx_model, False
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        transposeAttributeDict = attribute_to_dict(transposeNode.attribute)
        transposePerm = transposeAttributeDict.get("perm", list(range(len(reshapeOutShape))).reverse())
        delNodesList = []
        for blkOutNode in blkOutNodesList:
            if blkOutNode.name == reshapeNode.name or blkOutNode.op_type != "Reshape":
                continue
            otherReshapeNode = blkOutNode
            otherReshapeOutShape = get_shape_by_name(onnx_model, otherReshapeNode.output[0])
            if otherReshapeOutShape != reshapeOutShape:
                continue
            otherNextNodesList = get_node_by_input(onnx_model, otherReshapeNode.output)
            otherTransposeNodesList = [otherNode for otherNode in otherNextNodesList if otherNode.op_type == "Transpose"]
            meetOtherTransposeNodesList = []
            for otherTransposeNode in otherTransposeNodesList:
                otherTransposePerm = attribute_to_dict(otherTransposeNode.attribute).get("perm", list(range(len(reshapeOutShape))).reverse())
                if otherTransposePerm != transposePerm:
                    continue
                delNodesList.append(otherTransposeNode)
                meetOtherTransposeNodesList.append(otherTransposeNode)
                otherTranposeOutNodesList = get_node_by_input(onnx_model, otherTransposeNode.output)
                for otherTransposeOutNode in otherTranposeOutNodesList:
                    otherTransposeOutNode.input[list(otherTransposeOutNode.input).index(otherTransposeNode.output[0])] = transposeNode.output[0]
            if len(meetOtherTransposeNodesList) == len(otherTransposeNodesList) and len(otherTransposeNodesList) == len(otherNextNodesList):
                delNodesList.append(otherReshapeNode)
        if not delNodesList:
            return onnx_model, False
        for delNode in delNodesList:
            onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
        onnx_model = delete_nodes(onnx_model, delNodesList)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeReshapeReshapeTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Reshape', 'Transpose']):
        serialNodesList = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Reshape', 'Transpose'])
        topTPNode, topRSNode, botRSNode, botTPNode = serialNodesList
        topRSInShape = get_shape_by_name(onnx_model, topRSNode.input[0])
        botRSOutShape = get_shape_by_name(onnx_model, botRSNode.output[0])
        if topRSInShape != botRSOutShape:
            return onnx_model, False
        topTPPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topRSInShape))).reverse())
        botTPPerm = attribute_to_dict(botTPNode.attribute).get('perm', list(range(len(botRSOutShape))).reverse())
        backTopPerm = [topTPPerm[perm_id] for perm_id in topTPPerm]
        backTopPerm = topTPPerm if list(range(len(topRSInShape))) == backTopPerm else backTopPerm
        if botTPPerm != backTopPerm:
            newPerm = [topTPPerm[perm_id] for perm_id in botTPPerm]
            newTranspose = onnx.helper.make_node(name=topTPNode.name+'_fusionTranspose',
                                                 op_type='Transpose',
                                                 inputs=topTPNode.input,
                                                 outputs=botTPNode.output,
                                                 perm=newPerm)
            onnx_model.graph.node.insert(node_index, newTranspose)
        else:
            botTPOutNodesList = get_node_by_input(onnx_model, botTPNode.output)
            for botTPOutNode in botTPOutNodesList:
                for inId, botTPOutNodeIn in enumerate(botTPOutNode.input):
                    botTPOutNode.input[inId] = topTPNode.input[0] if botTPOutNodeIn == botTPNode.output[0] else botTPOutNodeIn
        onnx_model.graph.node.remove(botTPNode)
        netOutNames = [netOutput.name for netOutput in onnx_model.graph.output]
        for delNode in [botRSNode, topRSNode, topTPNode]:
            delOutNodesList = get_node_by_input(onnx_model, delNode.output)
            if not delOutNodesList and delNode.output[0] not in netOutNames:
                onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
                onnx_model.graph.node.remove(delNode)
            else:
                break
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionConvConvAdd(onnx_model, node, node_index):
    def check_param_can_merge(leftNode, rightNode):
        leftAttr = attribute_to_dict(leftNode.attribute)
        rightAttr = attribute_to_dict(rightNode.attribute)
        if leftAttr.get('auto_pad') is not None or rightAttr.get('auto_pad') is not None:
            return False
        lDilations = leftAttr.get('dilations', [1, 1])
        rDilations = rightAttr.get('dilations', [1, 1])
        lGroup = leftAttr.get('group', 1)
        rGroup = rightAttr.get('group', 1)
        lKnShape = leftAttr.get('kernel_shape', [1, 1])
        rKnShape = rightAttr.get('kernel_shape', [1, 1])
        lPads = leftAttr.get('pads', [0, 0, 0, 0])
        rPads = rightAttr.get('pads', [0, 0, 0, 0])
        lStrides = leftAttr.get('strides', [1, 1])
        rStrides = rightAttr.get('strides', [1, 1])
        if lDilations == rDilations and lGroup == rGroup and lKnShape == rKnShape and lPads == rPads and lStrides == rStrides:
            return True
        else:
            return False
    if check_node_serial_group(onnx_model, node, ['Conv', 'Add']):
        leftConvNode, addNode = get_node_serial_group(onnx_model, node, ['Conv', 'Add'])
        addOtherInput = addNode.input[1] if leftConvNode.output[0] == addNode.input[0] else addNode.input[0]
        if find_init_by_name(onnx_model, addOtherInput):
            return onnx_model, False
        rightConvNode = get_node_by_output(onnx_model, addOtherInput)
        if rightConvNode is None or rightConvNode.op_type != 'Conv':
            return onnx_model, False
        lCNodeOShape = get_shape_by_name(onnx_model, leftConvNode.output[0])
        rCNodeOShape = get_shape_by_name(onnx_model, addOtherInput)
        if lCNodeOShape != rCNodeOShape:
            return onnx_model, False
        leftConvWt = get_tensor_from_initializer(onnx_model, leftConvNode.input[1])
        rightConvWt = get_tensor_from_initializer(onnx_model, rightConvNode.input[1])
        if not leftConvWt.size or not rightConvWt.size or not check_param_can_merge(leftConvNode, rightConvNode):
            return onnx_model, False
        leftConvBs = get_tensor_from_initializer(onnx_model, leftConvNode.input[2]) if len(leftConvNode.input) == 3 else None
        rightConvBs = get_tensor_from_initializer(onnx_model, rightConvNode.input[2]) if len(rightConvNode.input) == 3 else None
        if leftConvBs is not None or rightConvBs is not None:
            leftConvBs = np.zeros(lCNodeOShape[1], dtype=np.float32) if leftConvBs is None else leftConvBs
            rightConvBs = np.zeros(rCNodeOShape[1], dtype=np.float32) if rightConvBs is None else rightConvBs
        newConcatNode = onnx.helper.make_node(name=addNode.name+"_toConcat",
                                              op_type='Concat',
                                              inputs=[leftConvNode.input[0], rightConvNode.input[0]],
                                              outputs=[addNode.output[0]+'_concat_out'],
                                              axis=1)
        newConvWt = np.concatenate((leftConvWt, rightConvWt), axis=1)
        newConvBs = None
        if leftConvBs is not None and rightConvBs is not None:
            newConvBs = np.concatenate(leftConvBs, rightConvBs)
        newConvWtTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, addNode.name+'_fusionConv_wt'),
                                                  data_type=NPDTYPE_2_ONNXDTYPE[newConvWt.dtype],
                                                  dims=newConvWt.shape,
                                                  vals=newConvWt.flatten().tolist())
        onnx_model.graph.initializer.append(newConvWtTensor)
        newConvInputs = [newConcatNode.output[0], newConvWtTensor.name]
        if newConvBs is not None:
            newConvBsTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, addNode.name+'_fusionConv_bs'),
                                                      data_type=NPDTYPE_2_ONNXDTYPE[newConvBs.dtype],
                                                      dims=newConvBs.shape,
                                                      vals=newConvBs.tolist())
            onnx_model.graph.initializer.append(newConvBsTensor)
            newConvInputs.append(newConvBsTensor.name)
        newConvAttrDict = attribute_to_dict(leftConvNode.attribute)
        newConvNode = onnx.helper.make_node(name=addNode.name+'_fusionConv',
                                            op_type="Conv",
                                            inputs=newConvInputs,
                                            outputs=addNode.output,
                                            **newConvAttrDict)
        newConcatOutShape = get_shape_by_name(onnx_model, leftConvNode.input[0])
        newConcatOutShape[1] += get_shape_by_name(onnx_model, rightConvNode.input[0])[1]
        newConcatOutValue = onnx.helper.make_tensor_value_info(newConcatNode.output[0], 1, newConcatOutShape)
        onnx_model.graph.value_info.append(newConcatOutValue)
        addNodeId = get_node_id(onnx_model, addNode)
        onnx_model.graph.node.insert(addNodeId, newConvNode)
        onnx_model.graph.node.insert(addNodeId, newConcatNode)
        onnx_model = delete_nodes(onnx_model, [leftConvNode, rightConvNode, addNode])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        onnx_model = delete_value_info_by_name(onnx_model, leftConvNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, rightConvNode.output[0])
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionMultiConcat(onnx_model, node, node_index):
    def get_concat_serial_group(onnx_model, concat1st):
        reConcatNodes = [concat1st]
        _1stAxis = attribute_to_dict(concat1st.attribute).get('axis', 1)
        nextConcatsList = [concat1st]
        while True:
            concatSerialNodes = []
            for nextConcatNode in nextConcatsList:
                nextNodes = get_node_by_input(onnx_model, nextConcatNode.output)
                nextConcatNodes = [next_node for next_node in nextNodes \
                    if next_node.op_type == 'Concat' and attribute_to_dict(next_node.attribute).get('axis', 1) == _1stAxis]
                nextConcatNodes = [nextConcatNode for nextConcatNode in nextConcatNodes if nextConcatNode not in concatSerialNodes]
                concatSerialNodes += nextConcatNodes
            nextConcatsList = concatSerialNodes
            if not nextConcatsList:
                break
            reConcatNodes += nextConcatsList
        return reConcatNodes
    if node.op_type == 'Concat':
        concatNodesList = get_concat_serial_group(onnx_model, node)
        if len(concatNodesList) == 1:
            return onnx_model, False
        newConcatInputs = []
        for concatNode in concatNodesList:
            concatNodeInputs = list(concatNode.input)
            concatNodeInputs.reverse()
            for input in concatNodeInputs:
                inNode = get_node_by_output(onnx_model, input)
                if inNode not in concatNodesList and input not in newConcatInputs:
                    newConcatInputs.append(input)
        newConcatInputs.reverse()
        lastConcatNode, lastId = get_last_node_by_serial(onnx_model, concatNodesList)
        newConcatNode = onnx.helper.make_node(name=lastConcatNode.name+'_fusionConcat',
                                              op_type='Concat', 
                                              inputs=newConcatInputs,
                                              outputs=lastConcatNode.output,
                                              axis=attribute_to_dict(lastConcatNode.attribute).get('axis', 1))
        onnx_model.graph.node.insert(lastId, newConcatNode)
        for concatNode in concatNodesList:
            if concatNode.name != lastConcatNode.name:
                onnx_model = delete_value_info_by_name(onnx_model, concatNode.output[0])
        onnx_model = delete_nodes(onnx_model, concatNodesList)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionConcatSlice(onnx_model, node, node_index):
    if node.op_type == 'Concat':
        concatOutShape = get_shape_by_name(onnx_model, node.output[0])
        if len(concatOutShape) != 4:
            return onnx_model, False
        inputsList = list(node.input)
        outNodesList = get_node_by_input(onnx_model, node.output)
        # if len(outNodesList) != len(inputsList):
        #     return onnx_model, False
        outSliceNodes = [outNode for outNode in outNodesList if outNode.op_type == 'Slice']
        if len(outSliceNodes) != len(inputsList):
            return onnx_model, False
        concatAxis = attribute_to_dict(node.attribute).get('axis', 1)
        concatAxis = len(concatOutShape) + concatAxis if concatAxis < 0 else concatAxis
        concatInDatas = {}
        concatOutData = np.array([])
        for concatInput in inputsList:
            inputShape = get_shape_by_name(onnx_model, concatInput)
            inputData = np.array(np.random.random(size=tuple(inputShape)), dtype=np.float32)
            if concatInput != inputsList[0]:
                while True:
                    trueList = [cInData for cInData in list(concatInDatas.values()) if (cInData==inputData).all()]
                    if trueList:
                        inputData = np.array(np.random.random(size=tuple(inputShape)), dtype=np.float32)
                    else:
                        break
            concatInDatas[concatInput] = inputData
            concatOutData = np.concatenate((concatOutData, inputData), axis=concatAxis) if concatInput != inputsList[0] else inputData
        sIdSpondConcat = []
        for sliceNode in outSliceNodes:
            sliceStart = get_tensor_from_initializer(onnx_model, sliceNode.input[1])
            sliceEnd = get_tensor_from_initializer(onnx_model, sliceNode.input[2])
            sliceAxes = get_tensor_from_initializer(onnx_model, sliceNode.input[3])
            sliceStep = get_tensor_from_initializer(onnx_model, sliceNode.input[4])
            sliceAxesPos = len(concatOutShape) + int(sliceAxes) if int(sliceAxes) < 0 else int(sliceAxes)
            if sliceStep != 1 or sliceStart.size != 1 or sliceEnd.size != 1 or sliceAxes.size != 1 or sliceAxesPos == 0:
                return onnx_model, False
            if sliceAxesPos != concatAxis:
                return onnx_model, False
            if sliceAxesPos == 1:
                sliceData = concatOutData[:, int(sliceStart):int(sliceEnd), :, :]
            elif sliceAxesPos == 2:
                sliceData = concatOutData[:, :, int(sliceStart):int(sliceEnd), :]
            else:
                sliceData = concatOutData[:, :, :, int(sliceStart):int(sliceEnd)]
            sIdFromConcat = [id for id, cInData in enumerate(list(concatInDatas.values())) if (cInData==sliceData).all()]
            if not sIdFromConcat:
                return onnx_model, False
            else:
                sIdFromConcat = sIdFromConcat[0]
            sIdSpondConcat.append(sIdFromConcat)
            
        for vid, sliceNode in enumerate(outSliceNodes):
            concatInShape = get_shape_by_name(onnx_model, inputsList[sIdSpondConcat[vid]])
            sliceInShape = get_shape_by_name(onnx_model, sliceNode.output[0])
            if sliceInShape != concatInShape:
                return onnx_model, False

        for vid, sliceNode in enumerate(outSliceNodes):
            sliceOutNodes = get_node_by_input(onnx_model, sliceNode.output)
            for sliceOutNode in sliceOutNodes:
                for sOId, sliceOutNodeInput in enumerate(sliceOutNode.input):
                    sliceOutNode.input[sOId] = inputsList[sIdSpondConcat[vid]] if sliceOutNodeInput == sliceNode.output[0] else sliceOutNodeInput
            onnx_model = delete_value_info_by_name(onnx_model, sliceNode.output[0])
        onnx_model = delete_nodes(onnx_model, outSliceNodes)
        concatOutNodes = get_node_by_input(onnx_model, node.output)
        if not concatOutNodes:
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            onnx_model.graph.node.remove(node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    else:
        return onnx_model, False
    
@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeReshapeTransposeReshapeTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Transpose', 'Reshape', 'Transpose']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Transpose', 'Reshape', 'Transpose'])
        topTPNode, topRSNode, midTPNode, botRSNode, botTPNode = nodes_serial
        topTPInShape = get_shape_by_name(onnx_model, topTPNode.input[0])
        if topTPInShape[0] == 1:
            topTPPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topTPInShape))).reverse())
            if topTPPerm != [0, 2, 3, 1]:
                return onnx_model, False
            topRSShape = get_shape_by_name(onnx_model, topRSNode.output[0])
            if len(topRSShape) != 6:
                return onnx_model, False
            topTPOutShape = get_shape_by_name(onnx_model, topTPNode.output[0])
            if topTPOutShape[1] != topRSShape[1] * topRSShape[2] or topTPOutShape[2] != topRSShape[3] * topRSShape[4]:
                return onnx_model, False
            midTPPerm = attribute_to_dict(midTPNode.attribute).get('perm', list(range(len(topRSShape))).reverse())
            if midTPPerm != [0, 1, 3, 2, 4, 5]:
                return onnx_model, False
            midTPOutShape = get_shape_by_name(onnx_model, midTPNode.output[0])
            botRSShape = get_shape_by_name(onnx_model, botRSNode.output[0])
            if len(botRSShape) != 4 or midTPOutShape[-3:] != botRSShape[-3:]:
                return onnx_model, False
            botTPPerm = attribute_to_dict(botTPNode.attribute).get('perm', list(range(len(botRSShape))).reverse())
            if botTPPerm != [0, 3, 1, 2]:
                return onnx_model, False
            sliceStartTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_sliceloc0',
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[0])
            sliceStepTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_sliceStep',
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[1])
            sliceAxesLTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_sliceAxesL',
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[2])
            sliceAxesRTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_sliceAxesR',
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[3])
            onnx_model.graph.initializer.append(sliceStartTensor)
            onnx_model.graph.initializer.append(sliceStepTensor)
            onnx_model.graph.initializer.append(sliceAxesLTensor)
            onnx_model.graph.initializer.append(sliceAxesRTensor)
            newSliceStartLTensor = sliceStartTensor
            newSliceStartL = 0
            concatInputs = []
            newNodesList = []
            for sliceNumL in range(topRSShape[1]):
                newSliceEndL = newSliceStartL + topRSShape[2]
                newSliceEndLTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_sliceloc_l%d'%sliceNumL,
                                                            data_type=TensorProto.INT64,
                                                            dims=[1],
                                                            vals=[newSliceEndL])
                newSliceStartRTensorName = sliceStartTensor.name
                newSliceStartR = 0
                for sliceNumR in range(topRSShape[3]):
                    newSliceEndR = newSliceStartR + topRSShape[4]
                    newSliceEndRTensorName = topTPNode.input[0]+'_sliceloc_r%d'%sliceNumR
                    if not find_init_by_name(onnx_model, newSliceEndRTensorName):
                        newSliceEndRTensor = onnx.helper.make_tensor(name=newSliceEndRTensorName,
                                                                    data_type=TensorProto.INT64,
                                                                    dims=[1],
                                                                    vals=[newSliceEndR])
                        onnx_model.graph.initializer.append(newSliceEndRTensor)
                    newSliceLInputs = [topTPNode.input[0], newSliceStartLTensor.name, newSliceEndLTensor.name, sliceAxesLTensor.name, sliceStepTensor.name]
                    newSliceLNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'_slice_lt_%d'%sliceNumL+'_%d'%sliceNumR,
                                                        op_type='Slice',
                                                        inputs=newSliceLInputs,
                                                        outputs=[topTPNode.output[0]+'_slice_lt_%d'%sliceNumL+'_%d'%sliceNumR])
                    newSliceRInputs = [newSliceLNode.output[0], newSliceStartRTensorName, newSliceEndRTensorName, sliceAxesRTensor.name, sliceStepTensor.name]
                    newSliceRNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'slice_rt_%d'%sliceNumL+'_%d'%sliceNumR,
                                                        op_type='Slice',
                                                        inputs=newSliceRInputs,
                                                        outputs=[topTPNode.output[0]+'_slice_rt_%d'%sliceNumL+'_%d'%sliceNumR])
                    concatInputs.append(newSliceRNode.output[0])
                    newNodesList.append(newSliceLNode)
                    newNodesList.append(newSliceRNode)
                    newSliceStartRTensorName = newSliceEndRTensorName
                    newSliceStartR = newSliceEndR
                newSliceStartLTensor = newSliceEndLTensor
                newSliceStartL = newSliceEndL
                onnx_model.graph.initializer.append(newSliceEndLTensor)
            newNodesList.reverse()
            newConcatNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'_concat',
                                                op_type='Concat',
                                                inputs=concatInputs,
                                                outputs=[botTPNode.output[0]+'_concat'],
                                                axis=1)
            botTPOutShape = get_shape_by_name(onnx_model, botTPNode.output[0])
            newShapeTensor = onnx.helper.make_tensor(name=botTPNode.output[0]+'_shape',
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(botTPOutShape)],
                                                    vals=botTPOutShape)
            onnx_model.graph.initializer.append(newShapeTensor)
            newReshapeNode = onnx.helper.make_node(name=botTPNode.name+'_toReshape',
                                                op_type='Reshape',
                                                inputs=[newConcatNode.output[0], newShapeTensor.name],
                                                outputs=botTPNode.output)
            onnx_model.graph.node.insert(node_index, newReshapeNode)
            onnx_model.graph.node.insert(node_index, newConcatNode)
            onnx_model = insert_node_by_list(onnx_model, newNodesList, node_index)
            nodes_serial.reverse()
            for sid, src_node in enumerate(nodes_serial):
                if sid < 1:
                    onnx_model.graph.node.remove(src_node)
                else:
                    srcOutNodes = get_node_by_input(onnx_model, src_node.output)
                    if len(srcOutNodes) > 0:
                        break
                    else:
                        onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
                        onnx_model.graph.node.remove(src_node)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        elif topTPInShape[0] > 1:
            topTPPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topTPInShape))).reverse())
            if topTPPerm != [0, 2, 3, 1]:
                return onnx_model, False
            topTPOutShape = get_shape_by_name(onnx_model, topTPNode.output[0])
            topRSShape = get_shape_by_name(onnx_model, topRSNode.output[0]) 
            if len(topRSShape) != 6 or topRSShape[1] * topRSShape[2] != topTPOutShape[0] or topTPOutShape[1:] != topRSShape[-3:]:
                return onnx_model, False
            midTPPerm = attribute_to_dict(midTPNode.attribute).get('perm', list(range(len(topRSShape))).reverse())
            if midTPPerm != [0, 1, 3, 2, 4, 5]:
                return onnx_model, False
            midTPOutShape = get_shape_by_name(onnx_model, midTPNode.output[0])
            botRSShape = get_shape_by_name(onnx_model, botRSNode.output[0])
            if len(botRSShape) != 4 or botRSShape[1] != midTPOutShape[1] * midTPOutShape[2] or botRSShape[2] != midTPOutShape[3] * midTPOutShape[4]:
                return onnx_model, False
            botTPPerm = attribute_to_dict(botTPNode.attribute).get('perm', list(range(len(botRSShape))).reverse())
            if botTPPerm != [0, 3, 1, 2]:
                return onnx_model, False
            newRSShape = [1, topTPInShape[0] * topTPInShape[1], topTPInShape[2], topTPInShape[3]]
            newRSShapeTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_newShape',
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(newRSShape)],
                                                       vals=newRSShape)
            onnx_model.graph.initializer.append(newRSShapeTensor)
            newRSNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'_reshape',
                                              op_type='Reshape',
                                              inputs=[topTPNode.input[0], newRSShapeTensor.name],
                                              outputs=[topTPNode.input[0]+'_newShapeOut'])
            sliceStepTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_slice_step',
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[1])
            onnx_model.graph.initializer.append(sliceStepTensor)
            sliceAxesTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_slice_axes',
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[1])
            onnx_model.graph.initializer.append(sliceAxesTensor)
            sliceStart = 0
            sliceStartTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_slice_loc0',
                                                       data_type=TensorProto.INT64,
                                                       dims=[1],
                                                       vals=[sliceStart])
            onnx_model.graph.initializer.append(sliceStartTensor)
            sliceNodesList = []
            for sliceNum in range(topTPInShape[0]):
                sliceEnd = sliceStart + topTPInShape[1]
                sliceEndTensor = onnx.helper.make_tensor(name=topTPNode.input[0]+'_slice_loc%d'%(sliceNum+1),
                                                         data_type=TensorProto.INT64,
                                                         dims=[1],
                                                         vals=[sliceEnd])
                onnx_model.graph.initializer.append(sliceEndTensor)
                sliceInputs = [newRSNode.output[0], sliceStartTensor.name, sliceEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
                newSliceNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'_slice%d'%sliceNum,
                                                     op_type='Slice',
                                                     inputs=sliceInputs,
                                                     outputs=[topTPNode.input[0]+'_slice%d'%sliceNum])
                sliceNodesList.append(newSliceNode)
                sliceStart = sliceEnd
                sliceStartTensor = sliceEndTensor
            lastConcatInputs = []
            axis3ConcatNodes = []
            indexStart = 0
            for axis3Num in range(topRSShape[1]):
                indexEnd = indexStart + topRSShape[2]
                concatInputNodes = sliceNodesList[indexStart:indexEnd]
                concatInputs = [concatInputNode.output[0] for concatInputNode in concatInputNodes]
                newConcatNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'_axis3Concat%d'%axis3Num,
                                                      op_type='Concat',
                                                      inputs=concatInputs,
                                                      outputs=[botTPNode.output[0]+'_axis3concat%d'%axis3Num],
                                                      axis=3)
                lastConcatInputs.append(newConcatNode.output[0])
                axis3ConcatNodes.append(newConcatNode)
                indexStart=indexEnd
            lastConcatNode = onnx.helper.make_node(name=topTPNode.name+'_to_'+botTPNode.name+'_axis2concat',
                                                   op_type='Concat',
                                                   inputs=lastConcatInputs,
                                                   outputs=botTPNode.output,
                                                   axis=2)
            onnx_model.graph.node.insert(node_index, lastConcatNode)
            axis3ConcatNodes.reverse()
            onnx_model = insert_node_by_list(onnx_model, axis3ConcatNodes, node_index)
            sliceNodesList.reverse()
            onnx_model = insert_node_by_list(onnx_model, sliceNodesList, node_index)
            onnx_model.graph.node.insert(node_index, newRSNode)
            nodes_serial.reverse()
            for sid, src_node in enumerate(nodes_serial):
                if sid < 1:
                    onnx_model.graph.node.remove(src_node)
                else:
                    srcOutNodes = get_node_by_input(onnx_model, src_node.output)
                    if len(srcOutNodes) > 0:
                        break
                    else:
                        onnx_model = delete_value_info_by_name(onnx_model, src_node.output[0])
                        onnx_model.graph.node.remove(src_node)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMultiBatchConvToOneBatch(onnx_model, node, node_index):
    def check_convert_flag(node):
        conv_attr = attribute_to_dict(node.attribute)
        dilations = conv_attr.get('dilations', [1, 1])
        group = conv_attr.get('group', 1)
        kernel_shape = conv_attr.get('kernel_shape', [1, 1])
        pads = conv_attr.get('pads', [0, 0, 0, 0])
        strides = conv_attr.get('strides', [1, 1])
        if dilations == [1, 1] and group == 1 and kernel_shape == [1, 1] and pads == [0, 0, 0, 0] and strides == [1, 1]:
            return True
        else:
            return False
        
    if node.op_type == 'Conv':
        convInShape = get_shape_by_name(onnx_model, node.input[0])
        convOutShape = get_shape_by_name(onnx_model, node.output[0])
        if convInShape[0] == 1 or len(convInShape) != 4:
            return onnx_model, False
        if not find_init_by_name(onnx_model, node.input[1]):
            return onnx_model, False
        weightArr = get_tensor_from_initializer(onnx_model, node.input[1])
        biasArr = None
        if len(node.input) == 3:
            biasArr = get_tensor_from_initializer(onnx_model, node.input[2])
        convOutNodes = get_node_by_input(onnx_model, node.output)
        convInNode = get_node_by_output(onnx_model, node.input[0])
        while True:
            if convInNode.op_type not in ['Sigmoid', 'Tanh', 'Relu', 'Elu', 'PRelu', 'LeakyRelu'] or convInNode is None:
                break
            convInNode = get_node_by_output(onnx_model, convInNode.input[0])
        splitOutNodes = [outNode for outNode in convOutNodes if outNode.op_type in ['Split', 'Slice']]
        topRSShape = [1, convInShape[0] * convInShape[1], convInShape[2], convInShape[3]]
        topRSShapeTensor = get_initial_by_value(onnx_model, np.array(topRSShape, dtype=np.int64))
        if topRSShapeTensor is None:
            topRSShapeTensor = onnx.helper.make_tensor(name=node.input[0]+'_newShape',
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(topRSShape)],
                                                    vals=topRSShape)
            onnx_model.graph.initializer.append(topRSShapeTensor)
        topRSNode = onnx.helper.make_node(name=node.input[0]+'_reshape',
                                        op_type='Reshape',
                                        inputs=[node.input[0], topRSShapeTensor.name],
                                        outputs=[node.input[0]+'_newShapeOut'])
        insertNodes = [topRSNode]
        newBotNode = None
        if check_convert_flag(node) and len(splitOutNodes) != len(convOutNodes) and convInNode.op_type != 'Concat':
        #if check_convert_flag(node):
            newWeightArr = np.zeros((convInShape[0] * weightArr.shape[0], convInShape[0] * weightArr.shape[1]), dtype=np.float32)
            newBiasArr = None
            if biasArr is not None:
                newBiasArr = np.zeros((convInShape[0]*biasArr.shape[0],), dtype=np.float32)
            for kn in range(convInShape[0]):
                indexW = kn * weightArr.shape[1]
                indexH = kn * weightArr.shape[0]
                newWeightArr[indexH:(indexH + weightArr.shape[0]), indexW:(indexW + weightArr.shape[1])] \
                    = weightArr.reshape(weightArr.shape[0], weightArr.shape[1])
                if newBiasArr is not None:
                    newBiasArr[indexH:(indexH + biasArr.shape[0])] = biasArr
            newWeightArr = np.reshape(newWeightArr, (newWeightArr.shape[0], newWeightArr.shape[1], 1, 1))
            newWeightTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, node.input[1]+'_new'),
                                                    data_type=NPDTYPE_2_ONNXDTYPE[newWeightArr.dtype],
                                                    dims=newWeightArr.shape,
                                                    vals=newWeightArr.flatten().tolist())
            onnx_model.graph.initializer.append(newWeightTensor)
            newConvInputs = [topRSNode.output[0], newWeightTensor.name]
            if newBiasArr is not None:
                newBiasTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, node.input[2]+'_new'),
                                                        data_type=NPDTYPE_2_ONNXDTYPE[newBiasArr.dtype],
                                                        dims=newBiasArr.shape,
                                                        vals=newBiasArr.tolist())
                onnx_model.graph.initializer.append(newBiasTensor)
                newConvInputs.append(newBiasTensor.name)
            newConvNode = onnx.helper.make_node(name=node.name+'_new',
                                                op_type='Conv',
                                                inputs=newConvInputs,
                                                outputs=[node.output[0]+'_newConv'],
                                                **attribute_to_dict(node.attribute))
            insertNodes.append(newConvNode)
            newBotNode = newConvNode
        else:
            sliceStart = 0
            sliceStartTensor = get_initial_by_value(onnx_model, np.array(sliceStart, dtype=np.int64))
            if sliceStartTensor is None:
                sliceStartTensor = onnx.helper.make_tensor(name=node.input[0]+'_slice_loc0',
                                                           data_type=TensorProto.INT64,
                                                           dims=[1],
                                                           vals=[sliceStart])
                onnx_model.graph.initializer.append(sliceStartTensor)
            sliceAxesTensor = get_initial_by_value(onnx_model, np.array(1, dtype=np.int64))
            if sliceAxesTensor is None:
                sliceAxesTensor = onnx.helper.make_tensor(name=node.input[0]+'_slice_param',
                                                          data_type=TensorProto.INT64,
                                                          dims=[1],
                                                          vals=[1])
                onnx_model.graph.initializer.append(sliceAxesTensor)
            sliceStepTensor = sliceAxesTensor
            sliceNodesList = []
            newConvNodesList = []
            newConcatInputs = []
            for sId in range(convInShape[0]):
                sliceEnd = sliceStart + convInShape[1]
                sliceEndTensor = get_initial_by_value(onnx_model, np.array(sliceEnd, dtype=np.int64))
                if sliceEndTensor is None:
                    sliceEndTensor = onnx.helper.make_tensor(name=node.input[0]+'_slice_loc%d'%(sId+1),
                                                             data_type=TensorProto.INT64,
                                                             dims=[1],
                                                             vals=[sliceEnd])
                    onnx_model.graph.initializer.append(sliceEndTensor)
                sliceInputs = [topRSNode.output[0], sliceStartTensor.name, sliceEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
                sliceNode = onnx.helper.make_node(name=node.name+'_slice_%d'%sId,
                                                  op_type='Slice',
                                                  inputs=sliceInputs,
                                                  outputs=[node.input[0]+'_sliceOut_%d'%sId])
                sliceNodesList.append(sliceNode)
                newConvNode = copy.deepcopy(node)
                newConvNode.name = node.name+'_new%d'%sId
                newConvNode.input[0] = sliceNode.output[0]
                newConvNode.output[0] = node.output[0]+'_new%d'%sId
                newConvNodesList.append(newConvNode)
                newConcatInputs.append(newConvNode.output[0])
                sliceStart = sliceEnd
                sliceStartTensor = sliceEndTensor
            newConcatNode = onnx.helper.make_node(name=node.name+'_concat',
                                                  op_type='Concat',
                                                  inputs=newConcatInputs,
                                                  outputs=[node.output[0]+'_concatOut'],
                                                  axis=1)
            insertNodes += sliceNodesList
            insertNodes += newConvNodesList
            insertNodes.append(newConcatNode)
            newBotNode = newConcatNode
        if not (len(convOutNodes) == 1 and convOutNodes[0].op_type == 'Reshape'):
            botRSShapeTensor = get_initial_by_value(onnx_model, np.array(convOutShape, dtype=np.int64))
            if botRSShapeTensor is None:
                botRSShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, node.output[0]+'_newShape'),
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(convOutShape)],
                                                    vals=convOutShape)
                onnx_model.graph.initializer.append(botRSShapeTensor)
            botRSNode = onnx.helper.make_node(name=node.output[0]+'_reshape',
                                            op_type='Reshape',
                                            inputs=[newBotNode.output[0], botRSShapeTensor.name],
                                            outputs=node.output)
            insertNodes.append(botRSNode)
        else:
            newBotNode.output[0] = node.output[0]
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            newOutShape = [1, convOutShape[0]*convOutShape[1], convOutShape[2], convOutShape[3]]
            convOutNodeOutShape = get_shape_by_name(onnx_model, convOutNodes[0].output[0])
            if newOutShape == convOutNodeOutShape:
                newBotNode.output[0] = convOutNodes[0].output[0]
                onnx_model.graph.node.remove(convOutNodes[0])
            else:
                newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, tuple(newOutShape))
                onnx_model.graph.value_info.append(newValueInfo)
        insertNodes.reverse()
        onnx_model = insert_node_by_list(onnx_model, insertNodes, node_index)
        onnx_model.graph.node.remove(node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True    
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMultiBatchSplit2OneBatchSliceConcat(onnx_model, node, node_index):
    if node.op_type == 'Split':
        inShape = get_shape_by_name(onnx_model, node.input[0])
        if inShape[0] <= 1:
            return onnx_model, False
        sliceInput = node.input[0]
        topSliceNum = inShape[0]
        splitOutNum = len(node.output)
        sliceAllNum = topSliceNum * splitOutNum
        splitAxis = attribute_to_dict(node.attribute).get('axis', 0)
        splitAxis = splitAxis if splitAxis >= 0 else len(inShape)+splitAxis
        if splitAxis != 1:
            return onnx_model, False
        opset_version = get_opset_version(onnx_model)
        if opset_version <= 12:
            splitNumArr = attribute_to_dict(node.attribute).get('split')
            if splitNumArr is None:
                return onnx_model, False
        else:
            splitNumArr = get_tensor_from_initializer(onnx_model, node.input[1]).tolist()
        sliceNumList = []
        for tn in range(topSliceNum):
            sliceNumList += splitNumArr
        if np.sum(np.array(sliceNumList, dtype=np.int64)) != inShape[0]*inShape[1]:
            return onnx_model, False
        newInShape = [1, inShape[0]*inShape[1]]
        if len(inShape) > 2:
            newInShape = newInShape + inShape[2:]
        splitInNode = get_node_by_output(onnx_model, node.input[0])
        splitParaNodes = get_node_by_input(onnx_model, splitInNode.output)
        if not (splitInNode.op_type == 'Reshape' and len(splitParaNodes) == 1):
            newInShapeTensor = get_initial_by_value(onnx_model, np.array(newInShape, dtype=np.int64))
            if newInShapeTensor is None:
                newInShapeTensor = onnx.helper.make_tensor(name=node.input[0]+'_newShape',
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(newInShape)],
                                                           vals=newInShape)
                onnx_model.graph.initializer.append(newInShapeTensor)
            newReshapeNode = onnx.helper.make_node(name=node.input[0]+'_reshape',
                                                   op_type='Reshape',
                                                   inputs=[node.input[0], newInShapeTensor.name],
                                                   outputs=[node.input[0]+'_newOut'])
            onnx_model.graph.node.insert(node_index, newReshapeNode)
            node_index += 1
            sliceInput = newReshapeNode.output[0]
        else:
            splitInNodeInShape = get_shape_by_name(onnx_model, splitInNode.input[0])
            if splitInNodeInShape == newInShape:
                onnx_model.graph.node.remove(splitInNode)
                node_index -= 1
                sliceInput = splitInNode.input[0]
            else:
                onnx_model = delete_value_info_by_name(onnx_model, splitInNode.output[0])
                newInShapeTensor = get_initial_by_value(onnx_model, np.array(newInShape, dtype=np.int64))
                if newInShapeTensor is None:
                    newInShapeTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, splitInNode.input[0]+'_new'),
                                                             data_type=TensorProto.INT64,
                                                             dims=[len(newInShape)],
                                                             vals=newInShape)
                    onnx_model.graph.initializer.append(newInShapeTensor)
                newInShapeValInfo = onnx.helper.make_tensor_value_info(node.input[0], 1, newInShape)
                onnx_model.graph.value_info.append(newInShapeValInfo)
        sliceStart = 0
        sliceStartTensor = get_initial_by_value(onnx_model, np.array(sliceStart, dtype=np.int64))
        if sliceStartTensor is None:
            sliceStartTensor = onnx.helper.make_tensor(name=sliceInput+'_slice_loc0',
                                                       data_type=TensorProto.INT64,
                                                       dims=[1],
                                                       vals=[sliceStart])
            onnx_model.graph.initializer.append(sliceStartTensor)
        sliceAxesTensor = get_initial_by_value(onnx_model, np.array(1, dtype=np.int64))
        if sliceAxesTensor is None:
            sliceAxesTensor = onnx.helper.make_tensor(name=sliceInput+'_slice_param',
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[1])
            onnx_model.graph.initializer.append(sliceAxesTensor)
        sliceStepTensor = sliceAxesTensor
        sliceNodeList = []
        for sId in range(sliceAllNum):
            sliceLen = sliceNumList[sId]
            sliceEnd = sliceStart + sliceLen
            sliceEndTensor = get_initial_by_value(onnx_model, np.array(sliceEnd, dtype=np.int64))
            if sliceEndTensor is None:
                sliceEndTensor = onnx.helper.make_tensor(name=sliceInput+'_slice_loc%d'%(sId+1),
                                                         data_type=TensorProto.INT64,
                                                         dims=[1],
                                                         vals=[sliceEnd])
                onnx_model.graph.initializer.append(sliceEndTensor)
            sliceNodeInputs = [sliceInput, sliceStartTensor.name, sliceEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
            newSliceNode = onnx.helper.make_node(name=node.name+'_toSlice_%d'%sId,
                                                 op_type='Slice',
                                                 inputs=sliceNodeInputs,
                                                 outputs=[node.name+'_sliceOut_%d'%sId])
            sliceNodeList.append(newSliceNode)
            sliceStart = sliceEnd
            sliceStartTensor = sliceEndTensor
        splitOutShapeList = [get_shape_by_name(onnx_model, splitOutName) for splitOutName in list(node.output)]
        newConcatList = []
        newOutReshapeList = []
        for cId in range(splitOutNum):
            concatInIds = [cId + splitOutNum * tId for tId in range(topSliceNum)]
            concatInputs = [sliceNodeList[vid].output[0] for vid in concatInIds]
            newConcatNode = onnx.helper.make_node(name=node.name+'_toConcat_%d'%cId,
                                                  op_type='Concat',
                                                  inputs=concatInputs,
                                                  outputs=[node.output[cId]+'_newOut'],
                                                  axis=1)
            newConcatOutShape = [1, splitOutShapeList[cId][0]*splitOutShapeList[cId][1]]
            if len(splitOutShapeList[cId]) > 2:
                newConcatOutShape += splitOutShapeList[cId][2:]
            splitOutNodes = get_node_by_input(onnx_model, [node.output[cId]])
            if not (splitOutNodes[0].op_type == 'Reshape' and len(splitOutNodes) == 1):
                newOutRSShapeTensor = get_initial_by_value(onnx_model, np.array(splitOutShapeList[cId], dtype=np.int64))
                if newOutRSShapeTensor is None:
                    newOutRSShapeTensor = onnx.helper.make_tensor(name=node.output[cId]+'_shape',
                                                                      data_type=TensorProto.INT64,
                                                                      dims=[len(splitOutShapeList[cId])],
                                                                      vals=splitOutShapeList[cId])
                    onnx_model.graph.initializer.append(newOutRSShapeTensor)
                newOutRSNode = onnx.helper.make_node(name=node.name+'_toReshape_%d'%cId,
                                                     op_type='Reshape',
                                                     inputs=[newConcatNode.output[0], newOutRSShapeTensor.name],
                                                     outputs=[node.output[cId]])
                newOutReshapeList.append(newOutRSNode)
            else:
                onnx_model = delete_value_info_by_name(onnx_model, node.output[cId])
                botRSOutShape = get_shape_by_name(onnx_model, splitOutNodes[0].output[0])
                if botRSOutShape == newConcatOutShape:
                    newConcatNode.output[0] = splitOutNodes[0].output[0]
                    onnx_model.graph.node.remove(splitOutNodes[0])
                else:
                    newConcatNode.output[0] = node.output[cId]
                    newOutValInfo = onnx.helper.make_tensor_value_info(node.output[cId], 1, newConcatOutShape)
                    onnx_model.graph.value_info.append(newOutValInfo)
            newConcatList.append(newConcatNode)
        onnx_model.graph.node.remove(node)
        newOutReshapeList.reverse()
        onnx_model = insert_node_by_list(onnx_model, newOutReshapeList, node_index)
        newConcatList.reverse()
        onnx_model = insert_node_by_list(onnx_model, newConcatList, node_index)
        sliceNodeList.reverse()
        onnx_model = insert_node_by_list(onnx_model, sliceNodeList, node_index)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMultiBatchAddReshapeToOneBatchReshapeAdd(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Add', 'Reshape']):
        addNode, reshapeNode = get_node_serial_group(onnx_model, node, ['Add', 'Reshape'])
        if find_init_by_name(onnx_model, addNode.input[0]) or find_init_by_name(onnx_model, addNode.input[1]):
            return onnx_model, False
        addOutShape = get_shape_by_name(onnx_model, addNode.output[0])
        if addOutShape[0] <= 1:
            return onnx_model, False
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        addInShape0 = get_shape_by_name(onnx_model, addNode.input[0])
        addInShape1 = get_shape_by_name(onnx_model, addNode.input[1])
        addInNode0 = get_node_by_output(onnx_model, addNode.input[0])
        addInNode1 = get_node_by_output(onnx_model, addNode.input[1])
        if addInShape0 != addInShape1 or addInNode0 == addInNode1:
            return onnx_model, False
        newShapeTensor = get_initial_by_value(onnx_model, np.array(reshapeOutShape, dtype=np.int64))
        if newShapeTensor is None:
            newShapeTensor = onnx.helper.make_tensor(name=addNode.input[0]+'_'+addNode.input[1]+'_newShape',
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(reshapeOutShape)],
                                                    vals=reshapeOutShape)
            onnx_model.graph.initializer.append(newShapeTensor)
        newReshapeNodesList = []
        for idx, addInNode in enumerate([addInNode0, addInNode1]):
            if addInNode.op_type == 'Reshape':
                addInNodeInShape = get_shape_by_name(onnx_model, addInNode.input[0])
                if addInNodeInShape == reshapeOutShape:
                    addNode.input[idx] = addInNode.input[0]
                    addParaNodes = get_node_by_input(onnx_model, addInNode.output)
                    if not addParaNodes:
                        onnx_model = delete_value_info_by_name(onnx_model, addInNode.output[0])
                        onnx_model.graph.node.remove(addInNode)
                else:
                    addParaNodes = get_node_by_input(onnx_model, addInNode.output)
                    if len(addParaNodes) > 1:
                        newReshapeNode = onnx.helper.make_node(name=addNode.input[idx]+'_newReshape',
                                                                op_type='Reshape',
                                                                inputs=[addNode.input[idx], newShapeTensor.name],
                                                                outputs=[addNode.input[idx]+'_newOut'])
                        addNode.input[idx] = newReshapeNode.output[0]
                        newReshapeNodesList.append(newReshapeNode)
                    else:
                        addInNode.input[1] = newShapeTensor.name
                        onnx_model = delete_value_info_by_name(onnx_model, addNode.input[idx])
            else:
                newReshapeNode = onnx.helper.make_node(name=addNode.input[idx]+'_newReshape',
                                                        op_type='Reshape',
                                                        inputs=[addNode.input[idx], newShapeTensor.name],
                                                        outputs=[addNode.input[idx]+'_newOut'])
                addNode.input[idx] = newReshapeNode.output[0]
                newReshapeNodesList.append(newReshapeNode)
        if newReshapeNodesList:
            newReshapeNodesList.reverse()
            onnx_model = insert_node_by_list(onnx_model, newReshapeNodesList, node_index)
        onnx_model = delete_value_info_by_name(onnx_model, addNode.output[0])
        addNode.output[0] = reshapeNode.output[0]
        onnx_model.graph.node.remove(reshapeNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMultiBatchReshapeConcatReshapeToOneBatchSliceConcat(onnx_model, node, node_index):
    if node.op_type == 'Concat':
        outShape = get_shape_by_name(onnx_model, node.output[0])
        concatAxis = attribute_to_dict(node.attribute).get('axis', 1)
        concatAxisPos = concatAxis if concatAxis >= 0 else len(outShape) + concatAxis
        if concatAxisPos != 1 or outShape[0] == 1:
            return onnx_model, False
        activation_ops = ['Sigmoid', 'Tanh', 'Relu', 'Elu', 'PRelu', 'LeakyRelu']
        actNode = None
        outRSNode = None
        outNodes = get_node_by_input(onnx_model, node.output)
        if len(outNodes) != 1:
            return onnx_model, False
        if outNodes[0].op_type == 'Reshape':
            outRSNode = outNodes[0]
        elif outNodes[0].op_type in activation_ops:
            actNode = outNodes[0]
            actOutNodes = get_node_by_input(onnx_model, actNode.output)
            if len(actOutNodes) != 1 or actOutNodes[0].op_type != 'Reshape':
                return onnx_model, False
            else:
                outRSNode = actOutNodes[0]
        else:
            return onnx_model, False
        inNodes = [get_node_by_output(onnx_model, inname) for inname in node.input]
        inRSNodesList = [inNode for inNode in inNodes if inNode.op_type == 'Reshape']
        if inNodes != inRSNodesList:
            return onnx_model, False
        for inRSNode in inRSNodesList:
            inRSInShape = get_shape_by_name(onnx_model, inRSNode.input[0])
            inRSOutShape = get_shape_by_name(onnx_model, inRSNode.output[0])
            cmpInRSShape = [1, inRSOutShape[0]*inRSOutShape[1]] + (inRSOutShape[2:] if len(inRSOutShape) > 2 else []) 
            if cmpInRSShape != inRSInShape:
                return onnx_model, False
        outRSOutShape = get_shape_by_name(onnx_model, outRSNode.output[0])
        cmpRSShape = [1, outShape[0]*outShape[1]]
        if len(outShape) > 2:
            cmpRSShape += outShape[2:]
        if cmpRSShape != outRSOutShape:
            return onnx_model, False
        sliceAllNum = outShape[0] * len(inNodes)
        sliceNodesList = []
        sliceTensorList = []
        sliceStart = 0
        sliceStartTensor = get_initial_by_value(onnx_model, np.array(sliceStart, dtype=np.int64))
        if sliceStartTensor is None:
            sliceStartTensor = onnx.helper.make_tensor(name=node.name+'_newslice_loc',
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[sliceStart])
            sliceTensorList.append(sliceStartTensor)
        sliceAxes = 1
        sliceAxesTensor = get_initial_by_value(onnx_model, np.array(sliceAxes, dtype=np.int64))
        if sliceAxesTensor is None:
            sliceAxesTensor = onnx.helper.make_tensor(name=node.name+'_newslice_param',
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[sliceAxes])
            sliceTensorList.append(sliceAxesTensor)
        sliceStepTensor = sliceAxesTensor
        for idx, inRSNode in enumerate(inRSNodesList):
            inRSOutShape = get_shape_by_name(onnx_model, inRSNode.output[0])
            curStart = sliceStart
            curStartTensor = sliceStartTensor
            for sNum in range(outShape[0]):
                curEnd = curStart + inRSOutShape[1]
                curEndTensor = get_initial_by_value(onnx_model, np.array(curEnd, dtype=np.int64))
                if curEndTensor is None:
                    curEndTensor = onnx.helper.make_tensor(name=node.name+'_slice_loc_%d'%idx +'_%d'%sNum,
                                                           data_type=TensorProto.INT64,
                                                           dims=[1],
                                                           vals=[curEnd])
                    sliceTensorList.append(curEndTensor)
                sliceInputs = [inRSNode.input[0], curStartTensor.name, curEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
                sliceNode = onnx.helper.make_node(name=inRSNode.name+'_toSlice%d'%sNum,
                                                  op_type='Slice',
                                                  inputs=sliceInputs,
                                                  outputs=[inRSNode.output[0]+'_%d'%sNum])
                sliceNodesList.append(sliceNode)
                curStart = curEnd
                curStartTensor = curEndTensor
        if len(sliceNodesList) != sliceAllNum:
            return onnx_model, False
        newConcatInputs = []
        sliceSortNodesList = []
        for sNum in range(outShape[0]):
            concatIds = [sNum + vid * outShape[0] for vid in range(len(inRSNodesList))]
            newConcatInputs += [sliceNodesList[vid].output[0] for vid in concatIds]
            sliceSortNodesList += [sliceNodesList[vid] for vid in concatIds]
        newConcatNode = onnx.helper.make_node(name=node.name,
                                              op_type='Concat',
                                              inputs=newConcatInputs,
                                              outputs=outRSNode.output,
                                              axis=concatAxisPos)
        if actNode is not None:
            onnx_model = delete_value_info_by_name(onnx_model, actNode.input[0])
            newConcatNode.output[0] = node.output[0]
            actNode.output[0] = outRSNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
        for inRSNode in inRSNodesList:
            onnx_model = delete_value_info_by_name(onnx_model, inRSNode.output[0])
        onnx_model.graph.node.remove(node)
        onnx_model.graph.node.insert(node_index, newConcatNode)
        sliceSortNodesList.reverse()
        onnx_model = insert_node_by_list(onnx_model, sliceSortNodesList, node_index)
        onnx_model = delete_nodes(onnx_model, inRSNodesList + [outRSNode])
        for sliceTensor in sliceTensorList:
            if not find_init_by_name(onnx_model, sliceTensor.name):
                onnx_model.graph.initializer.append(sliceTensor)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMultiBatchReshapeSliceReshapeToOneBatchSliceConcat(onnx_model, node, node_index):
    if node.op_type == 'Reshape':
        inShape = get_shape_by_name(onnx_model, node.input[0])
        outShape = get_shape_by_name(onnx_model, node.output[0])
        try:
            cmpShape = [1, outShape[0]*outShape[1]] + (outShape[2:] if len(outShape) > 2 else [])
        except:
            return onnx_model, False
        if cmpShape != inShape:
            return onnx_model, False
        outNodesList = get_node_by_input(onnx_model, node.output)
        sliceNodesList = [outNode for outNode in outNodesList if outNode.op_type == 'Slice']
        rsNodesList = []
        sliceSpaceList = []
        for sliceNode in sliceNodesList:
            sliceAxes = get_tensor_from_initializer(onnx_model, sliceNode.input[3])
            sliceAxes = list(sliceAxes) if list(sliceAxes.shape) else [sliceAxes]
            if len(sliceAxes) != 1 or sliceAxes[0] != 1:
                return onnx_model, False
            rsNodes = get_node_by_input(onnx_model, sliceNode.output)
            if len(rsNodes) != 1 or rsNodes[0].op_type != 'Reshape':
                return onnx_model, False
            rsInShape = get_shape_by_name(onnx_model, rsNodes[0].input[0])
            rsOutShape = get_shape_by_name(onnx_model, rsNodes[0].output[0])
            cmpInShape = [1, rsInShape[0]*rsInShape[1]] + (rsInShape[2:] if len(rsInShape) > 2 else [])
            if cmpInShape != rsOutShape:
                return onnx_model, False
            rsNodesList.append(rsNodes[0])
            sliceSpaceList += [rsInShape[1]]
        sliceSpaceList *= outShape[0]
        sliceStart = 0
        sliceStartTensor = get_initial_by_value(onnx_model, np.array(sliceStart, dtype=np.int64))
        if sliceStartTensor is None:
            sliceStartTensor = onnx.helper.make_tensor(name=node.input[0]+'_slice_loc0',
                                                       data_type=TensorProto.INT64,
                                                       dims=[1],
                                                       vals=[sliceStart])
            onnx_model.graph.initializer.append(sliceStartTensor)
        sliceAxesTensor = get_initial_by_value(onnx_model, np.array(1, dtype=np.int64))
        if sliceAxesTensor is None:
            sliceAxesTensor = onnx.helper.make_tensor(name=node.input[0]+'_slice_param',
                                                      data_type=TensorProto.INT64,
                                                      dims=[1],
                                                      vals=[1])
            onnx_model.graph.initializer.append(sliceAxesTensor)
        sliceStepTensor = sliceAxesTensor
        newSliceNodesList = []
        for vid, sliceSpace in enumerate(sliceSpaceList):
            sliceEnd = sliceStart + sliceSpace
            sliceEndTensor = get_initial_by_value(onnx_model, np.array(sliceEnd, dtype=np.int64))
            if sliceEndTensor is None:
                sliceEndTensor = onnx.helper.make_tensor(name=node.input[0]+'_slice_loc%d'%(vid+1),
                                                         data_type=TensorProto.INT64,
                                                         dims=[1],
                                                         vals=[sliceEnd])
                onnx_model.graph.initializer.append(sliceEndTensor)
            newSliceInputs = [node.input[0], sliceStartTensor.name, sliceEndTensor.name, sliceAxesTensor.name, sliceStepTensor.name]
            newSliceNode = onnx.helper.make_node(name=node.name+'_toSlice%d'%vid,
                                                 op_type='Slice',
                                                 inputs=newSliceInputs,
                                                 outputs=[node.input[0]+'_sliceOut%d'%vid])
            newSliceNodesList.append(newSliceNode)
            sliceStart = sliceEnd
            sliceStartTensor = sliceEndTensor
        sortSliceNodesList = []
        newConcatNodesList = []
        for cId, rsNode in enumerate(rsNodesList):
            concatIds = [cId + len(rsNodesList) * idx for idx in range(outShape[0])]
            concatInputs = [newSliceNodesList[idx].output[0] for idx in concatIds]
            sortSliceNodesList += [newSliceNodesList[idx] for idx in concatIds]
            newConcatNode = onnx.helper.make_node(name=node.name+'_toConcat%d'%cId,
                                                  op_type='Concat',
                                                  inputs=concatInputs,
                                                  outputs=rsNode.output,
                                                  axis=1)
            newConcatNodesList.append(newConcatNode)
        newConcatNodesList.reverse()
        sortSliceNodesList.reverse()
        onnx_model = insert_node_by_list(onnx_model, newConcatNodesList, node_index)
        onnx_model = insert_node_by_list(onnx_model, sortSliceNodesList, node_index)
        for sliceNode in sliceNodesList:
            onnx_model = delete_value_info_by_name(onnx_model, sliceNode.output[0])
            onnx_model.graph.node.remove(sliceNode)
        onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
        onnx_model = delete_nodes(onnx_model, rsNodesList + [node])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMSliceConcatNxMSlice(onnx_model, node, node_index):
    def get_cosponding_sliceid(onnx_model, inputs, out_nodes, n_axis):
        cosIdDict = {}
        outNodesDict = {}
        for out_node in out_nodes:
            startVal = get_tensor_from_initializer(onnx_model, out_node.input[1])
            if startVal.size > 1:
                return None
            outNodesDict[int(startVal)] = out_node
        sortOutNodesList = sorted(outNodesDict.items())
        sortOutNodes = [dictList[1] for dictList in sortOutNodesList]
        cur_id = -1
        for input_name in inputs:
            inShape = get_shape_by_name(onnx_model, input_name)
            cosIdDict[input_name] = []
            calNum = 0
            for vid, outNode in enumerate(sortOutNodes):
                if vid <= cur_id:
                    continue
                outNodeOutShape = get_shape_by_name(onnx_model, outNode.output[0])
                if outNodeOutShape[n_axis] > inShape[n_axis]:
                    return None
                calNum += outNodeOutShape[n_axis]
                if calNum > inShape[n_axis]:
                    return None
                cosIdDict[input_name].append(vid)
                if calNum == inShape[n_axis]:
                    cur_id = vid
                    break
            if calNum != inShape[n_axis]:
                return None
        return [cosIdDict, sortOutNodes]
        
    if node.op_type == 'Concat':
        concatOutShape = get_shape_by_name(onnx_model, node.output[0])
        if len(concatOutShape) != 4:
            return onnx_model, False
        inputsList = list(node.input)
        outNodesList = get_node_by_input(onnx_model, node.output)
        outSliceNodes = [outNode for outNode in outNodesList if outNode.op_type == 'Slice']
        if len(outSliceNodes) % len(inputsList) != 0:
            return onnx_model, False
        concatAxis = attribute_to_dict(node.attribute).get('axis', 1)
        concatAxis = len(concatOutShape) + concatAxis if concatAxis < 0 else concatAxis
        for sliceNode in outSliceNodes:
            sliceStart = get_tensor_from_initializer(onnx_model, sliceNode.input[1])
            sliceEnd = get_tensor_from_initializer(onnx_model, sliceNode.input[2])
            sliceAxes = get_tensor_from_initializer(onnx_model, sliceNode.input[3])
            sliceStep = get_tensor_from_initializer(onnx_model, sliceNode.input[4])
            sliceAxesPos = len(concatOutShape) + int(sliceAxes) if int(sliceAxes) < 0 else int(sliceAxes)
            if sliceStep.size != 1 or sliceStart.size != 1 or sliceEnd.size != 1 or sliceAxes.size != 1:
                return onnx_model, False
            if int(sliceStep) != 1 or sliceAxesPos == 0 or sliceAxesPos != concatAxis:
                return onnx_model, False
        cosInfos = get_cosponding_sliceid(onnx_model, inputsList, outSliceNodes, concatAxis)
        if cosInfos is None:
            return onnx_model, False
        cosIdsDict, sortSliceNodes = cosInfos
        newSliceNodesList = []
        for input_n in inputsList:
            cosIds = cosIdsDict[input_n]
            curSliceNodes = [sortSliceNodes[idx] for idx in cosIds]
            newStart = 0
            newStartTensor = get_initial_by_value(onnx_model, np.array(newStart, dtype=np.int64))
            if newStartTensor is None:
                newStartTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, curSliceNodes[0].input[1]+'_new'),
                                                         data_type=TensorProto.INT64,
                                                         dims=[1],
                                                         vals=[newStart])
                onnx_model.graph.initializer.append(newStartTensor)
            for snum in range(len(curSliceNodes)):
                srcStart = int(get_tensor_from_initializer(onnx_model, curSliceNodes[snum].input[1]))
                srcEnd = int(get_tensor_from_initializer(onnx_model, curSliceNodes[snum].input[2]))
                newEnd = newStart + (srcEnd - srcStart)
                newEndTensor = get_initial_by_value(onnx_model, np.array(newEnd, dtype=np.int64))
                if newEndTensor is None:
                    newEndTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, curSliceNodes[snum].input[2]+'_new'),
                                                           data_type=TensorProto.INT64,
                                                           dims=[1],
                                                           vals=[newEnd])
                    onnx_model.graph.initializer.append(newEndTensor)
                newSliceNode = copy.deepcopy(curSliceNodes[snum])
                newSliceNode.name = newSliceNode.name+'_new'
                newSliceNode.input[0] = get_node_by_output(onnx_model, input_n).output[0]
                newSliceNode.input[1] = newStartTensor.name
                newSliceNode.input[2] = newEndTensor.name
                newSliceNodesList.append(newSliceNode)
                newStart = newEnd
                newStartTensor = newEndTensor
        newSliceNodesList.reverse()
        onnx_model = insert_node_by_list(onnx_model, newSliceNodesList, node_index)
        onnx_model = delete_nodes(onnx_model, outSliceNodes)
        newOutNodes = get_node_by_input(onnx_model, node.output)
        if not newOutNodes:
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            onnx_model.graph.node.remove(node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertConcatNxSliceAddToNxAdd(onnx_model, node, node_index):
    if node.op_type == 'Add':
        dualDynamic = True if not find_init_by_name(onnx_model, node.input[0]) \
            and not find_init_by_name(onnx_model, node.input[1]) else False
        if dualDynamic:
            addOutShape = get_shape_by_name(onnx_model, node.output[0])
            if len(addOutShape) != 4:
                return onnx_model, False
            inNodes = [get_node_by_output(onnx_model, input_name) for input_name in node.input]
            inNodes = [in_node for in_node in inNodes if in_node is not None]
            concatNodes = [in_node for in_node in inNodes if in_node.op_type == 'Concat']
            if inNodes != concatNodes:
                return onnx_model, False
            outNodes = get_node_by_input(onnx_model, node.output)
            sliceNodes = [outNode for outNode in outNodes if outNode.op_type == 'Slice']
            if sliceNodes != outNodes:
                return onnx_model, False
            concatAxisList = [attribute_to_dict(concatNode.attribute).get('axis', 1) for concatNode in concatNodes]
            concatAxisList = [len(addOutShape) + concatAxis if concatAxis < 0 else concatAxis for concatAxis in concatAxisList]
            if concatAxisList[0] != concatAxisList[1] or len(concatNodes[0].input) != len(concatNodes[1].input) != len(sliceNodes):
                return onnx_model, False
            concatInValues = []
            concatOutValues = []
            for concatNode in concatNodes:
                concatInDatas = {}
                concatOutData = np.array([])
                for concatInput in concatNode.input:
                    inputShape = get_shape_by_name(onnx_model, concatInput)
                    inputData = np.array(np.random.random(size=tuple(inputShape)), dtype=np.float32)
                    if concatInput != concatNode.input[0]:
                        while True:
                            trueList = [cInData for cInData in list(concatInDatas.values()) if (cInData==inputData).all()]
                            if trueList:
                                inputData = np.array(np.random.random(size=tuple(inputShape)), dtype=np.float32)
                            else:
                                break
                    concatInDatas[concatInput] = inputData
                    concatOutData = np.concatenate((concatOutData, inputData), axis=concatAxisList[0]) \
                        if concatInput != concatNode.input[0] else inputData
                concatInValues.append(concatInDatas)
                concatOutValues.append(concatOutData)
            addOutData = concatOutValues[0] + concatOutValues[1]
            sliceOutDatas = []
            for slice_node in sliceNodes:
                sliceStart = get_tensor_from_initializer(onnx_model, slice_node.input[1])
                sliceEnd = get_tensor_from_initializer(onnx_model, slice_node.input[2])
                sliceAxes = get_tensor_from_initializer(onnx_model, slice_node.input[3])
                sliceStep = get_tensor_from_initializer(onnx_model, slice_node.input[4])
                if sliceStart.size != 1 or sliceEnd.size != 1 or sliceAxes.size != 1 \
                    or sliceStep.size not in [0, 1] or int(sliceAxes) != concatAxisList[0]:
                    return onnx_model, False
                if (sliceStep.size == 1 and int(sliceStep) != 1) or int(sliceAxes) == 0:
                    return onnx_model, False 
                sliceData =np.split(addOutData, [int(sliceStart), int(sliceEnd)], int(sliceAxes))[-2:][0]
                sliceOutDatas.append(sliceData)
            newGroupAddValues = [list(concatInValues[0].values())[idx] + list(concatInValues[1].values())[idx] for idx in range(len(sliceNodes))]
            spondIds = []
            for slice_value in sliceOutDatas:
                rid = [sid for sid, group_value in enumerate(newGroupAddValues) \
                    if np.array(group_value==slice_value).all() and sid not in spondIds]
                if rid:
                    spondIds.append(rid[0])
                else:
                    return onnx_model, False
            addNodesList = []
            for idx in range(len(concatNodes[0].input)):
                newAddNode = copy.deepcopy(node)
                newAddNode.input[0] = concatNodes[0].input[idx]
                newAddNode.input[1] = concatNodes[1].input[idx]
                newAddNode.output[0] += '_new%d'%idx
                newAddNode.name += '_new%d'%idx
                addNodesList.append(newAddNode)
            for idx, slice_node in enumerate(sliceNodes):
                cId = spondIds[idx]
                sliceOutNodes = get_node_by_input(onnx_model, slice_node.output)
                for sliceOutNode in sliceOutNodes:
                    for son, son_input in enumerate(list(sliceOutNode.input)):
                        sliceOutNode.input[son] = addNodesList[cId].output[0] \
                            if son_input == slice_node.output[0] else son_input
                onnx_model = delete_value_info_by_name(onnx_model, slice_node.output[0])
            onnx_model = insert_node_by_list(onnx_model, addNodesList, node_index)
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            onnx_model = delete_value_info_by_name(onnx_model, node.input[0])
            onnx_model = delete_value_info_by_name(onnx_model, node.input[1])
            onnx_model = delete_nodes(onnx_model, sliceNodes+[node]+concatNodes)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        else:
            return onnx_model, False
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionReshapeReshape(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Reshape', 'Reshape']):
        topRSNode, botRSNode = get_node_serial_group(onnx_model, node, ['Reshape', 'Reshape'])
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_deleteUselessReshapeSlice(onnx_model, node, node_index):
    def check_useless_node(onnx_model, node):
        inShape = get_shape_by_name(onnx_model, node.input[0])
        outShape = get_shape_by_name(onnx_model, node.output[0])
        if inShape == outShape:
            outNodesList = get_node_by_input(onnx_model, node.output)
            for outNode in outNodesList:
                for idx, outNodeInput in enumerate(outNode.input):
                    outNode.input[idx] = node.input[0] if outNodeInput == node.output[0] else outNodeInput
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
            onnx_model = delete_nodes(onnx_model, [node])
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        else:        
            return onnx_model, False
    if node.op_type == 'Reshape':
        return check_useless_node(onnx_model, node)
    elif node.op_type == 'Slice':
        if len(node.input) >= 5:
            stepsList = get_tensor_from_initializer(onnx_model, node.input[4])
            negFlag = False
            for step in stepsList:
                if step < 0:
                    negFlag = True
                    break
            if not negFlag:
                return check_useless_node(onnx_model, node)
            else:
                return onnx_model, False
        else:
            return check_useless_node(onnx_model, node)
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertConv1DTo2D(onnx_model, node, node_index):
    if node.op_type != 'Conv':
        return onnx_model, False
    conv_attr = attribute_to_dict(node.attribute)
    convInShape = get_shape_by_name(onnx_model, node.input[0])
    if len(convInShape) != 3:
        return onnx_model, False
    kernel_shape = conv_attr.get('kernel_shape')
    conv_attr['kernel_shape'] = [1, 1] if kernel_shape is None else [1, kernel_shape[0]]
    pads = conv_attr.get('pads')
    conv_attr['pads'] = [0, 0, 0, 0] if pads is None else [0, pads[0], 0, pads[1]]
    strides = conv_attr.get('strides')
    conv_attr['strides'] = [1, 1] if strides is None else [1, strides[0]]
    dilations = conv_attr.get('dilations')
    conv_attr['dilations'] = [1, 1] if dilations is None else [1, dilations[0]]
    
    newInShape = convInShape[:2] + [1, convInShape[-1]]
    newInShapeTensor = get_initial_by_value(onnx_model, np.array(newInShape, dtype=np.int64))
    if newInShapeTensor is None:
        newInShapeTensor = onnx.helper.make_tensor(name=node.name+'_4dInShape',
                                                   data_type=TensorProto.INT64,
                                                   dims=[len(newInShape)],
                                                   vals=newInShape)
        onnx_model.graph.initializer.append(newInShapeTensor)
    topRSNode = onnx.helper.make_node(name=node.name+'_4d_reshape',
                                      op_type='Reshape',
                                      inputs=[node.input[0], newInShapeTensor.name],
                                      outputs=[node.name+'_4d_input'])
    conv_wt_arr = get_tensor_from_initializer(onnx_model, node.input[1])
    new_conv_wt_arr = np.expand_dims(conv_wt_arr, axis=2)
    new_conv_wt_tensor = get_initial_by_value(onnx_model, new_conv_wt_arr)
    if new_conv_wt_tensor is None:
        new_conv_wt_tensor = onnx.helper.make_tensor(name=node.input[1]+'_2d',
                                                     data_type=NPDTYPE_2_ONNXDTYPE[new_conv_wt_arr.dtype],
                                                     dims=new_conv_wt_arr.shape,
                                                     vals=new_conv_wt_arr.flatten().tolist())
        onnx_model.graph.initializer.append(new_conv_wt_tensor)
    new2DConvNode = onnx.helper.make_node(name=node.name,
                                          op_type='Conv',
                                          inputs=node.input,
                                          outputs=[node.output[0]+'_4dOut'],
                                          **conv_attr)
    new2DConvNode.input[0] = topRSNode.output[0]
    new2DConvNode.input[1] = new_conv_wt_tensor.name
    
    convOutShape = get_shape_by_name(onnx_model, node.output[0])
    newOutShapeTensor = get_initial_by_value(onnx_model, np.array(convOutShape, dtype=np.int64))
    if newOutShapeTensor is None:
        newOutShapeTensor = onnx.helper.make_tensor(name=node.name+'_3dOutShape',
                                                    data_type=TensorProto.INT64,
                                                    dims=[len(convOutShape)],
                                                    vals=convOutShape)
        onnx_model.graph.initializer.append(newOutShapeTensor)
    
    botRSNode = onnx.helper.make_node(name=node.name+'_3d_reshape',
                                      op_type='Reshape',
                                      inputs=[new2DConvNode.output[0], newOutShapeTensor.name],
                                      outputs=[node.output[0]])
    
    onnx_model.graph.node.remove(node)
    onnx_model = insert_node_by_list(onnx_model, [botRSNode, new2DConvNode, topRSNode], node_index)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convert3DimTransposeXReshapeOrReshapeXTransposeXMulOrAdd(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Mul', 'Reshape']) \
        or check_node_serial_group(onnx_model, node, ['Transpose', 'Add', 'Reshape']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Mul', 'Reshape'])
        if len(serial_nodes) < 3:
            serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Add', 'Reshape'])
        tpNode, calNode, rsNode = serial_nodes
        if calNode.op_type not in ['Mul', 'Add']:
            return onnx_model, False
        tpInShape = get_shape_by_name(onnx_model, tpNode.input[0])
        rsOutShape = get_shape_by_name(onnx_model, rsNode.output[0])
        if len(tpInShape) != 3 or len(rsOutShape) != 4:
            return onnx_model, False
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpInShape))).reverse())
        if tpPerm[0] != 0:
            return onnx_model, False
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        if tpOutShape[:2] != rsOutShape[:2] or tpOutShape[-1] != rsOutShape[-1]:
            return onnx_model, False
        newTpPerm = [0, 3, 2, 1] if tpPerm == [0, 2, 1] else [0, 1, 2, 3]
        newRSShape = tpInShape[:2] + [1, tpInShape[-1]]
        newRSShapeTensor = get_initial_by_value(onnx_model, np.array(newRSShape, dtype=np.int64))
        if newRSShapeTensor is None:
            newRSShapeTensor = onnx.helper.make_tensor(name=rsNode.input[1]+'_new',
                                                       data_type=TensorProto.INT64,
                                                       dims=len(newRSShape),
                                                       vals=newRSShape)
            onnx_model.graph.initializer.append(newRSShapeTensor)
        newRSNode = onnx.helper.make_node(name=rsNode.name+'_new',
                                          op_type='Reshape',
                                          inputs=[tpNode.input[0], newRSShapeTensor.name],
                                          outputs=[tpNode.output[0]+'_rsout'])
        newTPNode = onnx.helper.make_node(name=tpNode.name+'_new',
                                          op_type='Transpose',
                                          inputs=newRSNode.output,
                                          outputs=tpNode.output,
                                          perm=newTpPerm)
        insertNodesList = [newRSNode, newTPNode]
        calOtInput = calNode.input[1] if calNode.input[0] == tpNode.output[0] else calNode.input[0]
        if calOtInput == tpNode.output[0]:
            newCalNode = copy.deepcopy(calNode)
            newCalNode.output[0] = rsNode.output[0]
        else:
            newCalOtInput = calOtInput
            if find_init_by_name(onnx_model, calOtInput):
                calArr = get_tensor_from_initializer(onnx_model, calOtInput)
                if calArr.size != 1:
                    while len(calArr.shape) < 3: calArr = np.expand_dims(calArr, axis=0)
                    calArr = np.expand_dims(calArr, axis=-2)
                    calArrTensor = get_initial_by_value(onnx_model, calArr)
                    if calArrTensor is None:
                        calArrTensor = onnx.helper.make_tensor(name=calOtInput+'_new',
                                                               data_type=NPDTYPE_2_ONNXDTYPE[calArr.dtype],
                                                               dims=calArr.shape,
                                                               vals=calArr.flatten().tolist())
                        onnx_model.graph.initializer.append(calArrTensor)
                    newCalOtInput = calArrTensor.name
            else:
                calOtInShape = get_shape_by_name(onnx_model, calOtInput)
                cal3DOtInShape = [1] * (len(tpOutShape) - len(calOtInShape)) + calOtInShape
                newCalOtInShape = cal3DOtInShape[:2] + [1, cal3DOtInShape[-1]]
                newCalOtInShapeTensor = get_initial_by_value(onnx_model, np.array(newCalOtInShape, dtype=np.int64))
                if newCalOtInShapeTensor is None:
                    newCalOtInShapeTensor = onnx.helper.make_tensor(name=calOtInput+'_newShape',
                                                                    data_type=TensorProto.INT64,
                                                                    dims=[4],
                                                                    vals=newCalOtInShape)
                    onnx_model.graph.initializer.append(newCalOtInShapeTensor)
                newOtRSNode = onnx.helper.make_node(name=calOtInput+'_reshape',
                                                    op_type='Reshape',
                                                    inputs=[calOtInput, newCalOtInShapeTensor.name],
                                                    outputs=[calOtInput+'_rsOut'])
                if newOtRSNode not in list(onnx_model.graph.node):
                    insertNodesList.append(newOtRSNode)
                newCalOtInput = newOtRSNode.output[0]
            newCalNode = onnx.helper.make_node(name=calNode.name+'_new',
                                               op_type=calNode.op_type,
                                               inputs=[newTPNode.output[0], newCalOtInput],
                                               outputs=[rsNode.output[0]])
        
        insertNodesList.append(newCalNode)  
        onnx_model = delete_value_info_by_name(onnx_model, calNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, tpNode.output[0])
        insertNodesList.reverse()
        onnx_model = insert_node_by_list(onnx_model, insertNodesList, node_index)
        onnx_model = delete_nodes(onnx_model, serial_nodes)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
        
    elif check_node_serial_group(onnx_model, node, ['Reshape', 'Mul', 'Transpose']) \
        or check_node_serial_group(onnx_model, node, ['Reshape', 'Add', 'Transpose']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Reshape', 'Mul', 'Transpose'])
        if len(serial_nodes) < 3:
            serial_nodes = get_node_serial_group(onnx_model, node, ['Reshape', 'Add', 'Transpose'])
        rsNode, calNode, tpNode = serial_nodes
        if calNode.op_type not in ['Mul', 'Add']:
            return onnx_model, False
        rsInShape = get_shape_by_name(onnx_model, rsNode.input[0])
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        if len(rsInShape) != 4 or len(tpOutShape) != 3:
            return onnx_model, False
        tpInShape = get_shape_by_name(onnx_model, tpNode.input[0])
        if tpInShape[:2] != rsInShape[:2] or tpInShape[-1] != rsInShape[-1]:
            return onnx_model, False
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpOutShape))).reverse())
        if tpPerm[0] != 0:
            return onnx_model, False
        calOtInput = calNode.input[1] if calNode.input[0] == rsNode.output[0] else calNode.input[0]
        insertNodesList = []
        if calOtInput == rsNode.output[0]:
            newCalNode = copy.deepcopy(calNode)
            newCalNode.input[0] = rsNode.input[0]
            newCalNode.input[1] = rsNode.input[0]
        else:
            newCalOtInput = calOtInput
            if find_init_by_name(onnx_model, calOtInput):
                calArr = get_tensor_from_initializer(onnx_model, calOtInput)
                if calArr.size != 1:
                    while len(calArr.shape) < 3: calArr = np.expand_dims(calArr, axis=0)
                    newCalArr = np.expand_dims(calArr, axis=-2)
                    newCalTensor = get_initial_by_value(onnx_model, newCalArr)
                    if newCalTensor is None:
                        newCalTensor = onnx.helper.make_tensor(name=calOtInput,
                                                               data_type=NPDTYPE_2_ONNXDTYPE[newCalArr.dtype],
                                                               dims=calArr.shape,
                                                               vals=newCalArr.flatten().tolist())
                        onnx_model.graph.initializer.append(newCalTensor)
                    newCalOtInput = newCalTensor.name
            else:
                calOtInShape = get_shape_by_name(onnx_model, calOtInput)
                calOtInShape = [1] * (len(tpInShape) - len(calOtInShape)) + calOtInShape
                calOt4DInShape = calOtInShape[:2] + [1, calOtInShape[-1]]
                newCalOtShapeTensor = get_initial_by_value(onnx_model, np.array(calOt4DInShape, dtype=np.int64))
                if newCalOtShapeTensor.name is None:
                    newCalOtShapeTensor = onnx.helper.make_tensor(name=calOtInput+'_newShape',
                                                                  data_type=TensorProto.INT64,
                                                                  dims=[len(calOt4DInShape)],
                                                                  vals=calOt4DInShape)
                    onnx_model.graph.initializer.append(newCalOtShapeTensor)
                calOtRSNode = onnx.helper.make_node(name=calOtInput+'_reshape',
                                                    op_type='Reshape',
                                                    inputs=[calOtInput, newCalOtShapeTensor.name],
                                                    outputs=[calOtInput+'_rsOut'])
                if calOtRSNode not in list(onnx_model.graph.node):
                    insertNodesList.append(calOtRSNode)
                newCalOtInput = calOtRSNode.output[0]
            newCalNode = onnx.helper.make_node(name=calNode.name+'_new',
                                               op_type=calNode.op_type,
                                               inputs=[rsNode.input[0], newCalOtInput],
                                               outputs=calNode.output)
        insertNodesList.append(newCalNode)
        newPerm = [0, 3, 2, 1] if tpPerm == [0, 2, 1] else [0, 1, 2, 3]
        newTPNode = onnx.helper.make_node(name=tpNode.name+'_new',
                                          op_type='Transpose',
                                          inputs=[newCalNode.output[0]],
                                          outputs=[tpNode.output[0]+'_tpOut'],
                                          perm=newPerm)
        newRSShapeTensor = get_initial_by_value(onnx_model, np.array(tpOutShape, dtype=np.int64))
        if newRSShapeTensor is None:
            newRSShapeTensor = onnx.helper.make_tensor(name=tpNode.output[0]+'_shape',
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(tpOutShape)],
                                                       vals=tpOutShape)
            onnx_model.graph.initializer.append(newRSShapeTensor)
        newRSNode = onnx.helper.make_node(name=rsNode.name+'_new',
                                          op_type='Reshape',
                                          inputs=[newTPNode.output[0], newRSShapeTensor.name],
                                          outputs=[tpNode.output[0]])
        insertNodesList += [newTPNode, newRSNode]
        calOutType = get_dtype_by_name(onnx_model, calNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, calNode.output[0])
        newCalOutValInfo = onnx.helper.make_tensor_value_info(newCalNode.output[0], calOutType, rsInShape)
        onnx_model.graph.value_info.append(newCalOutValInfo)
        onnx_model = insert_node_by_list(onnx_model, insertNodesList, node_index)
        onnx_model = delete_nodes(onnx_model, serial_nodes)
        
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convert3DimReshapeMulOrAddTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Reshape', 'Mul', 'Transpose']) \
        or check_node_serial_group(onnx_model, node, ['Reshape', 'Add', 'Transpose']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Reshape', 'Mul', 'Transpose'])
        if len(serial_nodes) < 3:
            serial_nodes = get_node_serial_group(onnx_model, node, ['Reshape', 'Add', 'Transpose'])
        rsNode, calNode, tpNode = serial_nodes
        if calNode.op_type not in ['Mul', 'Add']:
            return onnx_model, False
        rsInShape = get_shape_by_name(onnx_model, rsNode.input[0])
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        if len(rsInShape) != 4 or len(tpOutShape) != 3:
            return onnx_model, False
        tpInShape = get_shape_by_name(onnx_model, tpNode.input[0])
        if tpInShape[:2] != rsInShape[:2] or tpInShape[-1] != rsInShape[-1]:
            return onnx_model, False
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpOutShape))).reverse())
        if tpPerm[0] != 0:
            return onnx_model, False
        calOtInput = calNode.input[1] if calNode.input[0] == rsNode.output[0] else calNode.input[0]
        insertNodesList = []
        if calOtInput == rsNode.output[0]:
            newCalNode = copy.deepcopy(calNode)
            newCalNode.input[0] = rsNode.input[0]
            newCalNode.input[1] = rsNode.input[0]
        else:
            newCalOtInput = calOtInput
            if find_init_by_name(onnx_model, calOtInput):
                calArr = get_tensor_from_initializer(onnx_model, calOtInput)
                if calArr.size != 1:
                    while len(calArr.shape) < 3: calArr = np.expand_dims(calArr, axis=0)
                    newCalArr = np.expand_dims(calArr, axis=-2)
                    newCalTensor = get_initial_by_value(onnx_model, newCalArr)
                    if newCalTensor is None:
                        newCalTensor = onnx.helper.make_tensor(name=calOtInput,
                                                               data_type=NPDTYPE_2_ONNXDTYPE[newCalArr.dtype],
                                                               dims=calArr.shape,
                                                               vals=newCalArr.flatten().tolist())
                        onnx_model.graph.initializer.append(newCalTensor)
                    newCalOtInput = newCalTensor.name
            else:
                calOtInShape = get_shape_by_name(onnx_model, calOtInput)
                calOtInShape = [1] * (len(tpInShape) - len(calOtInShape)) + calOtInShape
                calOt4DInShape = calOtInShape[:2] + [1, calOtInShape[-1]]
                newCalOtShapeTensor = get_initial_by_value(onnx_model, np.array(calOt4DInShape, dtype=np.int64))
                if newCalOtShapeTensor.name is None:
                    newCalOtShapeTensor = onnx.helper.make_tensor(name=calOtInput+'_newShape',
                                                                  data_type=TensorProto.INT64,
                                                                  dims=[len(calOt4DInShape)],
                                                                  vals=calOt4DInShape)
                    onnx_model.graph.initializer.append(newCalOtShapeTensor)
                calOtRSNode = onnx.helper.make_node(name=calOtInput+'_reshape',
                                                    op_type='Reshape',
                                                    inputs=[calOtInput, newCalOtShapeTensor.name],
                                                    outputs=[calOtInput+'_rsOut'])
                if calOtRSNode not in list(onnx_model.graph.node):
                    insertNodesList.append(calOtRSNode)
                newCalOtInput = calOtRSNode.output[0]
            newCalNode = onnx.helper.make_node(name=calNode.name+'_new',
                                               op_type=calNode.op_type,
                                               inputs=[rsNode.output[0], newCalOtInput],
                                               outputs=calNode.output)
        insertNodesList.append(newCalNode)
        newPerm = [0, 3, 2, 1] if tpPerm == [0, 2, 1] else [0, 1, 2, 3]
        newTPNode = onnx.helper.make_node(name=tpNode.name+'_new',
                                          op_type='Transpose',
                                          inputs=[newCalNode.output[0]],
                                          outputs=[tpNode.output[0]+'_tpOut'],
                                          perm=newPerm)
        newRSShapeTensor = get_initial_by_value(onnx_model, np.array(tpOutShape, dtype=np.int64))
        if newRSShapeTensor is None:
            newRSShapeTensor = onnx.helper.make_tensor(name=tpNode.output[0]+'_shape',
                                                       data_type=TensorProto.INT64,
                                                       dims=[len(tpOutShape)],
                                                       vals=tpOutShape)
            onnx_model.graph.initializer.append(newRSShapeTensor)
        newRSNode = onnx.helper.make_node(name=rsNode.name+'_new',
                                          op_type='Reshape',
                                          inputs=[newTPNode.output[0], newRSShapeTensor.name],
                                          outputs=[tpNode.output[0]])
        insertNodesList += [newTPNode, newRSNode]
        calOutType = get_dtype_by_name(onnx_model, calNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, calNode.output[0])
        newCalOutValInfo = onnx.helper.make_tensor_value_info(newCalNode.output[0], calOutType, rsInShape)
        onnx_model.graph.value_info.append(newCalOutValInfo)
        onnx_model = insert_node_by_list(onnx_model, insertNodesList, node_index)
        onnx_model = delete_nodes(onnx_model, serial_nodes)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertCalculateTransposeReshapeMul(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Mul']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Mul'])
        tpNode, rsNode, calNode = serial_nodes
        nextNodesList = get_node_by_input(onnx_model, calNode.output)
        if len(nextNodesList) != 1 or nextNodesList[0].op_type not in ['Add']:
            return onnx_model, False
        if nextNodesList[0].op_type == 'Add' \
            and ((find_init_by_name(onnx_model, nextNodesList[0].input[0]) \
                or find_init_by_name(onnx_model, nextNodesList[0].input[1])) \
                    or nextNodesList[0].input[0] == nextNodesList[0].input[1]):
            return onnx_model, False
        preNode = get_node_by_output(onnx_model, tpNode.input[0])
        if preNode is None or preNode.op_type not in ['Conv', 'Add', 'Mul', 'Resize']:
            return onnx_model, False
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        rsOutShape = get_shape_by_name(onnx_model, rsNode.output[0])
        if len(tpOutShape) != 4 or len(rsOutShape) != 3 \
            or tpOutShape[:2] != rsOutShape[:2] or tpOutShape[-1] != rsOutShape[-1]:
            return onnx_model, False
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpOutShape))).reverse())
        newPerm = [tpPerm[idx] for idx in tpPerm]
        newPerm = [0, 3, 2, 1] if tpPerm == [0, 3, 2, 1] else newPerm
        calStInput = calNode.input[1] if find_init_by_name(onnx_model, calNode.input[1]) else calNode.input[0]
        if not find_init_by_name(onnx_model, calStInput):
            return onnx_model, False
        calArr = get_tensor_from_initializer(onnx_model, calStInput)
        newCalInput = calStInput
        if calArr.size != 1:
            while len(calArr.shape) < 3: calArr = np.expand_dims(calArr, axis=0)
            cal4DArr = np.expand_dims(calArr, axis=-2)
            newCalArr = np.transpose(cal4DArr, tuple(newPerm))
            newCalTensor = get_initial_by_value(onnx_model, newCalArr)
            if newCalTensor is None:
                newCalTensor = onnx.helper.make_tensor(name=calStInput+'_new',
                                                       data_type=NPDTYPE_2_ONNXDTYPE[newCalArr.dtype],
                                                       dims=newCalArr.shape,
                                                       vals=newCalArr.flatten().tolist())
                onnx_model.graph.initializer.append(newCalTensor)
            newCalInput = newCalTensor.name
        newCalNode = onnx.helper.make_node(name=calNode.name+'_new',
                                           op_type=calNode.op_type,
                                           inputs=[tpNode.input[0], newCalInput],
                                           outputs=[calNode.output[0]+'_new'])
        onnx_model = delete_value_info_by_name(onnx_model, rsNode.output[0])
        rsNode.output[0] = calNode.output[0]
        tpNode.input[0] = newCalNode.output[0]
        onnx_model.graph.node.insert(node_index, newCalNode)
        onnx_model.graph.node.remove(calNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertCalculateTransposeReshapeSoftmax(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'LogSoftmax']) \
        or check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Softmax']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'LogSoftmax'])
        if len(serial_nodes) != 3:
            serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Softmax'])
        tpNode, rsNode, calNode = serial_nodes
        calInShape = get_shape_by_name(onnx_model, calNode.input[0])
        axis = attribute_to_dict(calNode.attribute).get('axis', 1)
        posAxis = len(calInShape) + axis if axis < 0 else axis
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        if len(tpOutShape) != 4 or len(calInShape) != 3 \
            or tpOutShape[:2] != calInShape[:2] or tpOutShape[-1] != calInShape[-1]:
            return onnx_model, False
        pos4DAxis = posAxis + 1 if posAxis == 2 else posAxis
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpOutShape))).reverse())
        newAxis = tpPerm.index(pos4DAxis)
        if newAxis != 1:
            return onnx_model, False
        newCalNode = onnx.helper.make_node(name=calNode.name,
                                           op_type=calNode.op_type,
                                           inputs=[tpNode.input[0]],
                                           outputs=[calNode.output[0]+'_axis1'],
                                           axis=newAxis)
        tpNode.input[0] = newCalNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, rsNode.output[0])
        rsNode.output[0] = calNode.output[0]
        onnx_model.graph.node.remove(calNode)
        onnx_model.graph.node.insert(node_index, newCalNode)
        return onnx_model, True
    elif check_node_serial_group(onnx_model, node, ['Transpose', 'LogSoftmax']) \
        or check_node_serial_group(onnx_model, node, ['Transpose', 'Softmax']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'LogSoftmax'])
        if len(serial_nodes) != 2:
            serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Softmax'])
        tpNode, calNode = serial_nodes
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        if len(tpOutShape) != 4:
            return onnx_model, False
        calInShape = get_shape_by_name(onnx_model, calNode.input[0])
        axis = attribute_to_dict(calNode.attribute).get('axis', 1)
        posAxis = len(calInShape) + axis if axis < 0 else axis
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpOutShape))).reverse())
        newAxis = tpPerm.index(posAxis)
        if newAxis != 1:
            return onnx_model, False
        outNodesList = get_node_by_input(onnx_model, calNode.output)
        outTpNodesList = [outNode for outNode in outNodesList if outNode.op_type == 'Transpose']
        newCalNode = onnx.helper.make_node(name=calNode.name,
                                op_type=calNode.op_type,
                                inputs=[tpNode.input[0]],
                                outputs=[calNode.output[0]+'_axis1'],
                                axis=newAxis)
        netOutNamesList = [netOutput.name for netOutput in onnx_model.graph.output]
        if tpPerm == [0, 3, 2, 1]:
            tpInShape = get_shape_by_name(onnx_model, tpNode.input[0])
            tpInDtype = get_dtype_by_name(onnx_model, tpNode.input[0])
            newCalOutValue = onnx.helper.make_tensor_value_info(newCalNode.output[0], tpInDtype, tpInShape)
            onnx_model.graph.value_info.append(newCalOutValue)
            if len(outNodesList) != len(outTpNodesList):
                tpNode.input[0] = newCalNode.output[0]
                onnx_model = delete_value_info_by_name(onnx_model, tpNode.output[0])
                tpNode.output[0] = calNode.output[0]
                onnx_model.graph.node.remove(calNode)
                onnx_model.graph.node.insert(node_index, newCalNode)
            else:
                delNodesList = [tpNode, calNode]
                insertTpNodesList = []
                for outTpNode in outTpNodesList:
                    outTpPerm = attribute_to_dict(outTpNode.attribute).get('perm', list(range(len(calInShape))).reverse())
                    newTpPerm = [tpPerm[ip] for ip in outTpPerm]
                    if newTpPerm == list(range(len(calInShape))):
                        outTpOutNodesList = get_node_by_input(onnx_model, outTpNode.output)
                        for outTpOutNode in outTpOutNodesList:
                            for itx, outTpOutNodeInput in enumerate(outTpOutNode.input):
                                outTpOutNode.input[itx] = newCalNode.output[0] \
                                    if outTpOutNodeInput == outTpNode.output[0] else outTpOutNodeInput
                    else:
                        newTpNode = onnx.helper.make_node(name=tpNode.name+'_'+outTpNode.name+'_new',
                                                        op_type='Transpose',
                                                        inputs=newCalNode.output,
                                                        outputs=[outTpNode.output[0]],
                                                        perm=newTpPerm)
                        insertTpNodesList.append(newTpNode)
                    delNodesList.append(outTpNode)
                for delNode in delNodesList:
                    onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
                    onnx_model.graph.node.remove(delNode)
                insertTpNodesList.reverse()
                onnx_model = insert_node_by_list(onnx_model, insertTpNodesList, node_index)
                onnx_model.graph.node.insert(node_index, newCalNode)
                return onnx_model, True
        else:
            newTopTPPerm = [tpPerm[idx] for idx in tpPerm]
            tpInShape = get_shape_by_name(onnx_model, tpNode.input[0])
            newTopTPOutShape = [tpInShape[idx] for idx in newTopTPPerm]
            tpInDtype = get_dtype_by_name(onnx_model, tpNode.input[0])
            newCalOutValue = onnx.helper.make_tensor_value_info(newCalNode.output[0], tpInDtype, newTopTPOutShape)
            onnx_model.graph.value_info.append(newCalOutValue)
            delNodesList = [tpNode, calNode]
            if newTopTPPerm != list(range(len(tpInShape))):
                newCalNode.input[0] = tpNode.output[0]
                del tpNode.attribute[:]
                newTopTPAttr = onnx.helper.make_attribute('perm', newTopTPPerm)
                tpNode.attribute.append(newTopTPAttr)
                delNodesList.remove(tpNode)
            insertNodesList = [newCalNode]
            if len(outNodesList) != len(outTpNodesList):
                newTPTransPerm = [list(range(len(tpInShape)))[idx] for idx in newTopTPPerm]
                newTPTransNode = onnx.helper.make_node(name=tpNode.name+'_sync_' + calNode.name,
                                                  op_type='Transpose',
                                                  inputs=[newCalNode.output[0]],
                                                  outputs=[calNode.output[0]],
                                                  perm=[newTPTransPerm])
                insertNodesList.append(newTPTransNode)
                onnx_model = delete_value_info_by_name(onnx_model, calNode.output[0])
                newTPTransOutShape = [newTopTPOutShape[idx] for idx in newTPTransPerm]
                newTPTransOutValueInfo = onnx.helper.make_tensor_value_info(newTPTransNode.output[0], tpInDtype, newTPTransOutShape)
                onnx_model.graph.value_info.append(newTPTransOutValueInfo)
            else:
                for outTPNode in outTpNodesList:
                    outTPPerm = attribute_to_dict(outTPNode.attribute).get('perm', list(range(len(tpInShape))))
                    newOutTPPerm = [outTPPerm[idx] for idx in outTPPerm]
                    outTPOutNodesList = get_node_by_input(onnx_model, outTPNode.output)
                    if newOutTPPerm == list(range(len(tpInShape))):
                        if outTPNode.output[0] in netOutNamesList:
                            for netOutId, netOutput in enumerate(onnx_model.graph.output):
                                if netOutput.name == outTPNode.output[0]:
                                    onnx_model.graph.output.remove(netOutput)
                                    if newCalNode.output[0] not in netOutNamesList:
                                        netOutNamesList[netOutId] = newCalNode.output[0]
                                        onnx_model.graph.output.insert(netOutId, newCalOutValue)
                        else:
                            for outTPOutNode in outTPOutNodesList:
                                for cur_index, outTPOutNodeIn in enumerate(outTPOutNode.input):
                                    outTPOutNode.input[cur_index] = newCalNode.output[0] \
                                        if outTPOutNodeIn == outTPNode.output[0] else outTPOutNodeIn 
                    else:
                        del outTPNode.attribute[:]
                        newOutTPPermAttr = onnx.helper.make_attribute('perm', newOutTPPerm)
                        outTPNode.attribute.append(newOutTPPermAttr)
                        outTPNode.input[0] = newCalNode.output[0]
            calNodeIndex = get_node_id(onnx_model, calNode)
            onnx_model = insert_node_by_list(onnx_model, insertNodesList, calNodeIndex)
            onnx_model = delete_nodes(onnx_model, delNodesList)
            onnx_model = delete_useless_input_in_initializer(onnx_model)   
            return onnx_model, False
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeReshapeSMishReshapeTranspose(onnx_model, node, node_index):
    def check_smish_serial_nodes(onnx_model, node):
        if node.op_type != 'Mul' or find_init_by_name(onnx_model, node.input[0]) \
            or find_init_by_name(onnx_model, node.input[1]):
            return None
        sigmoidNode = get_node_by_output(onnx_model, node.input[0])
        if sigmoidNode is None or sigmoidNode.op_type != 'Sigmoid':
            sigmoidNode = get_node_by_output(onnx_model, node.input[1])
        if sigmoidNode is None or sigmoidNode.op_type != 'Sigmoid':
            return None
        mulOtInput = node.input[0] if sigmoidNode.output[0] == node.input[1] else node.input[1]
        if mulOtInput != sigmoidNode.input[0]:
            return None
        return [node, sigmoidNode]
    
    serial_nodes = check_smish_serial_nodes(onnx_model, node)
    if serial_nodes is not None:
        mulNode, sigNode = serial_nodes
        topRSNode = get_node_by_output(onnx_model, sigNode.input[0])
        if topRSNode.op_type != 'Reshape':
            return onnx_model, False
        topTPNode = get_node_by_output(onnx_model, topRSNode.input[0])
        if topTPNode.op_type != 'Transpose':
            return onnx_model, False
        topRSInShape = get_shape_by_name(onnx_model, topRSNode.input[0])
        topTPInShape = get_shape_by_name(onnx_model, topTPNode.input[0])
        topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
        nextNodesList = get_node_by_input(onnx_model, mulNode.output)
        nextSerialNodes = []
        nextRSNodesList = [nextNode for nextNode in nextNodesList if nextNode.op_type == 'Reshape']
        if len(nextRSNodesList) != len(nextNodesList):
            nextTPNodesList = [nextNode for nextNode in nextNodesList if nextNode.op_type == 'Transpose']
            if len(nextTPNodesList) != len(nextNodesList):
                return onnx_model, False
            for botTPNode in nextTPNodesList:
                botTPNextNodesList = get_node_by_input(onnx_model, botTPNode.output)
                if len(botTPNextNodesList) == 1 and botTPNextNodesList[0].op_type == 'Reshape':
                    nowSerialNode = [botTPNode, botTPNextNodesList[0]]
                    nextSerialNodes.append(nowSerialNode)
            if len(nextSerialNodes) != len(nextTPNodesList):
                return onnx_model, False
            topTPPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topTPInShape))).reverse())
            newTPPerm = [topTPPerm[idx] for idx in topTPPerm]
            newTPPerm = topTPPerm if newTPPerm == list(range(len(topTPInShape))) else newTPPerm
            tensor_size = int(np.prod(np.array(topRSOutShape, np.int64)))
            randArr = np.random.randn(tensor_size)
            randArr = np.reshape(randArr, tuple(topRSOutShape))
            topRSRandInArr = np.reshape(randArr, tuple(topRSInShape))
            topTPRandInArr = np.transpose(topRSRandInArr, tuple(newTPPerm))
            for nextSerial in nextSerialNodes:
                botTPOutShape = get_shape_by_name(onnx_model, nextSerial[0].output[0])
                botRSOutShape = get_shape_by_name(onnx_model, nextSerial[1].output[0])
                botTPPerm = attribute_to_dict(nextSerial[0].attribute).get('perm', list(range(len(botTPOutShape))).reverse())
                botTPRandOutArr = np.transpose(randArr, tuple(botTPPerm))
                botRSRandOutArr = np.reshape(botTPRandOutArr, tuple(botRSOutShape))
                if botRSOutShape != topTPInShape or not (botRSRandOutArr == topTPRandInArr).all():
                    return onnx_model, False
        else:
            for botRSNode in nextRSNodesList:
                botRSNextNodesList = get_node_by_input(onnx_model, botRSNode.output)
                if len(botRSNextNodesList) == 1 and botRSNextNodesList[0].op_type == 'Transpose':
                    nowSerialNode = [botRSNode, botRSNextNodesList[0]]
                    nextSerialNodes.append(nowSerialNode)
            if len(nextSerialNodes) != len(nextRSNodesList):
                return onnx_model, False
            for nextSerial in nextSerialNodes:
                botRSOutShape = get_shape_by_name(onnx_model, nextSerial[0].output[0])
                botTPOutShape = get_shape_by_name(onnx_model, nextSerial[1].output[0])
                if botRSOutShape != topRSInShape or botTPOutShape != topTPInShape:
                    return onnx_model, False
        mulNode.input[list(mulNode.input).index(sigNode.input[0])] = topTPNode.input[0]
        sigNode.input[0] = topTPNode.input[0]
        for nextSerial in nextSerialNodes:
            bot2stNextNodesList = get_node_by_input(onnx_model, nextSerial[1].output)
            for bot2stNextNode in bot2stNextNodesList:
                for curIdx, bot2stNextInput in enumerate(bot2stNextNode.input):
                    bot2stNextNode.input[curIdx] = mulNode.output[0] \
                        if bot2stNextInput == nextSerial[1].output[0] else bot2stNextInput
        onnx_model = delete_value_info_by_name(onnx_model, sigNode.output[0])
        mulOutDtype = get_dtype_by_name(onnx_model, mulNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, mulNode.output[0])
        mulOutValInfo = onnx.helper.make_tensor_value_info(mulNode.output[0], mulOutDtype, topTPInShape)
        onnx_model.graph.value_info.append(mulOutValInfo)
        for delNode in [topTPNode, topRSNode]:
            onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
            onnx_model.graph.node.remove(delNode)
        for delSerial in nextSerialNodes:
            onnx_model = delete_value_info_by_name(onnx_model, delSerial[0].output[0])
            onnx_model = delete_value_info_by_name(onnx_model, delSerial[1].output[0])
            onnx_model = delete_nodes(onnx_model, delSerial)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionMultiTransposeReshapeXMultiReshapeTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape']):
        topTPNode, topRSNode = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape'])
        topTPInShape = get_shape_by_name(onnx_model, topTPNode.input[0])
        topRSInShape = get_shape_by_name(onnx_model, topRSNode.input[0])
        topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
        netOutNamesList = [netOutput.name for netOutput in onnx_model.graph.output]
        if topRSNode.output[0] in netOutNamesList:
            return onnx_model, False 
        calNodesList = get_node_by_input(onnx_model, topRSNode.output)
        calRSNodesList = [calNode for calNode in calNodesList if calNode.op_type == 'Reshape']
        botSerialList = []
        if len(calRSNodesList) == len(calNodesList):
            for botRSNode in calRSNodesList:
                botRSNextNodesList = get_node_by_input(onnx_model, botRSNode.output)
                if len(botRSNextNodesList) == 1 and botRSNextNodesList[0].op_type == 'Transpose':
                    botSerialList.append([botRSNode, botRSNextNodesList[0]])
            if len(botSerialList) != len(calRSNodesList):
                return onnx_model, False
            for botSerialNodes in botSerialList:
                botRSNode = botSerialNodes[0]
                botTPNode = botSerialNodes[1]
                botRSOutShape = get_shape_by_name(onnx_model, botRSNode.output[0])
                botTPOutShape = get_shape_by_name(onnx_model, botTPNode.output[0])
                if botRSOutShape != topRSInShape or botTPOutShape != topTPInShape:
                    return onnx_model, False
            for botSerialNodes in botSerialList:
                botTPNextNodesList = get_node_by_input(onnx_model, botSerialNodes[1].output)
                for botTPNextNode in botTPNextNodesList:
                    for curIdx, botTPNextNodeIn in enumerate(botTPNextNode.input):
                        botTPNextNode.input[curIdx] = topTPNode.input[0] \
                            if botTPNextNodeIn == botSerialNodes[1].output[0] else botTPNextNodeIn
            for topNode in [topTPNode, topRSNode]:
                onnx_model = delete_value_info_by_name(onnx_model, topNode.output[0])
                onnx_model.graph.node.remove(topNode)
            for botSerialNodes in botSerialList:
                onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[0].output[0])
                onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[1].output[1])
                onnx_model = delete_nodes(onnx_model, botSerialNodes)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        elif len(calRSNodesList) == 0:
            double_types = ['Add', 'Mul']
            single_types = ['Sigmoid', 'Relu', 'PRelu', 'LeakyRelu', 'tanh']
            if len(calNodesList) != 1 or calNodesList[0].op_type not in (double_types + single_types):
                return onnx_model, False
            calNode = calNodesList[0]
            calOutNodesList = get_node_by_input(onnx_model, calNode.output)
            calOutRSNodesList = [calOutNode for calOutNode in calOutNodesList if calOutNode.op_type == 'Reshape']
            if len(calOutNodesList) != len(calOutRSNodesList):
                return onnx_model, False
            for calOutRSNode in calOutRSNodesList:
                calOutRSNextNodes = get_node_by_input(onnx_model, calOutRSNode.output)
                if len(calOutRSNextNodes) == 1 and calOutRSNextNodes[0].op_type == 'Transpose':
                    botSerialList.append([calOutRSNode, calOutRSNextNodes[0]])
            for botSerialNodes in botSerialList:
                botRSOutShape = get_shape_by_name(onnx_model, botSerialNodes[0].output[0])
                botTPOutShape = get_shape_by_name(onnx_model, botSerialNodes[1].output[0])
                if botRSOutShape != topRSInShape or botTPOutShape != topTPInShape:
                    return onnx_model, False
            if calNode.op_type in double_types:
                if find_init_by_name(onnx_model, calNode.input[0]) or find_init_by_name(onnx_model, calNode.input[1]):
                    calStIn = calNode.input[1] if find_init_by_name(onnx_model, calNode.input[1]) else calNode.input[0]
                    calArr = get_tensor_from_initializer(onnx_model, calStIn)
                    botTPPerm = attribute_to_dict(botSerialList[0][1].attribute).get('perm', list(range(len(topTPInShape))).reverse())
                    if calArr.size == 1:
                        newCalStIn = calStIn
                    elif calArr.size == int(np.sum(np.array(topRSInShape, dtype=np.int64))):
                        newCalArr = np.reshape(calArr, tuple(topRSInShape))
                        newCalArr = np.transpose(newCalArr, tuple(botTPPerm))
                        newCalTensor = get_initial_by_value(onnx_model, newCalArr)
                        if newCalTensor is None:
                            newCalTensor = onnx.helper.make_tensor(name=calStIn+'_new',
                                                                   data_type=NPDTYPE_2_ONNXDTYPE[newCalArr.dtype],
                                                                   dims=newCalArr.shape,
                                                                   vals=newCalArr.flatten().tolist())
                            onnx_model.graph.initializer.append(newCalTensor)
                        newCalStIn = newCalTensor.name
                    elif len(topRSInShape) == len(topRSOutShape) - 1 \
                        and topRSInShape[:(len(topRSInShape)-2)] == topRSOutShape[:(len(topRSOutShape)-1)] \
                            and topRSInShape[-1] == topRSOutShape[-1]:
                        while len(calArr.shape) < len(topRSOutShape): calArr = np.expand_dims(calArr, axis=0)
                        newCalArr = np.expand_dims(calArr, axis=-2)
                        newCalArr = np.transpose(newCalArr, tuple(botTPPerm))
                        newCalTensor = get_initial_by_value(onnx_model, newCalArr)
                        if newCalTensor is None:
                            newCalTensor = onnx.helper.make_tensor(name=calStIn+'_new',
                                                                   data_type=NPDTYPE_2_ONNXDTYPE[newCalArr.dtype],
                                                                   dims=newCalArr.shape,
                                                                   vals=newCalArr.flatten().tolist())
                            onnx_model.graph.initializer.append(newCalTensor)
                        newCalStIn = newCalTensor.name
                    else:
                        newCalArr = np.broadcast_to(calArr, tuple(topRSOutShape))
                        newCalArr = np.reshape(newCalArr, tuple(topRSInShape))
                        newCalArr = np.transpose(newCalArr, tuple(botTPPerm))
                        newCalTensor = get_initial_by_value(onnx_model, newCalArr)
                        if newCalTensor is None:
                            newCalTensor = onnx.helper.make_tensor(name=calStIn+'_new',
                                                                   data_type=NPDTYPE_2_ONNXDTYPE[newCalArr.dtype],
                                                                   dims=newCalArr.shape,
                                                                   vals=newCalArr.flatten().tolist())
                            onnx_model.graph.initializer.append(newCalTensor)
                        newCalStIn = newCalTensor.name
                    calNode.input[list(calNode.input).index(topRSNode.output[0])] = topTPNode.input[0]
                    calNode.input[list(calNode.input).index(calStIn)] = newCalStIn
                    for botSerialNodes in botSerialList:
                        botTPNextNodesList = get_node_by_input(onnx_model, botSerialNodes[1].output)
                        for botTPNextNode in botTPNextNodesList:
                            for curIdx, botTPNextInput in enumerate(botTPNextNode.input):
                                botTPNextNode.input[curIdx] = calNode.output[0] \
                                    if botTPNextInput == botSerialNodes[1].output[0] else botTPNextInput
                    for topNode in [topTPNode, topRSNode]:
                        onnx_model = delete_value_info_by_name(onnx_model, topNode.output[0])
                        onnx_model.graph.node.remove(topNode)
                    for botSerialNodes in botSerialList:
                        onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[0].output[0])
                        onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[1].output[0])
                        onnx_model = delete_nodes(onnx_model, botSerialNodes)
                    onnx_model = delete_useless_input_in_initializer(onnx_model)
                    return onnx_model, True
                else:
                    calOtIn = calNode.input[1] if topRSNode.output[0] == calNode.input[0] else calNode.input[0]
                    calOtPreRSNode = get_node_by_output(onnx_model, calOtIn)
                    if calOtPreRSNode is None or calOtPreRSNode.op_type != 'Reshape':
                        return onnx_model, False
                    calOtPreTPNode = get_node_by_output(onnx_model, calOtPreRSNode.input[0])
                    if calOtPreTPNode is None or calOtPreTPNode.op_type != 'Transpose':
                        return onnx_model, False
                    calOtPreRSInShape = get_shape_by_name(onnx_model, calOtPreRSNode.input[0])
                    calOtPreTPInShape = get_shape_by_name(onnx_model, calOtPreTPNode.input[0])
                    if calOtPreRSInShape != topRSInShape or calOtPreTPInShape != topTPInShape:
                        return onnx_model, False
                    calNode.input[list(calNode.input).index(topRSNode.output[0])] = topTPNode.input[0]
                    calNode.input[list(calNode.input).index(calOtIn)] = calOtPreTPNode.input[0]
                    for botSerialNodes in botSerialList:
                        botTPNextNodesList = get_node_by_input(onnx_model, botSerialNodes[1].output)
                        for botTPNextNode in botTPNextNodesList:
                            for curIdx, botTPNextInput in enumerate(botTPNextNode.input):
                                botTPNextNode.input[curIdx] = calNode.output[0] \
                                    if botTPNextInput == botSerialNodes[1].output[0] else botTPNextInput
                    for topNode in [topTPNode, topRSNode, calOtPreTPNode, calOtPreRSNode]:
                        onnx_model = delete_value_info_by_name(onnx_model, topNode.output[0])
                        onnx_model.graph.node.remove(topNode)
                    for botSerialNodes in botSerialList:
                        onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[0].output[0])
                        onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[1].output[0])
                        onnx_model = delete_nodes(onnx_model, botSerialNodes)
                    onnx_model = delete_useless_input_in_initializer(onnx_model)
                    return onnx_model, True
            elif calNode.op_type in single_types:
                calNode.input[0] = topTPNode.input[0]
                for botSerialNodes in botSerialList:
                    botTPNextNodesList = get_node_by_input(onnx_model, botSerialNodes[1].output)
                    for botTPNextNode in botTPNextNodesList:
                        for curIdx, botTPNextInput in enumerate(botTPNextNode.input):
                            botTPNextNode.input[curIdx] = calNode.output[0] \
                                if botTPNextInput == botSerialNodes[1].output[0] else botTPNextInput
                for topNode in [topTPNode, topRSNode]:
                    onnx_model = delete_value_info_by_name(onnx_model, topNode.output[0])
                    onnx_model.graph.node.remove(topNode)
                for botSerialNodes in botSerialList:
                    onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[0].output[0])
                    onnx_model = delete_value_info_by_name(onnx_model, botSerialNodes[1].output[0])
                    onnx_model = delete_nodes(onnx_model, botSerialNodes)
                onnx_model = delete_useless_input_in_initializer(onnx_model)
                return onnx_model, True
            else:
                return onnx_model, False                                
        else:
            return onnx_model, False
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionReshapeSplitSMishReshape(onnx_model, node, node_index):
    if node.op_type != 'Mul' \
        or find_init_by_name(onnx_model, node.input[0]) or find_init_by_name(onnx_model, node.input[1]):
        return onnx_model, False
    sigNode = get_node_by_output(onnx_model, node.input[1])
    splitNode = get_node_by_output(onnx_model, node.input[0])
    if sigNode is None or splitNode is None:
        return onnx_model, False
    sigNode, splitNode = [sigNode, splitNode] if sigNode.op_type == 'Sigmoid' else [splitNode, sigNode]
    if sigNode.op_type != 'Sigmoid' or splitNode.op_type != 'Split':
        return onnx_model, False
    sigInNode = get_node_by_output(onnx_model, sigNode.input[0])
    if sigInNode != splitNode:
        return onnx_model, False
    topRSNode = get_node_by_output(onnx_model, splitNode.input[0])
    if topRSNode.op_type != 'Reshape':
        return onnx_model, False
    topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
    topRSInShape = get_shape_by_name(onnx_model, topRSNode.input[0])
    if len(topRSOutShape) != len(topRSInShape) - 1 \
        or topRSOutShape[:(len(topRSOutShape) - 1)] != topRSInShape[:(len(topRSOutShape) - 1)] \
            or topRSOutShape[-1] != topRSInShape[-1]:
                return onnx_model, False
    splitAxis = attribute_to_dict(splitNode.attribute).get('axis', 1)
    splitPosAxis = len(topRSOutShape) + splitAxis if splitAxis < 0 else splitAxis
    newSplitAxis = splitPosAxis if splitPosAxis <= len(topRSOutShape) - 2 else len(topRSInShape) - 1
    botNextNodesList = get_node_by_input(onnx_model, node.output)
    botNextRSNodesList = [botNode for botNode in botNextNodesList if botNode.op_type == 'Reshape']
    if len(botNextRSNodesList) != len(botNextNodesList):
        return onnx_model, False
    topRSSplitInShape = copy.deepcopy(topRSInShape)
    topRSSplitInShape[newSplitAxis] //= 2
    for botRSNode in botNextRSNodesList:
        botRSOutShape = get_shape_by_name(onnx_model, botRSNode.output[0])
        if botRSOutShape != topRSSplitInShape:
            return onnx_model, False
    newSplitAxisAttr = onnx.helper.make_attribute('axis', newSplitAxis)
    del splitNode.attribute[:]
    splitNode.attribute.append(newSplitAxisAttr)
    splitNode.input[0] = topRSNode.input[0]
    for botRSNode in botNextRSNodesList:
        botRSNextNodes = get_node_by_input(onnx_model, botRSNode.output)
        onnx_model = delete_value_info_by_name(onnx_model, botRSNode.output[0])
        for botRSNextNode in botRSNextNodes:
            for curIdx, botRSNextIn in enumerate(botRSNextNode.input):
                botRSNextNode.input[curIdx] = node.output[0] if botRSNextIn == botRSNode.output[0] else botRSNextIn
    mulOutType = get_dtype_by_name(onnx_model, node.output[0])
    onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
    newOutValInfo = onnx.helper.make_tensor_value_info(node.output[0], mulOutType, 
                                                       get_shape_by_name(onnx_model, botRSNextNodes[0].output[0]))
    onnx_model.graph.value_info.append(newOutValInfo)
    onnx_model = delete_value_info_by_name(onnx_model, sigNode.output[0])
    onnx_model = delete_value_info_by_name(onnx_model, splitNode.output[0])
    onnx_model = delete_value_info_by_name(onnx_model, splitNode.output[1])
    onnx_model = delete_nodes(onnx_model, [topRSNode] + botNextRSNodesList)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_moveForwardTranspose(onnx_model, node, node_index):
    act_types = ['Relu', 'Elu', 'PRelu', 'sigomid', 'tanh']
    axis_types = ['ReduceMean', 'ReduceSum', 'Softmax']
    cal_types = ['Mul', 'Add', 'Div', 'Sub']
    conv_types = ['Conv', 'ConvTranspose']
    attr_types = ['Resize', 'Split']
    if node.op_type != 'Transpose':
        return onnx_model, False
    tpOutShape = get_shape_by_name(onnx_model, node.output[0])
    if len(tpOutShape) != 4 or tpOutShape[-2] >= tpOutShape[-1]:
        return onnx_model, False
    tpPerm = attribute_to_dict(node.attribute).get('perm', list(range(len(tpOutShape))).reverse())
    orderPerm = list(range(len(tpOutShape)))
    if tpPerm != orderPerm[:-2] + [orderPerm[-1], orderPerm[-2]]:
        return onnx_model, False
    preNode = get_node_by_output(onnx_model, node.input[0])
    if preNode is None:
        return onnx_model, False
    paralNodesList = get_node_by_input(onnx_model, preNode.output)
    if len(paralNodesList) > 1:
        return onnx_model, False
    if preNode.op_type in act_types:
        newPreNode = copy.deepcopy(preNode)
        newPreNode.input[0] = preNode.output[0]
        newPreNode.output[0] = node.output[0]
        node.input[0] = preNode.input[0]
        node.output[0] = preNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
        preNodeInShape = get_shape_by_name(onnx_model, preNode.input[0])
        newOutShape = [preNodeInShape[idx] for idx in tpPerm]
        newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newOutShape)
        onnx_model.graph.value_info.append(newValueInfo)
        onnx_model.graph.node.remove(preNode)
        onnx_model.graph.node.insert(node_index, newPreNode)
        return onnx_model, True
    elif preNode.op_type in cal_types:
        if not find_init_by_name(onnx_model, preNode.input[0]) and not find_init_by_name(onnx_model, preNode.input[1]):
            return onnx_model, False
        stInput = preNode.input[1] if find_init_by_name(onnx_model, preNode.input[1]) else preNode.input[0]
        stArr = get_tensor_from_initializer(onnx_model, stInput)
        newStInput = stInput
        if stArr.size != 1:
            while len(stArr.shape) < len(tpOutShape): stArr = np.expand_dims(stArr, axis=0)
            newStArr = np.transpose(stArr, tuple(tpPerm))
            newStTensor = get_initial_by_value(onnx_model, newStArr)
            if newStTensor is None:
                newStTensor = onnx.helper.make_tensor(name=stInput+'_new',
                                                      data_type=NPDTYPE_2_ONNXDTYPE[newStArr.dtype],
                                                      dims=newStArr.shape,
                                                      vals=newStArr.flatten().tolist())
                onnx_model.graph.initializer.append(newStTensor)
            newStInput = newStTensor.name
        newPreNode = copy.deepcopy(preNode)
        newPreNode.input[list(newPreNode.input).index(stInput)] = newStInput
        newPreNode.input[list(newPreNode.input).index(preNode.input[0] if stInput == preNode.input[1] else preNode.input[1])] = preNode.output[0]
        newPreNode.output[0] = node.output[0]
        node.input[0] = preNode.input[0]
        node.output[0] = preNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
        preNodeInShape = get_shape_by_name(onnx_model, preNode.input[0] if stInput == preNode.input[1] else preNode.input[1])
        newOutShape = [preNodeInShape[idx] for idx in tpPerm]
        newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newOutShape)
        onnx_model.graph.value_info.append(newValueInfo)
        onnx_model.graph.node.remove(preNode)
        onnx_model.graph.node.insert(node_index, newPreNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    elif preNode.op_type in conv_types:
        if not find_init_by_name(onnx_model, preNode.input[1]):
            return onnx_model, False
        else:
            wtArr = get_tensor_from_initializer(onnx_model, preNode.input[1])
            convAttr = attribute_to_dict(preNode.attribute)
            kernel_shape = convAttr.get('kernel_shape')
            pads = convAttr.get('pads')
            strides = convAttr.get('strides')
            dilations = convAttr.get('dilations')
            if kernel_shape is not None:
                convAttr['kernel_shape'] = [kernel_shape[1], kernel_shape[0]]
            if pads is not None:
                convAttr['pads'] = [pads[1], pads[0], pads[3], pads[2]]
            if strides is not None:
                convAttr['strides'] = [strides[1], strides[0]]
            if dilations is not None:
                convAttr['dilations'] = [dilations[1], dilations[0]]
            newWtArr = np.transpose(wtArr, tuple(tpPerm))
            newWtTensor = get_initial_by_value(onnx_model, newWtArr)
            if newWtTensor is None:
                newWtTensor = onnx.helper.make_tensor(name=preNode.input[1]+'_new',
                                                      data_type=NPDTYPE_2_ONNXDTYPE[newWtArr.dtype],
                                                      dims=newWtArr.shape,
                                                      vals=newWtArr.flatten().tolist())
                onnx_model.graph.initializer.append(newWtTensor)
            newPreNodeInputs = [preNode.output[0], newWtTensor.name]
            if len(preNode.input) >= 3:
                newPreNodeInputs.append(preNode.input[2])
            newPreNode = onnx.helper.make_node(name=preNode.name,
                                               op_type=preNode.op_type,
                                               inputs=newPreNodeInputs,
                                               outputs=[node.output[0]],
                                               **convAttr)
            node.input[0] = preNode.input[0]
            node.output[0] = preNode.output[0]
            onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
            preNodeInShape = get_shape_by_name(onnx_model, preNode.input[0])
            newOutShape = [preNodeInShape[idx] for idx in tpPerm]
            newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newOutShape)
            onnx_model.graph.value_info.append(newValueInfo)
            onnx_model.graph.node.remove(preNode)
            onnx_model.graph.node.insert(node_index, newPreNode)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
    elif preNode.op_type in axis_types:
        preAttr = attribute_to_dict(preNode.attribute)
        axis_key = 'axes'
        axis = preAttr.get('axes')
        if axis is None:
            axis = preAttr.get('axis')
            axis_key = 'axis'
        if axis is not None:
            if isinstance(axis, list):
                for sgId, sgAxis in enumerate(axis):
                    posAxis = len(tpOutShape) + sgAxis if sgAxis < 0 else sgAxis
                    if posAxis == len(tpOutShape) - 1:
                        axis[sgId] = posAxis - 1
                    elif posAxis == len(tpOutShape) - 2:
                        axis[sgId] = posAxis + 1
                    else:
                        axis[sgId] = posAxis
            else:
                posAxis = len(tpOutShape) + axis if axis < 0 else axis
                if posAxis == len(tpOutShape) - 1:
                    axis = posAxis - 1
                elif posAxis == len(tpOutShape) - 2:
                    axis = posAxis + 1
                else:
                    axis = posAxis
            preAttr[axis_key] = axis
        newPreNode = onnx.helper.make_node(name=preNode.name,
                                           op_type=preNode.op_type,
                                           inputs=[preNode.output[0]],
                                           outputs=[node.output[0]],
                                           **preAttr)
        node.input[0] = preNode.input[0]
        node.output[0] = preNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
        preNodeInShape = get_shape_by_name(onnx_model, preNode.input[0])
        newOutShape = [preNodeInShape[idx] for idx in tpPerm]
        newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newOutShape)
        onnx_model.graph.value_info.append(newValueInfo)
        onnx_model.graph.node.remove(preNode)
        onnx_model.graph.node.insert(node_index, newPreNode)
        return onnx_model, True
    elif preNode.op_type in attr_types:
        return onnx_model, False
    else:
        return onnx_model, False
    
@OnnxDebuggerMeet.opt_convert_wrapper
def opt_moveForwardUnsqueeze(onnx_model, node, node_index):
    act_types = ['Relu', 'Elu', 'PRelu', 'sigomid', 'tanh']
    axis_types = ['ReduceMean', 'ReduceSum', 'Softmax']
    cal_types = ['Mul', 'Add', 'Div', 'Sub']
    if node.op_type != 'Unsqueeze':
        return onnx_model, False
    nodeOutShape = get_shape_by_name(onnx_model, node.output[0])
    nodeInShape = get_shape_by_name(onnx_model, node.input[0])
    if len(nodeOutShape) != 4:
        return onnx_model, False
    paralNodesList = get_node_by_input(onnx_model, [node.input[0]])
    if nodeInShape == nodeOutShape:
        for paralNode in paralNodesList:
            for curIdx, paralInput in enumerate(paralNode.input):
                paralNode.input[curIdx] = node.output[0] if paralInput == node.output[0] else paralInput
        onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
        preNode.output[0] = node.output[0]
        onnx_model.graph.node.remove(node)
        return onnx_model, True
    preNode = get_node_by_output(onnx_model, node.input[0])
    if preNode is None:
        netInputNames = [netInput.name for netInput in onnx_model.graph.input]
        if node.input[0] in netInputNames:
            if len(paralNodesList) == 1:
                netInputDtype = get_dtype_by_name(onnx_model, node.input[0])
                newNetInput = onnx.helper.make_tensor_value_info(node.input[0], netInputDtype, nodeOutShape)
                netInputId = netInputNames.index(node.input[0])
                del onnx_model.graph.input[netInputId]
                onnx_model.graph.input.insert(netInputId, newNetInput)
                nextNodesList = get_node_by_input(onnx_model, node.output)
                for nextNode in nextNodesList:
                    for curId, nextInput in enumerate(nextNode.input):
                        nextNode.input[curId] = newNetInput.name if nextInput == node.output[0] else nextInput
                onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
                onnx_model.graph.node.remove(node)
            else:
                newShapeTensor = get_initial_by_value(onnx_model, np.array(nodeOutShape, dtype=np.int64))
                if newShapeTensor is None:
                    newShapeTensor = onnx.helper.make_tensor(name=node.name+'_toReshape_shape',
                                                             data_type=TensorProto.INT64,
                                                             dims=[len(nodeOutShape)],
                                                             vals=nodeOutShape)
                    onnx_model.graph.initializer.append(newShapeTensor)
                newRSNode = onnx.helper.make_node(name=node.name+'_toReshape',
                                                  op_type='Reshape',
                                                  inputs=[node.input[0], newShapeTensor],
                                                  outputs=[node.output[0]])
                onnx_model.graph.node.remove(node)
                onnx_model.graph.node.insert(node_index, newRSNode)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        else:
            return onnx_model, False
    if len(paralNodesList) > 1:
        return onnx_model, False
    if onnx_model.opset_import[0].version > 11:
        if not find_init_by_name(onnx_model, node.input[1]):
            return onnx_model, False
        axes_list = get_tensor_from_initializer(onnx_model, node.input[1])
    else:
        axes_list = attribute_to_dict(node.attribute).get('axes')
    if preNode.op_type in act_types:
        newPreNode = copy.deepcopy(preNode)
        newPreNode.input[0] = preNode.output[0]
        newPreNode.output[0] = node.output[0]
        node.input[0] = preNode.input[0]
        node.output[0] = preNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
        preNodeInShape = get_shape_by_name(onnx_model, preNode.input[0])
        newOutShape = copy.deepcopy(preNodeInShape)
        for axes in axes_list:
            newOutShape.insert(axes, 1)
        newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newOutShape)
        onnx_model.graph.value_info.append(newValueInfo)
        onnx_model.graph.node.remove(preNode)
        onnx_model.graph.node.insert(newPreNode)
        return onnx_model, True
    elif preNode.op_type in cal_types:
        if not find_init_by_name(onnx_model, preNode.input[0]) and not find_init_by_name(onnx_model, preNode.input[1]):
            return onnx_model, False
        stInput = preNode.input[0] if find_init_by_name(onnx_model, preNode.input[0]) else preNode.input[1]
        stArr = get_tensor_from_initializer(onnx_model, stInput)
        newCalNode = copy.deepcopy(preNode)
        if stArr.size == 1:
            while len(stArr.shape) < len(nodeInShape): stArr = np.expand_dims(stArr, axis=0)
            for axes in axes_list:
                stArr = np.expand_dims(stArr, axis=axes)
            newStTensor = get_initial_by_value(onnx_model, stArr)
            if newStTensor is None:
                newStTensor = onnx.helper.make_tensor(name=stInput+'_new',
                                                      data_type=NPDTYPE_2_ONNXDTYPE[stArr.dtype],
                                                      dims=stArr.shape,
                                                      vals=stArr.flatten().tolist())
                onnx_model.graph.initializer.append(newStTensor)
            newCalNode.input[list(newCalNode.input).index(stInput)] = newStTensor.name
        newCalNode.input[list(newCalNode.input).index(preNode.input[0] if stInput == preNode.input[1] else preNode.input[1])] = preNode.output[0]
        newCalNode.output[0] = node.output[0]
        node.input[0] = preNode.input[1] if preNode.input[0] == stInput else preNode.input[0]
        node.output[0] = preNode.output[0]
        onnx_model = delete_value_info_by_name(onnx_model, preNode.output[0])
        preNodeInShape = get_shape_by_name(onnx_model, preNode.input[0] if stInput == preNode.input[1] else preNode.input[1])
        newOutShape = copy.deepcopy(preNodeInShape)
        for axes in axes_list:
            newOutShape.insert(axes, 1)
        newValueInfo = onnx.helper.make_tensor_value_info(node.output[0], 1, newOutShape)
        onnx_model.graph.value_info.append(newValueInfo)
        onnx_model.graph.node.remove(preNode)
        onnx_model.graph.node.insert(node_index, newCalNode)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    elif preNode.op_type in axis_types:
        return onnx_model, False
    else:
        return onnx_model, False
    
@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceReshapeReduceMean2Conv(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Reshape', 'ReduceMean']):
        rsNode, rmNode = get_node_serial_group(onnx_model, node, ['Reshape', 'ReduceMean'])
        rsInShape = get_shape_by_name(onnx_model, rsNode.input[0])
        rsOutShape = get_shape_by_name(onnx_model, rsNode.output[0])
        if len(rsInShape) != 4 or len(rsOutShape) != 5 or rsInShape[0] != rsOutShape[0] \
            or rsInShape[-2:] != rsOutShape[-2:] or rsInShape[1] != rsOutShape[1] * rsOutShape[2]:
            return onnx_model, False
        rmAxesList = attribute_to_dict(rmNode.attribute).get('axes')
        if rmAxesList is None or len(rmAxesList) != 1:
            return onnx_model, False
        rmAxes = rmAxesList[0]
        rmPosAxes = len(rsOutShape) + rmAxes if rmAxes < 0 else rmAxes
        if rmPosAxes != 2:
            return onnx_model, False
        keepDim = attribute_to_dict(rmNode.attribute).get('keepdims', 0)
        wtValue = np.array([1. / rsOutShape[2]] * rsOutShape[2], dtype=np.float32)
        wtZeroArr = np.zeros((rsOutShape[1], rsInShape[1], 1, 1), dtype=np.float32)
        #wtZeroArr = np.zeros((rsOutShape[1], rsOutShape[2], 1, 1), dtype=np.float32)
        for wtOutC in range(wtZeroArr.shape[0]):
            wtZeroArr[wtOutC, wtOutC*rsOutShape[2]:(wtOutC+1)*rsOutShape[2], :, :] = wtValue.reshape(rsOutShape[2], 1, 1)
            #wtZeroArr[wtOutC, :rsOutShape[2], :, :] = wtValue.reshape(rsOutShape[2], 1, 1)
        conv_attr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        #conv_attr = {'dilations': [1, 1], 'group': rsOutShape[1], 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        convWtTensor = get_initial_by_value(onnx_model, wtZeroArr)
        if convWtTensor is None:
            convWtTensor = onnx.helper.make_tensor(name=rmNode.name+'_wt',
                                                   data_type=NPDTYPE_2_ONNXDTYPE[wtZeroArr.dtype],
                                                   dims=wtZeroArr.shape,
                                                   vals=wtZeroArr.flatten().tolist())
            onnx_model.graph.initializer.append(convWtTensor)
        newConvNode = onnx.helper.make_node(name=rmNode.name+'_toConv',
                                            op_type='Conv',
                                            inputs=[rsNode.input[0], convWtTensor.name],
                                            outputs=rmNode.output,
                                            **conv_attr)
        if keepDim == 1:
            rmOutShape = get_shape_by_name(onnx_model, rmNode.output[0])
            newShapeTensor = get_initial_by_value(onnx_model, np.array(rmOutShape, dtype=np.int64))
            if newShapeTensor is None:
                newShapeTensor = onnx.helper.make_tensor(name=rmNode.output[0]+'_newShape',
                                                         data_type=TensorProto.INT64,
                                                         dims=[len(rmOutShape)],
                                                         vals=rmOutShape)
                onnx_model.graph.initializer.append(newShapeTensor)
            newRSNode = onnx.helper.make_node(name=rmNode.name+'_toConv'+'_reshape',
                                              op_type='Reshape',
                                              inputs=[rmNode.output[0]+'_4d', newShapeTensor],
                                              outputs=[rmNode.output[0]])
            newConvNode.output[0] = newRSNode.input[0]
            onnx_model.graph.node.insert(node_index, newConvNode)
        onnx_model.graph.node.insert(node_index, newConvNode)
        onnx_model = delete_value_info_by_name(onnx_model, rsNode.output[0])
        onnx_model = delete_nodes(onnx_model, [rmNode, rsNode])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_replaceWhere(onnx_model, node, node_index):
    if node.op_type == 'Where':
        repInput0 = node.input[1]
        repInput1 = node.input[2]
        repInNode0 = get_node_by_output(onnx_model, repInput0)
        repInNode1 = get_node_by_output(onnx_model, repInput1)
        if (repInNode0 is not None and repInNode1 is not None) \
            or (repInNode0 is None and repInNode1 is None):
            return onnx_model, False
        repStInput, repDyInput = [node.input[1], node.input[2]] if repInNode0 is None else [node.input[2], node.input[1]]
        repInArr = get_tensor_from_initializer(onnx_model, repStInput)
        if repInArr.size != 1 or repInArr not in [0, np.array(-np.inf), np.array(np.inf)]:
            return onnx_model, False
        conditionInput = node.input[0]
        if not find_init_by_name(onnx_model, conditionInput):
            return onnx_model, False
        outNodesList = get_node_by_input(onnx_model, node.output)
        conditionArr = get_tensor_from_initializer(onnx_model, conditionInput)
        posBoolArr = np.reshape(np.array([True] * conditionArr.size), conditionArr.shape)
        negBoolArr = np.reshape(np.array([False] * conditionArr.size), conditionArr.shape)
        if (conditionArr == posBoolArr).all():
            if repInNode0 is None:
                return onnx_model, False
            for outNode in outNodesList:
                for idx, outNodeInput in enumerate(outNode.input):
                    outNode.input[idx] == repDyInput if outNodeInput == node.output[0] else outNodeInput
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
        elif (conditionArr == negBoolArr).all():
            if repInNode1 is None:
                return onnx_model, False
            for outNode in outNodesList:
                for idx, outNodeInput in enumerate(outNode.input):
                    outNode.input[idx] = repDyInput if outNodeInput == node.output[0] else outNodeInput
            onnx_model = delete_value_info_by_name(onnx_model, node.output[0])
        else:
            conditionNegArr = ~conditionArr
            conditionFPArr = conditionNegArr.astype(np.float32)
            if repInArr.size == 1:
                # rep_val = 10e9 if repInArr == np.inf else repInArr
                # rep_val = -10e9 if rep_val == -np.inf else rep_val
                conditionFP1DArr = conditionFPArr.flatten()
                for idx in range(conditionFP1DArr.size):
                    conditionFP1DArr[idx] = repInArr if conditionFP1DArr[idx] == 0 else conditionFP1DArr[idx]
                conditionFPArr = np.reshape(conditionFP1DArr, conditionFPArr.shape)
                fpWhereTensor = get_initial_by_value(onnx_model, conditionFPArr)
                if fpWhereTensor is None:
                    fpWhereTensor = onnx.helper.make_tensor(name=node.input[0]+'_fp32',
                                                            data_type=TensorProto.FLOAT,
                                                            dims=conditionFPArr.shape,
                                                            vals=conditionFPArr.flatten().tolist())
                    onnx_model.graph.initializer.append(fpWhereTensor)
                newMulNode = onnx.helper.make_node(name=node.name+'_toMul',
                                                op_type='Mul',
                                                inputs=[repDyInput, fpWhereTensor.name],
                                                outputs=node.output)
                onnx_model.graph.node.insert(node_index, newMulNode)
            else:
                return onnx_model, False
        onnx_model.graph.node.remove(node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeReshapeTransposeReshapeOrReverse(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Transpose', 'Reshape']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape', 'Transpose', 'Reshape'])
        topTPNode, topRSNode, botTPNode, botRSNode = serial_nodes
        topTPInShape = get_shape_by_name(onnx_model, topTPNode.input[0])
        botRSOutShape = get_shape_by_name(onnx_model, botRSNode.output[0])
        if topTPInShape != botRSOutShape:
            return onnx_model, False
        data_size = int(np.prod(np.array(topTPInShape, dtype=np.int64)))
        initInArr = np.random.randn(data_size).reshape(tuple(topTPInShape))
        topPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topTPInShape))).reverse())
        topTPOutSpArr = np.transpose(initInArr, tuple(topPerm))
        topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
        topRSOutSpArr = np.reshape(topTPOutSpArr, tuple(topRSOutShape))
        botPerm = attribute_to_dict(botTPNode.attribute).get('perm', list(range(len(topRSOutShape))).reverse())
        botTPOutSpArr = np.transpose(topRSOutSpArr, tuple(botPerm))
        botRSOutSpArr = np.reshape(botTPOutSpArr, tuple(botRSOutShape))
        if not (botRSOutSpArr == initInArr).all():
            return onnx_model, False
        botRSOutNodesList = get_node_by_input(onnx_model, botRSNode.output)
        for botRSOutNode in botRSOutNodesList:
            for idx, botRSOutNodeInput in enumerate(botRSOutNode.input):
                botRSOutNode.input[idx] = topTPNode.input[0] if botRSOutNodeInput == botRSNode.output[0] else botRSOutNodeInput
        serial_nodes.reverse()
        for delNode in serial_nodes:
            delOutNodesList = get_node_by_input(onnx_model, delNode.output)
            if not delOutNodesList:
                onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
                onnx_model.graph.node.remove(delNode)
            else:
                break
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    elif check_node_serial_group(onnx_model, node, ['Reshape', 'Transpose', 'Reshape', 'Transpose']):
        serial_nodes = get_node_serial_group(onnx_model, node, ['Reshape', 'Transpose', 'Reshape', 'Transpose'])
        topRSNode, topTPNode, botRSNode, botTPNode = serial_nodes
        topRSInShape = get_shape_by_name(onnx_model, topRSNode.input[0])
        botTPOutShape = get_shape_by_name(onnx_model, botTPNode.output[0])
        if topRSInShape != botTPOutShape:
            return onnx_model, False
        data_size = int(np.prod(np.array(topRSInShape, dtype=np.int64)))
        initInArr = np.random.randn(data_size).reshape(tuple(topRSInShape))
        topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
        topRSOutSpArr = np.reshape(initInArr, tuple(topRSOutShape))
        topPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topRSOutShape))).reverse())
        topTPOutSpArr = np.transpose(topRSOutSpArr, tuple(topPerm))
        botRSOutShape = get_shape_by_name(onnx_model, botRSNode.output[0])
        botRSOutSpArr = np.reshape(topTPOutSpArr, tuple(botRSOutShape))
        botPerm = attribute_to_dict(botTPNode.attribute).get('perm', list(range(len(botRSOutShape))).reverse())
        botTPOutSpArr = np.transpose(botRSOutSpArr, tuple(botPerm))
        if not (initInArr == botTPOutSpArr).all():
            return onnx_model, False
        botTPOutNodesList = get_node_by_input(onnx_model, botTPNode.output)
        for botTPOutNode in botTPOutNodesList:
            for idx, botTPOutNodeInput in enumerate(botTPOutNode.input):
                botTPOutNode.input[idx] = topRSNode.input[0] if botTPOutNodeInput == botTPNode.output[0] else botTPOutNodeInput
        serial_nodes.reverse()
        for delNode in serial_nodes:
            delOutNodesList = get_node_by_input(onnx_model, delNode.output)
            if not delOutNodesList:
                onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
                onnx_model.graph.node.remove(delNode)
            else:
                break
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertSliceConcatTranspose2ConcatTransposeConv(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Concat', 'Transpose']):
        concatNode, tpNode = get_node_serial_group(onnx_model, node, ['Concat', 'Transpose'])
        newCtInputs = []
        init_slice_axes = None
        delNodesList = []
        for idx, ctInput in enumerate(concatNode.input):
            pre_node = get_node_by_output(onnx_model, ctInput)
            if pre_node.op_type != 'Slice' or len(pre_node.input) < 5:
                return onnx_model, False
            preInShape = get_shape_by_name(onnx_model, pre_node.input[0])
            if len(preInShape) != 4:
                return onnx_model, False
            stepsList = get_tensor_from_initializer(onnx_model, pre_node.input[4]).tolist()
            if len(stepsList) != 1 or stepsList[0] != -1:
                return onnx_model, False
            axesList = get_tensor_from_initializer(onnx_model, pre_node.input[3]).tolist()
            if len(axesList) != 1:
                return onnx_model, False
            axesPos = axesList[0] + len(preInShape) if axesList[0] < 0 else axesList[0]
            if init_slice_axes is None:
                init_slice_axes = axesPos
            elif init_slice_axes != axesPos:
                return onnx_model, False
            startsList = get_tensor_from_initializer(onnx_model, pre_node.input[1]).tolist()
            if len(startsList) != 1 or startsList[0] not in [-1, preInShape[axesPos]]:
                return onnx_model, False
            endsList = get_tensor_from_initializer(onnx_model, pre_node.input[2]).tolist()
            if len(endsList) != 1 or (endsList[0] != 0 and endsList[0] > preInShape[axesPos] * -1):
                return onnx_model, False
            newCtInputs.append(pre_node.input[0])
            delNodesList.append(pre_node)
        if init_slice_axes is None:
            return onnx_model, False
        newPerm = [0, init_slice_axes, 1, 2 if init_slice_axes == 3 else 3] \
            if init_slice_axes != 1 else list(range(4))
        tpInShape = get_shape_by_name(onnx_model, tpNode.input[0])
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(tpInShape))).reverse())
        if newPerm != tpPerm:
            return onnx_model, False
        tpOutShape = [tpInShape[index] for index in tpPerm]
        
        newCtNode = onnx.helper.make_node(name=concatNode.name,
                                          op_type='Concat',
                                          inputs=newCtInputs,
                                          outputs=concatNode.output,
                                          axis=attribute_to_dict(concatNode.attribute).get('axis'))
        
        conv_attr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        wt_arr = np.eye(tpOutShape[1], dtype=np.float32)[::-1].reshape(tpOutShape[1], tpOutShape[1], 1, 1)
        wt_tensor = get_initial_by_value(onnx_model, wt_arr)
        if wt_tensor is None:
            wt_tensor = onnx.helper.make_tensor(name='Slice_'+ concatNode.name +'_wt',
                                                data_type=NPDTYPE_2_ONNXDTYPE[wt_arr.dtype],
                                                dims=wt_arr.shape,
                                                vals=wt_arr.flatten().tolist())
            onnx_model.graph.initializer.append(wt_tensor)
        newConvNode = onnx.helper.make_node(name='Slice_' + concatNode.name + '_toConv',
                                            op_type='Conv',
                                            inputs=[tpNode.output[0], wt_tensor.name],
                                            outputs=['Slice_' + concatNode.name + '_toConv_output'],
                                            **conv_attr)
        
        nextNodesList = get_node_by_input(onnx_model, tpNode.output)
        
        for nextNode in nextNodesList:
            for id, nextInput in enumerate(nextNode.input):
                nextNode.input[id] = newConvNode.output[0] if nextInput == tpNode.output[0] else nextInput
                
        tp_id = get_node_id(onnx_model, tpNode)
        onnx_model.graph.node.insert(tp_id+1, newConvNode)
        onnx_model.graph.node.remove(concatNode)
        onnx_model.graph.node.insert(node_index, newCtNode)
        for delNode in delNodesList:
            onnx_model = delete_value_info_by_name(onnx_model, delNode.output[0])
            onnx_model.graph.node.remove(delNode)
        
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertMatMulPadSliceTranspose2ConvConvSliceConv(onnx_model, node, node_index):
    if node.op_type == 'MatMul':
        if find_init_by_name(onnx_model, node.input[0]) or find_init_by_name(onnx_model, node.input[0]):
            return onnx_model, False
        
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionReshapeTransposeReshapeToReshapeTranspose(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Reshape', 'Transpose', 'Reshape']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Reshape', 'Transpose', 'Reshape'])
        topRSNode, tpNode, botRSNode = nodes_serial
        topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
        botRSInShape = get_shape_by_name(onnx_model, botRSNode.input[0])
        botRSOutShape = get_shape_by_name(onnx_model, botRSNode.output[0])
        if len(topRSOutShape) == 3 and len(botRSOutShape) == 4 and 1 in botRSOutShape[1:] \
            and botRSInShape[1] in botRSOutShape[1:] and botRSInShape[2] in botRSOutShape[1:]:
            tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(botRSInShape))).reverse())
            if tpPerm == [0, 1, 2]:
                botRSNode.input[0] = topRSNode.input[0]
                for cur_node in [topRSNode, tpNode]:
                    curOutNodes = get_node_by_input(onnx_model, cur_node.output)
                    if not curOutNodes:
                        onnx_model = delete_value_info_by_name(onnx_model, cur_node.output[0])
                        onnx_model.graph.node.remove(cur_node)
                return onnx_model, True
            elif tpPerm[0] != 0:
                return onnx_model, False
            newRSOutShape = topRSOutShape[:2] + [1, topRSOutShape[2]]
            newPerm = []
            for newPerm in [[0, 3, 2, 1], [0, 3, 1, 2], [0, 2, 3, 1]]:
                newTPOutShape = [newRSOutShape[p] for p in newPerm]
                if newTPOutShape == botRSOutShape:
                    newPerm = newPerm
                    break
            if not newPerm:
                return onnx_model, False
            newRSOutShapeTensor = get_initial_by_value(onnx_model, np.array(newRSOutShape, dtype=np.int64))
            if newRSOutShapeTensor is None:
                newRSOutShapeTensor = onnx.helper.make_tensor(name=topRSNode.input[1] + '_new_' + topRSNode.name,
                                                              data_type=TensorProto.INT64,
                                                              dims=[4],
                                                              vals=newRSOutShape)
                onnx_model.graph.initializer.append(newRSOutShapeTensor)
            topRSNode.input[1] = newRSOutShapeTensor.name
            del tpNode.attribute[:]
            newPermAttr = onnx.helper.make_attribute('perm', newPerm)
            tpNode.attribute.append(newPermAttr)
            tpNode.output[0] = botRSNode.output[0]
            tpInDType = get_dtype_by_name(onnx_model, tpNode.input[0])
            onnx_model = delete_value_info_by_name(onnx_model, botRSNode.input[0])
            onnx_model = delete_value_info_by_name(onnx_model, tpNode.input[0])
            newTPInValueInfo = onnx.helper.make_tensor_value_info(tpNode.input[0], tpInDType, newRSOutShape)
            onnx_model.graph.value_info.append(newTPInValueInfo)
            onnx_model.graph.node.remove(botRSNode)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertW1TransH1WithTransposeSoftmaxTransposeReshape(onnx_model, node, node_index):
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Softmax', 'Transpose', 'Reshape']):
        nodes_serial = get_node_serial_group(onnx_model, node, ['Transpose', 'Softmax', 'Transpose', 'Reshape'])
        tpNode1, softmaxNode, tpNode2, rsNode = nodes_serial
        tp1OutShape = get_shape_by_name(onnx_model, tpNode1.output[0])
        if tp1OutShape[-1] != 1 or len(tp1OutShape) != 4:
            return onnx_model, False
        tpPerm1 = attribute_to_dict(tpNode1.attribute).get('perm', list(range(len(tp1OutShape))).reverse())
        tpPerm2 = attribute_to_dict(tpNode2.attribute).get('perm', list(range(len(tp1OutShape))).reverse())
        softmaxAxis = attribute_to_dict(softmaxNode.attribute).get('axis', 1)
        softmaxPosAxis = softmaxAxis if softmaxAxis >= 0 else len(tp1OutShape) + softmaxAxis
        newPerm1 = [tpPerm1[p] for p in [0, 1, 3, 2]]
        newSoftmaxAxis = [0, 1, 3, 2].index(softmaxPosAxis)
        del tpNode1.attribute[:]
        newPerm1Attr = onnx.helper.make_attribute('perm', newPerm1)
        tpNode1.attribute.append(newPerm1Attr)
        del softmaxNode.attribute[:]
        newAxisAttr = onnx.helper.make_attribute('axis', newSoftmaxAxis)
        softmaxNode.attribute.append(newAxisAttr)
        newPerm2 = [[0, 1, 3, 2][p] for p in tpPerm2]
        del tpNode2.attribute[:]
        newPerm2Attr = onnx.helper.make_attribute('perm', newPerm2)
        tpNode2.attribute.append(newPerm2Attr)
        curDtype = get_dtype_by_name(onnx_model, tpNode1.input[0])
        onnx_model = delete_value_info_by_name(onnx_model, tpNode1.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, softmaxNode.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, tpNode2.output[0])
        newTP1OutShape = [tp1OutShape[idx] for idx in newPerm1]
        newSoftmaxOutShape = newTP1OutShape
        newTP2OutShape = [newSoftmaxOutShape[idx] for idx in newPerm2]
        newTP1OutTensor = onnx.helper.make_tensor_value_info(tpNode1.output[0], curDtype, newTP1OutShape)
        newSoftmaxOutTensor = onnx.helper.make_tensor_value_info(softmaxNode.output[0], curDtype, newSoftmaxOutShape)
        newTP2OutTensor = onnx.helper.make_tensor_value_info(tpNode2.output[0], curDtype, newTP2OutShape)
        onnx_model.graph.value_info.append(newTP1OutTensor)
        onnx_model.graph.value_info.append(newSoftmaxOutTensor)
        onnx_model.graph.value_info.append(newTP2OutTensor)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeReshapeReduceSumMulReduceSumMulToConv(onnx_model, node, node_index):
    def convert_serial_nodes_conv(onnx_model, cur_node):
        nodes_serial = get_node_serial_group(onnx_model, cur_node, ['ReduceSum', 'Mul', 'ReduceSum'])
        reduceSum1, mul1, reduceSum2 = nodes_serial
        botNode = reduceSum2
        del_nodes_list = nodes_serial
        reduceSum1InShape = get_shape_by_name(onnx_model, reduceSum1.input[0])
        reduceSum1AttrDict = attribute_to_dict(reduceSum1.attribute)
        reduceSum1AxesList = reduceSum1AttrDict.get('axes')
        reduceSum1KeepDim = reduceSum1AttrDict.get('keepdims', 1)
        if reduceSum1AxesList is None or len(reduceSum1AxesList) != 1:
            return None
        reduceSum1Axes = reduceSum1AxesList[0]
        reduceSum1PosAxes = len(reduceSum1InShape) + reduceSum1Axes if reduceSum1Axes < 0 else reduceSum1Axes
        if reduceSum1PosAxes not in [len(reduceSum1InShape) - 2, len(reduceSum1InShape) - 1]:
            return None
        if not find_init_by_name(onnx_model, mul1.input[0]) and not find_init_by_name(onnx_model, mul1.input[1]):
            return None
        mul1StIn = mul1.input[0] if reduceSum1.output[0] == mul1.input[1] else mul1.input[1]
        mul1Arr = get_tensor_from_initializer(onnx_model, mul1StIn)
        reduceSum1OutShape = get_shape_by_name(onnx_model, reduceSum1.output[0])
        reduceSum1EndDim = reduceSum1OutShape[-2] \
            if reduceSum1KeepDim == 1 and reduceSum1PosAxes == len(reduceSum1InShape)-1 else reduceSum1OutShape[-1]
        if mul1Arr.size != 1:
            mul1ArrEndDim = mul1Arr.shape[-2] \
                if reduceSum1KeepDim == 1 and reduceSum1PosAxes == len(reduceSum1InShape)-1 else reduceSum1OutShape[-1]
            if np.prod(np.array(mul1Arr.shape)) != reduceSum1EndDim or mul1ArrEndDim != reduceSum1EndDim:
                return None
        reduceSum2AttrDict = attribute_to_dict(reduceSum2.attribute)
        reduceSum2AxesList = reduceSum2AttrDict.get('axes')
        if reduceSum2AxesList is None or len(reduceSum2AxesList) != 1:
            return None
        reudceSum2Axes =  reduceSum2AxesList[0]
        reduceSum2PosAxes = len(reduceSum1OutShape) + reudceSum2Axes if reudceSum2Axes < 0 else reudceSum2Axes
        if (reduceSum1KeepDim == 0 and reduceSum2PosAxes != len(reduceSum1OutShape) - 1) \
            or (reduceSum1KeepDim == 1 and reduceSum2PosAxes*reduceSum1PosAxes != (len(reduceSum1InShape)-1)*(len(reduceSum1InShape)-2)):
                return None
        conv_wt_arr = np.ones((reduceSum1InShape[-2], reduceSum1InShape[-1]), dtype=np.float32)
        conv_wt_arr *= mul1Arr.reshape(-1, 1) if reduceSum1PosAxes == len(reduceSum1InShape) - 1 else mul1Arr.reshape(1, -1)
        
        init_bais_arr = np.array([0], dtype=np.float32)
        outNodesList = get_node_by_input(onnx_model, reduceSum2.output)
        if len(outNodesList) != 1 or outNodesList[0].op_type not in ['Mul', 'Add', 'Sub']:
            return [botNode, del_nodes_list, conv_wt_arr, init_bais_arr]
        nextNode = outNodesList[0]
        if not find_init_by_name(onnx_model, nextNode.input[0]) and not find_init_by_name(onnx_model, nextNode.input[1]):
            return [botNode, del_nodes_list, conv_wt_arr, init_bais_arr]
        nextStIn = nextNode.input[0] if nextNode.input[1] == reduceSum2.output[0] else nextNode.input[1]
        nextArr = get_tensor_from_initializer(onnx_model, nextStIn)
        if nextArr.size != 1:
            return [botNode, del_nodes_list, conv_wt_arr, init_bais_arr]
        if nextNode.op_type == 'Mul':
            conv_wt_arr *= nextArr.reshape(1, -1) if reduceSum1PosAxes == len(reduceSum1InShape)-1 else nextArr.reshape(-1, 1)
            del_nodes_list.append(nextNode)
            botNode = nextNode
            outNodesList = get_node_by_input(onnx_model, nextNode.output)
            if len(outNodesList) != 1 or outNodesList[0].op_type not in ['Add', 'Sub']:
                return [botNode, del_nodes_list, conv_wt_arr, init_bais_arr]
            baisNode = outNodesList[0]
        else:
            baisNode = nextNode
        if baisNode.name != nextNode.name:
            baisStIn = baisNode.input[0] if find_init_by_name(onnx_model, baisNode.input[0]) else baisNode.input[1]
            if not find_init_by_name(onnx_model, baisStIn):
                return [botNode, del_nodes_list, conv_wt_arr, init_bais_arr]
            baisArr = get_tensor_from_initializer(onnx_model, baisStIn)
            if baisArr.size != 1:
                return [botNode, del_nodes_list, conv_wt_arr, init_bais_arr]
        else:
            baisStIn = nextStIn
            baisArr = nextArr
        baisArr = baisArr * (-1 if baisNode.op_type == 'Sub' and baisStIn == baisNode.input[1] else 1)
        if baisNode.op_type == 'Sub' and baisStIn == baisNode.input[0]:
            conv_wt_arr *= -1
        botNode = baisNode
        del_nodes_list.append(baisNode)
        conv_bais_arr = np.reshape(np.array(baisArr, dtype=np.float32), (1,))
        return [botNode, del_nodes_list, conv_wt_arr, conv_bais_arr] 
    
    if check_node_serial_group(onnx_model, node, ['Transpose', 'Reshape']):
        topTPNode, topRSNode = get_node_serial_group(onnx_model, node, ['Transpose', 'Reshape'])
        topRSOutShape = get_shape_by_name(onnx_model, topRSNode.output[0])
        topRSInShape = get_shape_by_name(onnx_model, topRSNode.input[0])
        if len(topRSInShape) != len(topRSOutShape) != 4 or topRSOutShape[:2] != topRSInShape[:2] or 1 not in topRSInShape[2:] \
            or topRSInShape[2]*topRSInShape[3] != topRSOutShape[2]*topRSOutShape[3]:
                return onnx_model, False
        topTPInShape = get_shape_by_name(onnx_model, topTPNode.input[0])
        if topTPInShape[1] != topRSInShape[2] * topRSInShape[3]:
            return onnx_model, False
        
        data_type = get_dtype_by_name(onnx_model, topTPNode.input[0])
        
        total_conv_wt_arr = None
        total_conv_bais_arr = None
        bot_nodes_list = []
        dst_rm_nodes_list = []
        topRSOutNodesList = get_node_by_input(onnx_model, topRSNode.output)
        for topRSOutNode in topRSOutNodesList:
            if check_node_serial_group(onnx_model, topRSOutNode, ['ReduceSum', 'Mul', 'ReduceSum']):
                search_infos_list = convert_serial_nodes_conv(onnx_model, topRSOutNode)
                if search_infos_list is None:
                    continue
                end_node, rm_nodes_list, conv_wt_arr, conv_bais_arr = search_infos_list
                conv_wt_arr = np.reshape(conv_wt_arr, (1, topTPInShape[1], 1, 1))
                total_conv_wt_arr = conv_wt_arr if total_conv_wt_arr is None \
                    else np.concatenate((total_conv_wt_arr, conv_wt_arr), axis=0)
                total_conv_bais_arr = conv_bais_arr if total_conv_bais_arr is None \
                    else np.concatenate((total_conv_bais_arr, conv_bais_arr))
                bot_nodes_list.append(end_node)
                dst_rm_nodes_list.append(rm_nodes_list)
        if total_conv_wt_arr is None:
            return onnx_model, False
        conv_wt_tensor = get_initial_by_value(onnx_model, total_conv_wt_arr)
        if conv_wt_tensor is None:
            conv_wt_tensor = onnx.helper.make_tensor(name=topRSNode.name+'_multiReduceSum_toConv_wt',
                                                     data_type=NPDTYPE_2_ONNXDTYPE[total_conv_wt_arr.dtype],
                                                     dims=total_conv_wt_arr.shape,
                                                     vals=total_conv_wt_arr.flatten().tolist())
            onnx_model.graph.initializer.append(conv_wt_tensor)
        conv_inputs = [topTPNode.input[0], conv_wt_tensor.name]
        if total_conv_bais_arr.any():
            conv_bais_tensor = get_initial_by_value(onnx_model, total_conv_bais_arr)
            if conv_bais_tensor is None:
                conv_bais_tensor = onnx.helper.make_tensor(name=topRSNode.name+'_multiReduceSum_toConv_bais',
                                                           data_type=NPDTYPE_2_ONNXDTYPE[total_conv_bais_arr.dtype],
                                                           dims=total_conv_bais_arr.shape,
                                                           vals=total_conv_bais_arr.flatten().tolist())
                onnx_model.graph.initializer.append(conv_bais_tensor)
            conv_inputs.append(conv_bais_tensor.name)
        conv_attr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        conv_node = onnx.helper.make_node(name=topRSNode.name+'_multiReduceSum_toConv',
                                          op_type='Conv',
                                          inputs=conv_inputs,
                                          outputs=[topRSNode.output[0]+'_ReduceSum2Conv'],
                                          **conv_attr)
        topTPPerm = attribute_to_dict(topTPNode.attribute).get('perm', list(range(len(topRSInShape))).reverse())
        conv_out_shape = [1, total_conv_wt_arr.shape[0], topTPInShape[-2], topTPInShape[-1]]
        conv_out_shape_value_info = onnx.helper.make_tensor_value_info(conv_node.output[0], data_type, conv_out_shape)
        onnx_model.graph.value_info.append(conv_out_shape_value_info)
        #newTPPerm = [0, 3, 2, 1] if topTPPerm == [0, 3, 1, 2] and conv_out_shape[2] == 1 else topTPPerm
        newTPNode = onnx.helper.make_node(name=topTPNode.name+'_new',
                                          op_type='Transpose',
                                          inputs=[conv_node.output[0]],
                                          outputs=[topTPNode.output[0]+'_ReduceSum2Conv'],
                                          perm=topTPPerm)
        newTPOutShape = [conv_out_shape[idx] for idx in topTPPerm]
        insert_nodes_list = []
        if len(bot_nodes_list) > 1:
            slice_start = 0
            slice_start_tensor = get_initial_by_value(onnx_model, np.array([slice_start], dtype=np.int64))
            if slice_start_tensor is None:
                slice_start_tensor = onnx.helper.make_tensor(name=newTPNode.output[0]+'_idx_0',
                                                            data_type=TensorProto.INT64,
                                                            dims=[1],
                                                            vals=[slice_start])
                onnx_model.graph.initializer.append(slice_start_tensor)
            slice_axes = topTPPerm.index(1)
            slice_axes_tensor = get_initial_by_value(onnx_model, np.array([slice_axes], dtype=np.int64))
            if slice_axes_tensor is None:
                slice_axes_tensor = onnx.helper.make_tensor(name=newTPNode.output[0]+'_param_0',
                                                            data_type=TensorProto.INT64,
                                                            dims=[1],
                                                            vals=[slice_axes])
                onnx_model.graph.initializer.append(slice_axes_tensor)
            slice_step = 1
            slice_step_tensor = get_initial_by_value(onnx_model, np.array([slice_step], dtype=np.int64))
            if slice_step_tensor is None:
                slice_step_tensor = onnx.helper.make_tensor(name=newTPNode.output[0]+'_param_1',
                                                            data_type=TensorProto.INT64,
                                                            dims=[1],
                                                            vals=[slice_step])
                onnx_model.graph.initializer.append(slice_step_tensor)
            for bix in range(conv_out_shape[1]):
                slice_end = slice_start + slice_step
                slice_end_tensor = get_initial_by_value(onnx_model, np.array([slice_end], dtype=np.float32))
                if slice_end_tensor is None:
                    slice_end_tensor = onnx.helper.make_tensor(name=newTPNode.output[0]+'_idx_%d'%(bix+1),
                                                            data_type=TensorProto.INT64,
                                                            dims=[1],
                                                            vals=[slice_end])
                    onnx_model.graph.initializer.append(slice_end_tensor)
                slice_inputs = [newTPNode.output[0], slice_start_tensor.name, slice_end_tensor.name, 
                                slice_axes_tensor.name, slice_step_tensor.name]
                slice_node = onnx.helper.make_node(name=newTPNode.output[0]+'_slice_%d'%bix,
                                                op_type='Slice',
                                                inputs=slice_inputs,
                                                outputs=[bot_nodes_list[bix].output[0]])
                insert_nodes_list.append(slice_node)
                slice_out_shape = copy.deepcopy(newTPOutShape)
                slice_out_shape[slice_axes] = 1
                src_bot_shape = get_shape_by_name(onnx_model, bot_nodes_list[bix].output[0])
                if src_bot_shape != slice_out_shape:
                    newSliceOutShapeTensor = get_initial_by_value(onnx_model, np.array(src_bot_shape, dtype=np.int64))
                    if newSliceOutShapeTensor is None:
                        newSliceOutShapeTensor = onnx.helper.make_tensor(name=bot_nodes_list[bix].output[0]+'_Shape',
                                                                        data_type=TensorProto.INT64,
                                                                        dims=[len(src_bot_shape)],
                                                                        vals=src_bot_shape)
                        onnx_model.graph.initializer.append(newSliceOutShapeTensor)
                    slice_node.output[0] = bot_nodes_list[bix].output[0]+'_4d'
                    insert_nodes_list[-1].output[0] = bot_nodes_list[bix].output[0]+'_4d'
                    sliceRSNode = onnx.helper.make_node(name=bot_nodes_list[bix].output[0]+'_squeeze',
                                                        op_type='Reshape',
                                                        inputs=[slice_node.output[0], newSliceOutShapeTensor.name],
                                                        outputs=[bot_nodes_list[bix].output[0]])
                    insert_nodes_list.append(sliceRSNode)
                    slice_new_shape_value_info = onnx.helper.make_tensor_value_info(slice_node.output[0], data_type, slice_out_shape)
                    onnx_model.graph.value_info.append(slice_new_shape_value_info)
                slice_start = slice_end
                slice_start_tensor = slice_end_tensor
        else:
            src_bot_shape = get_shape_by_name(onnx_model, bot_nodes_list[0].output[0])
            if src_bot_shape != newTPOutShape:
                newBotRSShapeTensor = get_initial_by_name(onnx_model, np.array(src_bot_shape, dtype=np.int64))
                if newBotRSShapeTensor is None:
                    newBotRSShapeTensor = onnx.helper.make_tensor(name=bot_nodes_list[0].output[0]+'_shape',
                                                                  data_type=TensorProto.INT64,
                                                                  dims=[len(src_bot_shape)],
                                                                  vals=src_bot_shape)
                    onnx_model.graph.initializer.append(newBotRSShapeTensor)
                newBotRSNode = onnx.helper.make_node(name=bot_nodes_list[0].output[0]+'_squeeze',
                                                     op_type='Reshape',
                                                     inputs=[newTPNode.output[0], newBotRSShapeTensor.name],
                                                     outputs=[bot_nodes_list[0].output[0]])
                insert_nodes_list.append(newBotRSNode)
                newTPOutShape_value_info = onnx.helper.make_tensor_value_info(newTPNode.output[0], data_type, newTPOutShape)
                onnx_model.graph.value_info.append(newTPOutShape_value_info)
            else:
                newTPNode.output[0] = bot_nodes_list[0].output[0]
        
        for del_nodes_sub_list in dst_rm_nodes_list:
            for del_node in del_nodes_sub_list:
                if del_node not in bot_nodes_list:
                    onnx_model = delete_value_info_by_name(onnx_model, del_node.output[0])
                onnx_model.graph.node.remove(del_node)
        insert_nodes_list.reverse()
        onnx_model = insert_node_by_list(onnx_model, insert_nodes_list + [newTPNode, conv_node], node_index)
        for top_node in [topRSNode, topTPNode]:
            topCurOutNodesList = get_node_by_input(onnx_model, top_node.output)
            if not topCurOutNodesList:
                onnx_model = delete_value_info_by_name(onnx_model, top_node.output[0])
                onnx_model.graph.node.remove(top_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_fusionTransposeSqueezeMulReduceSumMulToConv(onnx_model, node, node_index):
    def find_conv_bais_tensor(onnx_model, cur_node):
        next_nodes_list = get_node_by_input(onnx_model, cur_node.output)
        if len(next_nodes_list) != 1 or next_nodes_list[0].op_type not in ['Mul', 'Add', 'Sub']:
            return None
        next_node = next_nodes_list[0]
        if not find_init_by_name(onnx_model, next_node.input[0]) and not find_init_by_name(onnx_model, next_node.input[1]):
            return None
        next_static_in = next_node.input[0] if cur_node.output[0] == next_node.input[1] else next_node.input[1]
        next_arr = get_tensor_from_initializer(onnx_model, next_static_in)
        if next_arr.size != 1:
            return None
        if next_node.op_type == 'Mul':
            wt_arr = next_arr
            next_nodes_list = get_node_by_input(onnx_model, next_node.output)
            if len(next_nodes_list) != 1 or next_nodes_list[0].op_type not in ['Add', 'Sub']:
                return [wt_arr, None, [next_node]]
            bais_node = next_nodes_list[0]
            if not find_init_by_name(onnx_model, bais_node.input[0]) and not find_init_by_name(onnx_model, bais_node.input[1]):
                return [wt_arr, None, [next_node]]
            bais_static_in = bais_node.input[0] if bais_node.input[1] == next_node.output[0] else bais_node.input[1]
            bais_arr = get_tensor_from_initializer(onnx_model, bais_static_in)
            if bais_arr.size != 1:
                return [wt_arr, None, [next_node]]
            elif bais_node.op_type == 'Sub':
                if bais_static_in == bais_node.input[0]:
                    wt_arr = wt_arr * -1
                else:
                    bais_arr = bais_arr * -1
                bais_arr = bais_arr.astype(np.float32)
                return [wt_arr, bais_arr, [next_node, bais_node]]
        else:
            bais_arr = next_arr
            if next_node.op_type == 'Sub' and next_static_in == next_node.input[0]:
                wt_arr = np.array((-1), dtype=np.float32)
                return [wt_arr, bais_arr, [next_node]]
            return [None, bais_arr, [next_node]]
            
    if check_node_serial_group(onnx_model, node, ['Mul', 'ReduceSum']):
        topMul, reduceSum = get_node_serial_group(onnx_model, node, ['Mul', 'ReduceSum'])
        topMulOutShape = get_shape_by_name(onnx_model, topMul.output[0])
        pre_node = get_node_by_output(onnx_model, topMul.input[0])
        if pre_node is None:
            return onnx_model, False
        rstp_rm_nodes_list = []
        bot_node = None
        pre_node_out_shape = get_shape_by_name(onnx_model, pre_node.output[0])
        pre_node_in_shape = get_shape_by_name(onnx_model, pre_node.input[0])
        if len(pre_node_out_shape) + 1 != len(pre_node_in_shape):
            return onnx_model, False
        if pre_node.op_type == 'Reshape':
            if len(pre_node_out_shape) != 3 or pre_node_in_shape[:-1] != pre_node_out_shape \
                and pre_node_in_shape[:-2] + [pre_node_in_shape[-1]] != pre_node_out_shape:
                return onnx_model, False
            tpNode = get_node_by_output(onnx_model, pre_node.input[0])
            if tpNode.op_type != 'Transpose':
                return onnx_model, False
            rstp_rm_nodes_list.append(pre_node)
            rstp_rm_nodes_list.append(tpNode)
            bot_node = pre_node
        elif pre_node.op_type == 'Squeeze':
            pre_axes_list = attribute_to_dict(pre_node.attribute).get('axes')
            if pre_axes_list is None or len(pre_axes_list) > 1:
                return onnx_model, False
            pre_axes = pre_axes_list[0]
            pre_pos_axes = len(pre_node_in_shape) + pre_axes if pre_axes < 0 else pre_axes
            if len(pre_node_out_shape) != 3 or pre_pos_axes not in [len(pre_node_in_shape)-2, len(pre_node_in_shape)-1]:
                return onnx_model, False
            tpNode = get_node_by_output(onnx_model, pre_node.input[0])
            if tpNode.op_type != 'Transpose':
                return onnx_model, False    
            rstp_rm_nodes_list.append(pre_node)
            rstp_rm_nodes_list.append(tpNode)
            bot_node = pre_node   
        elif pre_node.op_type == 'Transpose' and len(pre_node_out_shape) == 4:
            tpNode = pre_node
            bot_node = tpNode
        else:
            return onnx_model, False
        data_type = get_dtype_by_name(onnx_model, tpNode.input[0])
        tpPerm = attribute_to_dict(tpNode.attribute).get('perm', list(range(len(pre_node_in_shape))).reverse())
        tpOutShape = get_shape_by_name(onnx_model, tpNode.output[0])
        if not (tpPerm.index(1) == 2 and pre_node_in_shape[:-1] == pre_node_out_shape) \
            and not (tpPerm.index(1) == 3 and pre_node_in_shape[:-2]+[pre_node_in_shape[-1]] == pre_node_out_shape):
            return onnx_model, False
        if not find_init_by_name(onnx_model, topMul.input[0]) and not find_init_by_name(onnx_model, topMul.input[1]):
            return onnx_model, False
        topMulStIn = topMul.input[0] if topMul.input[1] == bot_node.output[0] else topMul.input[1]
        topMulArr = get_tensor_from_initializer(onnx_model, topMulStIn)
        while len(topMulArr.shape) != len(tpOutShape):
            topMulArr = np.expand_dims(topMulArr, 0)
        if tuple(topMulArr.shape[:2]) != tuple([1, 1]):
            return onnx_model, False
        topMulArrShapeList = list(topMulArr.shape)
        if tpOutShape[tpPerm.index(1)] not in topMulArrShapeList or tpOutShape[tpPerm.index(1)] != np.prod(np.array(topMulArrShapeList)):
            return onnx_model, False
        reduceSumAxesList = attribute_to_dict(reduceSum.attribute).get('axes')
        if reduceSumAxesList is None or len(reduceSumAxesList) != 1:
            return onnx_model, False
        reduceSumAxes = reduceSumAxesList[0]
        reduceSumPosAxes = len(topMulOutShape) + reduceSumAxes if reduceSumAxes < 0 else reduceSumAxes
        if not (len(topMulOutShape) == 3 and reduceSumPosAxes == 2) and \
            not (len(topMulOutShape) == 4 and reduceSumPosAxes == tpPerm.index(1)):
            return onnx_model, False
        tpNodeInShape = get_shape_by_name(onnx_model, tpNode.input[0])
        conv_wt_arr = np.ones((1, tpNodeInShape[1]), dtype=np.float32)
        conv_wt_arr = conv_wt_arr * topMulArr.reshape(1, -1)
        out_node = reduceSum
        
        conv_bais_arr = None
        out_rm_nodes_list = []
        bais_serial = find_conv_bais_tensor(onnx_model, reduceSum)
        if bais_serial is not None:
            stack_wt_arr, conv_bais_arr, out_rm_nodes_list = bais_serial
            if stack_wt_arr is not None:
                conv_wt_arr = conv_wt_arr * stack_wt_arr.reshape(1, -1)
            if conv_bais_arr is not None:
                conv_bais_arr = np.reshape(conv_bais_arr, (1,))
            out_node = out_rm_nodes_list[-1]
        conv_wt_arr = conv_wt_arr.reshape(1, -1, 1, 1)
        conv_wt_tensor = get_initial_by_value(onnx_model, conv_wt_arr)
        if conv_wt_tensor is None:
            conv_wt_tensor = onnx.helper.make_tensor(name=reduceSum.name+'_toConv_wt',
                                                     data_type=NPDTYPE_2_ONNXDTYPE[conv_wt_arr.dtype],
                                                     dims=conv_wt_arr.shape,
                                                     vals=conv_wt_arr.flatten().tolist())
            onnx_model.graph.initializer.append(conv_wt_tensor)
        conv_inputs = [tpNode.input[0], conv_wt_tensor.name]
        if conv_bais_arr is not None:
            conv_bais_tensor = get_initial_by_value(onnx_model, conv_bais_arr)
            if conv_bais_tensor is None:
                conv_bais_tensor = onnx.helper.make_tensor(name=reduceSum.name+'_toConv_bais',
                                                           data_type=NPDTYPE_2_ONNXDTYPE[conv_bais_arr.dtype],
                                                           dims=conv_bais_arr.shape,
                                                           vals=conv_bais_arr.flatten().tolist())
                onnx_model.graph.initializer.append(conv_bais_tensor)
            conv_inputs.append(conv_bais_tensor.name)
        conv_out_shape = copy.deepcopy(tpNodeInShape)
        conv_out_shape[1] = 1
        src_out_shape = get_shape_by_name(onnx_model, out_node.output[0])
        conv_output = out_node.output[0] + ('_newShape' if conv_out_shape != src_out_shape else '')
        conv_attr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
        conv_node = onnx.helper.make_node(name=reduceSum.name+'_toConv',
                                          op_type='Conv',
                                          inputs=conv_inputs,
                                          outputs=[conv_output],
                                          **conv_attr)
        if src_out_shape != conv_out_shape:
            src_shape_tensor = get_initial_by_value(onnx_model, np.array(src_out_shape, dtype=np.int64))
            if src_shape_tensor is None:
                src_shape_tensor = onnx.helper.make_tensor(name=out_node.output[0]+'_shape',
                                                           data_type=TensorProto.INT64,
                                                           dims=[len(src_out_shape)],
                                                           vals=src_out_shape)
                onnx_model.graph.initializer.append(src_shape_tensor)
            new_reshape_node = onnx.helper.make_node(name=out_node.output[0]+'_reshape',
                                                     op_type='Reshape',
                                                     inputs=[conv_node.output[0], src_shape_tensor.name],
                                                     outputs=[out_node.output[0]])
            onnx_model.graph.node.insert(node_index, new_reshape_node)
            conv_out_value_info = onnx.helper.make_tensor_value_info(conv_node.output[0], data_type, conv_out_shape)
            onnx_model.graph.value_info.append(conv_out_value_info)
        onnx_model.graph.node.insert(node_index, conv_node)
        for out_rm_node in out_rm_nodes_list:
            if out_rm_node.name != out_node.name:
                onnx_model = delete_value_info_by_name(onnx_model, out_rm_node.output[0])
            onnx_model.graph.node.remove(out_rm_node)
        if out_node.name != reduceSum.name:
            onnx_model = delete_value_info_by_name(onnx_model, reduceSum.output[0])
        onnx_model.graph.node.remove(reduceSum)
        onnx_model.graph.node.remove(topMul)
        for rstp_node in rstp_rm_nodes_list:
            rstp_next_nodes_list = get_node_by_input(onnx_model, rstp_node.output)
            if not rstp_next_nodes_list:
                onnx_model = delete_value_info_by_name(onnx_model, rstp_node.output[0])
                onnx_model.graph.node.remove(rstp_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertSpecial3DimReshapeConcatTo4DimConcat(onnx_model, node, node_index):        
    if node.op_type != 'Concat':
        return onnx_model, False
    concat_out_shape = get_shape_by_name(onnx_model, node.output[0])
    if len(concat_out_shape) != 3:
        return onnx_model, False
    net_out_names_list = [net_out.name for net_out in onnx_model.graph.output]
    if node.output[0] not in net_out_names_list:
        return onnx_model, False
    axis = attribute_to_dict(node.attribute).get('axis', 1)
    pos_axis = len(concat_out_shape) + axis if axis < 0 else axis
    if pos_axis == 0:
        return onnx_model, False
    concat_in_nodes_list = [get_node_by_output(onnx_model, concat_in) for concat_in in node.input]
    reshape_nodes_list = [rs_node for rs_node in concat_in_nodes_list if rs_node.op_type == 'Reshape']
    if len(reshape_nodes_list) != len(concat_in_nodes_list):
        return onnx_model, False
    new_concat_inputs = []
    dst_insert_nodes_list = []
    for reshape_node in reshape_nodes_list:
        concat_in_shape = get_shape_by_name(onnx_model, reshape_node.output[0])
        reshape_in_shape = get_shape_by_name(onnx_model, reshape_node.input[0])
        if len(reshape_in_shape) != 4 or concat_in_shape[0] != reshape_in_shape[0] != 1:
            return onnx_model, False
        if concat_in_shape[:3] == reshape_in_shape[:3]:
            new_concat_inputs.append(reshape_node.input[0])
        elif reshape_in_shape[-1] == np.prod(np.array(concat_in_shape[1:])) and 1 in concat_in_shape[1:]:
            new_perm = [0, 3, 1, 2] if reshape_in_shape[-1] == concat_in_shape[1] else [0, 1, 3, 2]
            new_tp_node = onnx.helper.make_node(name=reshape_node.name+'_toTranspose',
                                                op_type='Transpose',
                                                inputs=[reshape_node.input[0]],
                                                outputs=[reshape_node.output[0]],
                                                perm=new_perm)
            dst_insert_nodes_list.append(new_tp_node)
            new_concat_inputs.append(new_tp_node.output[0])
        else:
            return onnx_model, False
    for idx, src_concat_input in enumerate(new_concat_inputs):
        node.input[idx] = src_concat_input
    dst_insert_nodes_list.reverse()
    onnx_model = insert_node_by_list(onnx_model, dst_insert_nodes_list, node_index)
    for reshape_node in reshape_nodes_list:
        onnx_model = delete_value_info_by_name(onnx_model, reshape_node.output[0])
        onnx_model.graph.node.remove(reshape_node)
    concat_out_shape.append(1)
    out_data_type = get_dtype_by_name(onnx_model, node.output[0])
    for net_out in onnx_model.graph.output:
        if net_out.name == node.output[0]:
            onnx_model.graph.output.remove(net_out)
            break
    new_net_out_value = onnx.helper.make_tensor_value_info(node.output[0], out_data_type, concat_out_shape)
    onnx_model.graph.output.append(new_net_out_value)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertCHReshapeTransposeToCW(onnx_model, node, node_index): 
    if check_node_serial_group(onnx_model, node, ['Reshape', 'Transpose']):
        reshape_node, transpose_node = get_node_serial_group(onnx_model, node, ['Reshape', 'Transpose'])
        reshape_out_shape = get_shape_by_name(onnx_model, reshape_node.output[0])
        if len(reshape_out_shape) != 4 or reshape_out_shape[0] != 1 or reshape_out_shape[-1] != 1:
            return onnx_model, False
        transpose_perm = attribute_to_dict(transpose_node.attribute).get('perm', list(range(len(reshape_out_shape))).reverse())
        new_transpose_perm = copy.deepcopy(transpose_perm)
        new_transpose_perm[transpose_perm.index(2)] = 3
        new_transpose_perm[transpose_perm.index(3)] = 2
        new_reshape_shape = [reshape_out_shape[idx] for idx in [0, 1, 3, 2]]
        new_reshape_shape_tensor = get_initial_by_value(onnx_model, np.array(new_reshape_shape, dtype=np.int64))
        if new_reshape_shape_tensor is None:
            new_reshape_shape_tensor = onnx.helper.make_tensor(
                name=get_unique_node_tensor_name(onnx_model, reshape_node.input[0]+'_new'),
                data_type=TensorProto.INT64,
                dims=[len(reshape_out_shape)],
                vals=new_reshape_shape)
            onnx_model.graph.initializer.append(new_reshape_shape_tensor)
        reshape_node.input[1] = new_reshape_shape_tensor.name
        new_perm_attr = onnx.helper.make_attribute('perm', new_transpose_perm)
        del transpose_node.attribute[:]
        transpose_node.attribute.append(new_perm_attr)
        out_dtype = get_dtype_by_name(onnx_model, reshape_node.output[0])
        onnx_model = delete_value_info_by_name(onnx_model, reshape_node.output[0])
        new_reshape_out_value_info = onnx.helper.make_tensor_value_info(reshape_node.output[0], out_dtype, new_reshape_shape)
        onnx_model.graph.value_info.append(new_reshape_out_value_info)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_deleteSameAxesSliceWithConcat(onnx_model, node, node_index): 
    if node.op_type != 'Concat':
        return onnx_model, False
    concat_out_shape = get_shape_by_name(onnx_model, node.output[0])
    concat_axis = attribute_to_dict(node.attribute).get('axis')
    concat_pos_axis = len(concat_out_shape) + concat_axis if concat_axis < 0 else concat_axis
    concat_input_nodes_list = []
    for concat_input in node.input:
        concat_in_node = get_node_by_output(onnx_model, concat_input)
        if concat_in_node is not None:
            concat_input_nodes_list.append(concat_in_node)
    concat_in_slices_list = [concat_in_node for concat_in_node in concat_input_nodes_list if concat_in_node.op_type == 'Slice']
    same_axis_slices_dict = {}
    for in_slice_node in concat_in_slices_list:
        if len(in_slice_node.input) < 4:
            continue
        slice_axes_arr = get_tensor_from_initializer(onnx_model, in_slice_node.input[3])
        if not slice_axes_arr.shape:
            slice_axes_arr = np.reshape(slice_axes_arr, (1,))
        if slice_axes_arr.size > 1:
            continue
        slice_axis = slice_axes_arr[0] if slice_axes_arr[0] >= 0 else len(concat_out_shape) + slice_axes_arr[0]
        if slice_axis != concat_pos_axis:
            continue
        dict_key_name = tuple([in_slice_node.input[0], slice_axis])
        if dict_key_name not in same_axis_slices_dict:
            same_axis_slices_dict[dict_key_name] = []
        same_axis_slices_dict[dict_key_name].append(in_slice_node)
    if not same_axis_slices_dict:
        return onnx_model, False
    src_concat_inputs_list = list(node.input)
    same_axis_slices_dict_cp = copy.deepcopy(same_axis_slices_dict)
    for same_tuple in same_axis_slices_dict_cp:
        src_local_idx = 0
        accord_flag = True
        for cur_slice_node in same_axis_slices_dict[same_tuple]:
            if src_local_idx == 0:
                src_local_idx = src_concat_inputs_list.index(cur_slice_node.output[0])
            elif src_local_idx + 1 != src_concat_inputs_list.index(cur_slice_node.output[0]):
                del same_axis_slices_dict_cp[same_tuple]
                accord_flag = False
                break
        if not accord_flag:
            break             
    new_concat_slice_inputs_list = []
    dst_rm_slices_list = []
    for same_tuple in same_axis_slices_dict:
        same_input_shape = get_shape_by_name(onnx_model, same_tuple[0])
        cur_slice_nodes_list = same_axis_slices_dict[same_tuple]
        cur_concat_out_shape = copy.deepcopy(same_input_shape)
        cur_concat_out_shape[same_tuple[1]] = 0
        for cur_slice_node in cur_slice_nodes_list:
            cur_slice_out_shape = get_shape_by_name(onnx_model, cur_slice_node.output[0])
            cur_concat_out_shape[same_tuple[1]] += cur_slice_out_shape[same_tuple[1]]
        if cur_concat_out_shape == same_input_shape:
            new_concat_slice_inputs_list.append(same_tuple[0])
            dst_rm_slices_list += cur_slice_nodes_list
    new_concat_inputs_list = []
    for src_concat_input in src_concat_inputs_list:
        src_concat_in_node = get_node_by_output(onnx_model, src_concat_input)
        if src_concat_in_node is None or src_concat_in_node.input[0] not in new_concat_slice_inputs_list:
            new_concat_inputs_list.append(src_concat_input)
        elif not new_concat_inputs_list or src_concat_in_node.input[0] != new_concat_inputs_list[-1]:
            new_concat_inputs_list.append(src_concat_in_node.input[0])
    new_concat_node = onnx.helper.make_node(name=node.name,
                                            op_type='Concat',
                                            inputs=new_concat_inputs_list,
                                            outputs=node.output,
                                            axis=concat_pos_axis)
    onnx_model.graph.node.remove(node)
    onnx_model.graph.node.insert(node_index, new_concat_node)
    for rm_slice_node in dst_rm_slices_list:
        onnx_model = delete_value_info_by_name(onnx_model, rm_slice_node.output[0])
        onnx_model.graph.node.remove(rm_slice_node)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    return onnx_model, True

@OnnxDebuggerMeet.opt_convert_wrapper
def opt_convertTransposeConcatToReverse(onnx_model, node, node_index):
    if node.op_type != 'Concat':
        return onnx_model, False
    concat_out_shape = get_shape_by_name(onnx_model, node.output[0])
    net_inputs_list = [net_input.name for net_input in onnx_model.graph.input]
    concat_in_nodes_list = []
    concat_in_initials_dict = {}
    for concat_input in node.input:
        if concat_input in net_inputs_list:
            return onnx_model, False
        elif find_init_by_name(onnx_model, concat_input):
            if concat_input not in concat_in_initials_dict:
                concat_in_initials_dict[concat_input] = get_tensor_from_initializer(onnx_model, concat_input)
            continue
        concat_in_node = get_node_by_output(onnx_model, concat_input)
        if concat_in_node is not None and concat_in_node not in concat_in_nodes_list:
            concat_in_nodes_list.append(concat_in_node)
    concat_in_transpose_nodes_list = [idx_node for idx_node in concat_in_nodes_list if idx_node.op_type == 'Transpose']
    if len(concat_in_transpose_nodes_list) != len(concat_in_nodes_list):
        return onnx_model, False
    init_perm = None
    for transpose_node in concat_in_transpose_nodes_list:
        cur_perm = attribute_to_dict(transpose_node.attribute).get('perm', list(range(len(concat_out_shape))).reverse())
        if init_perm is None: init_perm = cur_perm
        if cur_perm != init_perm: 
            return onnx_model, False
    new_initials_tensor_dict = {}
    for input_init_key in concat_in_initials_dict:
        input_init_arr = concat_in_initials_dict[input_init_key]
        new_input_arr = np.transpose(input_init_arr, tuple(init_perm))
        new_init_tensor = get_initial_by_value(onnx_model, new_input_arr)
        if new_init_tensor is None:
            new_init_tensor = onnx.helper.make_tensor(name=input_init_key+'_new',
                                                      data_type=NPDTYPE_2_ONNXDTYPE[new_input_arr.dtype],
                                                      dims=new_input_arr.shape,
                                                      vals=new_input_arr.flatten().tolist())
            onnx_model.graph.initializer.append(new_init_tensor)
        new_initials_tensor_dict[input_init_key] = new_init_tensor
    concat_axis = attribute_to_dict(node.attribute).get('axis')
    concat_pos_axis = len(concat_out_shape) + concat_axis if concat_axis < 0 else concat_axis
    for transpose_node in concat_in_transpose_nodes_list:
        for idx, src_input in enumerate(node.input):
            node.input[idx] = transpose_node.input[0] if src_input == transpose_node.output[0] else src_input
    for new_initial_key in new_initials_tensor_dict:
        for idx, src_input in enumerate(node.input):
            node.input[idx] = new_initials_tensor_dict[new_initial_key].name \
                if src_input == new_initial_key else src_input
    del node.attribute[:]
    new_concat_axis = init_perm[concat_pos_axis]
    new_axis_attr = onnx.helper.make_attribute('axis', new_concat_axis)
    node.attribute.append(new_axis_attr)
    new_transpose_node = onnx.helper.make_node(name=concat_in_transpose_nodes_list[0].name+'_cvt',
                                               op_type='Transpose',
                                               inputs=[node.output[0]+'_reserve'],
                                               outputs=[node.output[0]],
                                               perm=init_perm)
    out_dtype = get_dtype_by_name(onnx_model, node.output[0])
    node.output[0] = node.output[0] + '_reserve'
    onnx_model.graph.node.insert(node_index+1, new_transpose_node)
    reverse_perm = [init_perm[i] for i in init_perm]
    if reverse_perm == list(range(len(concat_out_shape))):
        reverse_perm = init_perm
    new_concat_out_shape = [concat_out_shape[x] for x in reverse_perm]
    new_concat_value_info = onnx.helper.make_tensor_value_info(node.output[0], out_dtype, new_concat_out_shape)
    onnx_model.graph.value_info.append(new_concat_value_info)
    for src_transpose_node in concat_in_transpose_nodes_list:
        onnx_model = delete_value_info_by_name(onnx_model, src_transpose_node.output[0])
    onnx_model = delete_nodes(onnx_model, concat_in_transpose_nodes_list)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    return onnx_model, True