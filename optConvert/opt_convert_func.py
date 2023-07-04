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
        onnx_model = delete_nodes(onnx_model, delNodesList)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        onnx_model = delete_useless_value_info(onnx_model)
        if not delNodesList:
            return onnx_model, False
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
        for delNode in [botRSNode, topRSNode, topTPNode]:
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
        if rightConvNode.op_type != 'Conv':
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