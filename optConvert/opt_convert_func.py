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