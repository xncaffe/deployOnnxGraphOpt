import copy
from basicUtil.baseUtil import *
from basicUtil.convertDebugger import *

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_deleteGatherInput(onnx_model):
    restart = False
    onnx_modelCp = copy.deepcopy(onnx_model)
    for input_index, input in enumerate(onnx_modelCp.graph.input):
        nodes = get_node_by_input(onnx_modelCp, [input.name])
        for node in nodes:
            if node.op_type == "Gather" and \
                (find_init_by_name(onnx_modelCp, node.input[0]) or find_init_by_name(onnx_modelCp, node.input[1])):
                    gather_outShape = get_shape_by_name(onnx_model, node.output[0])
                    onnx_model.graph.node.remove(node)
                    gather_outType = get_dtype_by_name(onnx_model, node.output[0])
                    new_input_value_info = onnx.helper.make_tensor_value_info(node.output[0], gather_outType, gather_outShape)
                    onnx_model.graph.input.append(new_input_value_info)
                    restart = True
    if restart:
        onnx_model = delete_useless_inputOfModel(onnx_model)
    return onnx_model, restart

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_mulReplaceWhereBoolInput(onnx_model):
    for index, node in enumerate(onnx_model.graph.node):
        if check_node_serial_group(onnx_model, node, ["Unsqueeze", "Cast", "Equal", "Where"]):
            nodes_list = get_node_serial_group(onnx_model, node, ["Unsqueeze", "Cast", "Equal", "Where"])
            unsqueeze_node = nodes_list[0]
            netInputs_name = [input.name for input in onnx_model.graph.input]
            if unsqueeze_node.input[0] not in netInputs_name:
                continue
            parallInputNodes = get_node_by_input(onnx_model, [unsqueeze_node.input[0]])
            otherNodes = [parallNode for parallNode in parallInputNodes if parallNode.op_type != "Unsqueeze"]
            parallInputNodes = [parallNode for parallNode in parallInputNodes if parallNode not in otherNodes]
            if not parallInputNodes:
                continue
            parallBranchDicts = {}
            parallInputNodesCp = copy.deepcopy(parallInputNodes)
            for id, parallUnsqueeze in enumerate(parallInputNodesCp):
                castNodes = get_node_by_input(onnx_model, parallUnsqueeze.output)
                if castNodes[0].op_type != "Cast" or len(castNodes) != 1:
                    parallInputNodes.remove(parallUnsqueeze)
                    otherNodes.append(parallUnsqueeze)
                else:
                    parallBranchDicts[id] = [parallUnsqueeze, castNodes[0]]
            if not parallBranchDicts:
                continue
            parallBranchDictsCp = copy.deepcopy(parallBranchDicts)
            for id in parallBranchDictsCp:
                castNode = parallBranchDictsCp[id][1]
                equalNodes = get_node_by_input(onnx_model, castNode.output)
                if equalNodes[0].op_type != "Equal" or len(equalNodes) != 1:
                    parallBranchDicts.remove(parallBranchDictsCp[id])
                    parallInputNodes.remove(parallBranchDictsCp[id][0])
                    otherNodes.append(parallBranchDictsCp[id][0])
                else:
                    parallBranchDicts[id].append(equalNodes[0])
            if not parallBranchDicts:
                continue
            parallBranchDictsCp = copy.deepcopy(parallBranchDicts)
            sperateShapes = []
            for id in parallBranchDictsCp:
                equalNode = parallBranchDictsCp[id][2]
                whereNodes = get_node_by_input(onnx_model, equalNode.output)
                if whereNodes[0].op_type != "Where" or len(whereNodes) != 1:
                    parallBranchDicts.remove(parallBranchDictsCp[id])
                    parallInputNodes.remove(parallBranchDictsCp[id][0])
                    otherNodes.append(parallBranchDictsCp[id][0])
                else:
                    parallBranchDicts[id].append(whereNodes[0])
                    equalOutShape = get_shape_by_name(onnx_model, equalNode.output[0])
                    if equalOutShape not in sperateShapes:
                        sperateShapes.append(equalOutShape)
            sperateIds = []
            for shape in sperateShapes:
                shapeEqualIds = []
                for id in parallBranchDicts:
                    equalNode = parallBranchDicts[id][2]
                    equalOutShape = get_shape_by_name(onnx_model, equalNode.output[0])
                    if shape == equalOutShape:
                        shapeEqualIds.append(id)
                if shapeEqualIds:
                    sperateIds.append(shapeEqualIds)
            if len(sperateIds) != len(sperateShapes):
                logger.error("The branch id grouping is inconsistent with the shape grouping, please check!")
                assert(0)
            deleteNodesList = []
            for shapeId, branchIds_list in enumerate(sperateIds):
                new_input_name = parallBranchDicts[branchIds_list[0]][0].input[0] + "_new_%d"%shapeId
                newInsertMulDynamicInShape = sperateShapes[shapeId]
                newNetInputTensor = onnx.helper.make_tensor_value_info(new_input_name, TensorProto.FLOAT, newInsertMulDynamicInShape)
                for branchId in branchIds_list:
                    whereNode = parallBranchDicts[branchId][3]
                    equalNode = parallBranchDicts[branchId][2]
                    whereTrueInput = None
                    for whereInput in whereNode.input:
                        whereInputNode = get_node_by_output(onnx_model, whereInput)
                        if whereInputNode is not None and whereInputNode != equalNode:
                            whereTrueInput = whereInput
                    if whereTrueInput is None:
                        logger.error("The correct input corresponding to where node(name:{}) was not found, please check".format(whereNode.name))
                        assert(0)
                    delNodes = [delNode for delNode in parallBranchDicts[branchId] if delNode not in deleteNodesList]
                    deleteNodesList.extend(delNodes)
                    newMulNode = onnx.helper.make_node(name=whereNode.name+"_relpace_Mul",
                                                       op_type="Mul",
                                                       inputs=[whereTrueInput, newNetInputTensor.name],
                                                       outputs=whereNode.output)
                    whereNodeId = get_node_id(onnx_model, whereNode)
                    onnx_model.graph.node.insert(whereNodeId, newMulNode)
                onnx_model.graph.input.append(newNetInputTensor)
            onnx_model = delete_nodes(onnx_model, deleteNodesList)
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            onnx_model = delete_useless_value_info(onnx_model)
            onnx_model = delete_useless_inputOfModel(onnx_model)
            return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_moveForwardInputReshapeTranspose(onnx_model):
    for netInput in onnx_model.graph.input:
        firstNodesList = get_node_by_input(onnx_model, [netInput.name])
        for firNode in firstNodesList:
            if check_node_serial_group(onnx_model, firNode, ['Mul', 'Reshape', 'Transpose']):
                mulNode, reshapeNode, transposeNode = get_node_serial_group(onnx_model, firNode, ['Mul', 'Reshape', 'Transpose'])
                stMulIn = mulNode.input[1] if mulNode.input[0] == netInput.name else mulNode.input[0]
                if not find_init_by_name(onnx_model, stMulIn):
                    continue
                netInShape = get_shape_by_name(onnx_model, netInput.name)
                mulOutShape = get_shape_by_name(onnx_model, mulNode.output[0])
                reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
                if netInShape != mulOutShape:
                    continue
                tpPerm = attribute_to_dict(transposeNode.attribute).get('perm', list(range(len(reshapeOutShape))).reverse())
                stMulInArr = get_tensor_from_initializer(onnx_model, stMulIn)
                if stMulInArr.size != 1:
                    if np.array(stMulInArr.shape, dtype=np.int32).prod() in reshapeOutShape:
                        if len(stMulInArr.shape) == 1:
                            for i in range(len(reshapeOutShape)-1):
                                stMulInArr = np.expand_dims(stMulInArr, axis=0)
                    else:
                        stMulInArr = np.reshape(np.broadcast_to(stMulInArr, tuple(mulOutShape)), tuple(reshapeOutShape))
                    stMulInArr = np.transpose(stMulInArr, tuple(tpPerm))
                    stMulInTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stMulIn+'_new'),
                                                            data_type=NPDTYPE_2_ONNXDTYPE[stMulInArr.dtype],
                                                            dims=stMulInArr.shape,
                                                            vals=stMulInArr.flatten().tolist())
                    onnx_model.graph.initializer.append(stMulInTensor)
                    mulNode.input[list(mulNode.input).index(stMulIn)] = stMulInTensor.name
                    stMulIn = stMulInTensor.name
                transposeOutShape = get_shape_by_name(onnx_model, transposeNode.output[0])
                tpOutNodesList = get_node_by_input(onnx_model, transposeNode.output)
                newMulNode = onnx.helper.make_node(name=mulNode.name+'_backMove',
                                                   op_type='Mul',
                                                   inputs=[transposeNode.output[0], stMulIn],
                                                   outputs=[mulNode.output[0]+'_'+transposeNode.output[0]])
                newMulOutValue = onnx.helper.make_tensor_value_info(newMulNode.output[0], netInput.type.tensor_type.elem_type, transposeOutShape)
                onnx_model.graph.value_info.append(newMulOutValue)
                onnx_model = delete_value_info_by_name(onnx_model, mulNode.output[0])
                reshapeNode.input[0] = netInput.name
                for tpOutNode in tpOutNodesList:
                    for inId,tpOutNodeIn in enumerate(tpOutNode.input):
                        tpOutNode.input[inId] = newMulNode.output[0] if tpOutNodeIn == transposeNode.output[0] else tpOutNodeIn
                tpNodeId = get_node_id(onnx_model, transposeNode)
                onnx_model.graph.node.insert(tpNodeId+1, newMulNode)
                onnx_model.graph.node.remove(mulNode)
                return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_mulReshapeTransposeInputMove(onnx_model):
    for netInput in onnx_model.graph.input:
        firstNodesList = get_node_by_input(onnx_model, [netInput.name])
        if len(firstNodesList) != 1:
            continue
        mulNode = firstNodesList[0]
        if mulNode.op_type != 'Mul':
            continue
        mulInShape = get_shape_by_name(onnx_model, netInput.name)
        if len(mulInShape) != 3 or mulInShape[0] != 1:
            continue
        stMulIn = mulNode.input[1] if netInput.name == mulNode.input[0] else mulNode.input[0]
        if not find_init_by_name(onnx_model, stMulIn):
            continue
        stMulInArr = get_tensor_from_initializer(onnx_model, stMulIn)
        reshapeNodesList = get_node_by_input(onnx_model, mulNode.output)
        if len(reshapeNodesList) != 1 or reshapeNodesList[0].op_type != 'Reshape':
            continue
        reshapeNode = reshapeNodesList[0]
        reshapeOutShape = get_shape_by_name(onnx_model, reshapeNode.output[0])
        reshapeInShape = get_shape_by_name(onnx_model, reshapeNode.input[0])
        if len(reshapeOutShape) != 4 or reshapeOutShape[:2] != reshapeInShape[:2]:
            continue
        reshapeShape = get_tensor_from_initializer(onnx_model, reshapeNode.input[0])
        newStMulInArr = stMulInArr
        if stMulInArr.size != 1:
            newStMulInArr = stMulInArr[np.newaxis, np.newaxis, :] if (stMulInArr.shape) == 1 else stMulInArr
            if newStMulInArr.shape == reshapeInShape:
                newStMulInArr = np.reshape(newStMulInArr, tuple(reshapeShape))
            elif newStMulInArr.shape[-1] == 1:
                newStMulInArr = np.reshape(newStMulInArr, (newStMulInArr.shape[0], newStMulInArr.shape[1], 1, 1))
            else:
                newStMulInArr = np.reshape(newStMulInArr, (newStMulInArr.shape[0], 1, reshapeShape[2], reshapeShape[3]))
        newNetInShape = [1, 1, mulInShape[-1]] if len(mulInShape) == 1 else mulInShape
        if mulInShape == reshapeInShape:
            newNetInShape = reshapeOutShape
        elif mulInShape[-1] == 1:
            newNetInShape.append(1)
        else:
            newNetInShape = [newNetInShape[0], newNetInShape[1], reshapeOutShape[2], reshapeOutShape[3]]
        transposeNodesList = get_node_by_input(onnx_model, reshapeNode.output)
        if len(transposeNodesList) != 1 or transposeNodesList[0].op_type != 'Transpose':
            if newStMulInArr.size != 1:
                newStMulInTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stMulIn+'_new'),
                                                        data_type=NPDTYPE_2_ONNXDTYPE[newStMulInArr.dtype],
                                                        dims=newStMulInTensor.shape,
                                                        vals=newStMulInArr.flatten().tolist())
                onnx_model.graph.initializer.append(newStMulInTensor)
                mulNode.input[list(mulNode.input).index(stMulIn)] = newStMulInTensor.name
            for reshapeOutNode in transposeNodesList:
                for inId, rsOutNodeIn in enumerate(reshapeOutNode.input):
                    reshapeOutNode.input[inId] = mulNode.output[0] if rsOutNodeIn == reshapeNode.output[0] else rsOutNodeIn
            newNetInput = onnx.helper.make_tensor_value_info(netInput.name, netInput.elem_type, newNetInShape)
            onnx_model.graph.input.remove(netInput)
            onnx_model.graph.input.append(newNetInput)
            onnx_model = delete_value_info_by_name(onnx_model, mulNode.output[0])
            onnx_model = delete_value_info_by_name(onnx_model, reshapeNode.output[0])
            newMulOutValue = onnx.helper.make_tensor_value_info(mulNode.output[0], 1, reshapeOutShape)
            onnx_model.graph.value_info.append(newMulOutValue)
            onnx_model.graph.node.remove(reshapeNode)
        else:
            transposeNode = transposeNodesList[0]
            transposePerm = attribute_to_dict(transposeNode.attribute).get('perm', [3, 2, 1, 0])
            if newStMulInArr.size != 1:
                newStMulInArr = np.transpose(newStMulInArr, tuple(transposePerm))
                newStMulInTensor = onnx.helper.make_tensor(name=get_unique_node_tensor_name(onnx_model, stMulIn+'_new'),
                                                        data_type=NPDTYPE_2_ONNXDTYPE[newStMulInArr.dtype],
                                                        dims=newStMulInArr.shape,
                                                        vals=newStMulInArr.flatten().tolist())
                onnx_model.graph.initializer.append(newStMulInTensor)
                mulNode.input[list(mulNode.input).index(stMulIn)] = newStMulInTensor.name
            transposeOutNodesList = get_node_by_input(onnx_model, transposeNode.output)
            for transposeOutNode in transposeOutNodesList:
                for inId, tpOutNodeInput in enumerate(transposeOutNode.input):
                    transposeOutNode.input[inId] = mulNode.output[0] if tpOutNodeInput == transposeNode.output[0] else tpOutNodeInput
            transposeOutShape = get_shape_by_name(onnx_model, transposeNode.output[0])
            newNetInShape = [newNetInShape[perm_id] for perm_id in transposePerm]
            newNetInput = onnx.helper.make_tensor_value_info(netInput.name, netInput.type.tensor_type.elem_type, newNetInShape)
            onnx_model.graph.input.remove(netInput)
            onnx_model.graph.input.append(newNetInput)
            onnx_model = delete_value_info_by_name(onnx_model, mulNode.output[0])
            onnx_model = delete_value_info_by_name(onnx_model, reshapeNode.output[0])
            onnx_model = delete_value_info_by_name(onnx_model, transposeNode.output[0])
            newMulOutValue = onnx.helper.make_tensor_value_info(mulNode.output[0], 1, transposeOutShape)
            onnx_model.graph.value_info.append(newMulOutValue)
            onnx_model = delete_nodes(onnx_model, [reshapeNode, transposeNode])
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_fusionInputTranspose(onnx_model):
    for netInput in onnx_model.graph.input:
        inputNodesList = get_node_by_input(onnx_model, [netInput.name])
        trNodesList = [inputNode for inputNode in inputNodesList if inputNode.op_type == 'Transpose']
        if not trNodesList:
            continue
        netInShape = get_shape_by_name(onnx_model, netInput.name)
        trPermDict = {}
        for trNode in trNodesList:
            trPerm = attribute_to_dict(trNode.attribute).get('perm', list(range(len(netInShape))).reverse())
            trPermStr = ''
            for trPId in trPerm:
                trPermStr+=str(trPId)
                trPermStr+='_'
            if trPermStr not in trPermDict:
                trPermDict[trPermStr] = []
            trPermDict[trPermStr].append(trNode)
        allOneFlag = False
        for trSPerm in list(trPermDict.keys()):
            if len(trPermDict[trSPerm]) > 1:
                for trNode in trPermDict[trSPerm][1:]:
                    trOutNodesList = get_node_by_input(onnx_model, trNode.output)
                    for trOutNode in trOutNodesList:
                        for inId, trOutNodeIn in enumerate(trOutNode.input):
                            trOutNode.input[inId] = trPermDict[trSPerm][0].output[0] \
                                if trOutNodeIn == trNode.output[0] else trOutNodeIn
                    onnx_model = delete_value_info_by_name(onnx_model, trNode.output[0])
                    onnx_model.graph.node.remove(trNode)
                allOneFlag = True
        if not allOneFlag:
            continue
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_deleteInputTranspose(onnx_model):
    for netInput in onnx_model.graph.input:
        inputNodesList = get_node_by_input(onnx_model, [netInput.name])
        trNodesList = [inputNode for inputNode in inputNodesList if inputNode.op_type == 'Transpose']
        if not trNodesList:
            continue
        netInShape = get_shape_by_name(onnx_model, netInput.name)
        trPermDict = {}
        for trNode in trNodesList:
            trPerm = attribute_to_dict(trNode.attribute).get('perm', list(range(len(netInShape))).reverse())
            trPermStr = ''
            for trPId in trPerm:
                trPermStr+=str(trPId)
                trPermStr+='_'
            if trPermStr not in trPermDict:
                trPermDict[trPermStr] = []
            trPermDict[trPermStr].append(trNode)
        for trSPerm in list(trPermDict.keys()):
            trOutShape = get_shape_by_name(onnx_model, trPermDict[trSPerm][0].output[0])
            newNetInValue = onnx.helper.make_tensor_value_info(name=netInput.name+'_{}'.format(trSPerm), 
                                                               elem_type=netInput.type.tensor_type.elem_type,
                                                               shape=trOutShape)
            onnx_model.graph.input.append(newNetInValue)
            for trNode in trPermDict[trSPerm]:
                trOutNodesList = get_node_by_input(onnx_model, trNode.output)
                for trOutNode in trOutNodesList:
                    for inId, trOutNodeIn in enumerate(trOutNode.input):
                        trOutNode.input[inId] = newNetInValue.name if trOutNodeIn == trNode.output[0] else trOutNodeIn
                onnx_model = delete_value_info_by_name(onnx_model, trNode)
                onnx_model.graph.node.remove(trNode)
        onnx_model = delete_useless_inputOfModel(onnx_model)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebuggerMeet.opt_inout_wrapper
def opt_3dimInputReshapeTo4dim(onnx_model):
    for netInput in onnx_model.graph.input:
        netInShape = get_shape_by_name(onnx_model, netInput.name)
        if len(netInShape) != 3:
            continue
        netInNodesList = get_node_by_input(onnx_model, [netInput.name])
        rsNodesList = [netInNode for netInNode in netInNodesList if netInNode.op_type == 'Reshape']
        if len(netInNodesList) != len(rsNodesList):
            continue
        rsOutShape = get_shape_by_name(onnx_model, rsNodesList[0].output[0])
        oneNum = rsOutShape.count(1)
        delRsFlag = False
        if len(rsNodesList) > 1 or len(rsOutShape) != 4 or oneNum != 2:
            netInShape.append(1)
        else:
            firIndex = rsOutShape.index(1)
            secIndex = rsOutShape.index(1, firIndex+1)
            netInShape.insert(secIndex, 1)
            delRsFlag = True
        newNetInValue = onnx.helper.make_tensor_value_info(netInput.name+'_new', netInput.type.tensor_type.elem_type, netInShape)
        onnx_model.graph.input.append(newNetInValue)
        if delRsFlag:
            rsOutNodesList = get_node_by_input(onnx_model, rsNodesList[0].output)
            for rsOutNode in rsOutNodesList:
                for inId, rsOutNodeIn in enumerate(rsOutNode.input):
                    rsOutNode.input[inId] = newNetInValue.name if rsOutNodeIn == rsNodesList[0].output[0] else rsOutNodeIn
            onnx_model = delete_value_info_by_name(onnx_model, rsNodesList[0].output[0])
            onnx_model.graph.node.remove(rsNodesList[0])
            onnx_model = delete_useless_input_in_initializer(onnx_model)
        else:
            for rsNode in rsNodesList:
                rsNode.input[0] = newNetInValue.name
        onnx_model = delete_useless_inputOfModel(onnx_model)
        return onnx_model, True
    return onnx_model, False