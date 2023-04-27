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