import argparse
import onnx
import onnx.version_converter
from onnxsim import simplify
from run_onnx_opt import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", type=str, default="/workspace/nxu/customer/gelingshentong/20230717/encoder_new.onnx", help="input onnx model path")
    parser.add_argument("-o", "--output_model", type=str, default="/workspace/nxu/customer/gelingshentong/20230717/encoder_new_preopt.onnx", help="output onnx model path")
    parser.add_argument("-v", "--convert_opset", type=int, default=11, help="whether to convert opset version")
    parser.add_argument("--debug", action='store_true', help="run mode is debug or release, defualt release")
    args = parser.parse_args()
    return args

def main(args):
    dstOptSetVer = args.convert_opset
    args.debug = True
    srcPath = args.input_model
    dstPath = args.output_model
    debug_mode = 'debug' if args.debug else 'release'
    '''
    PreProcess
    '''
    logger = logging.getLogger("[PreProcess]")
    onnx_model = onnx.load_model(srcPath)
    if dstOptSetVer and onnx_model.opset_import[0].version != dstOptSetVer:
        onnx_model = onnx.version_converter.convert_version(onnx_model, dstOptSetVer)
    onnx_model.ir_version = 7 if onnx_model.ir_version > 7 else onnx_model.ir_version
    onnx_model = model_preprocess(onnx_model)
    #model.opset_import.extend([onnx.helper.make_opsetid('art.custom.add', 1)])

    '''
    Explanation run opt
    '''
    logger = logging.getLogger("[OPTPROC]")
    clsOpt = OnnxConvertOptimizer(onnx_model=onnx_model, debug_mode=debug_mode, save_path=dstPath)
    logger.info("Start run opt ... ")
    onnx_model = clsOpt.opt()
    logger.info("Opt finish!")
    
    '''
    PostProcess
    '''
    logger = logging.getLogger("[PostProcess]")
    logger.info("Start simplifier after graph optimization ...")
    onnx_model, check = simplify(onnx_model)
    logger.info("Finish simplifier after graph optimization")
    onnx.save_model(onnx_model, dstPath)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)