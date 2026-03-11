import os
import argparse
import time
from loguru import logger
import onnxruntime as ort

def build_tensorrt_engine(onnx_path, cache_dir="./trt_cache", fp16=True):
    """
    Build TensorRT engine for an ONNX model by running it once with TensorrtExecutionProvider.
    """
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found: {onnx_path}")
        return False
        
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Building TensorRT Engine for {onnx_path}")
    logger.info(f"Cache Directory: {cache_dir}")
    logger.info(f"FP16 Enabled: {fp16}")
    
    start_time = time.time()
    
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
            'trt_fp16_enable': fp16,
            'trt_max_workspace_size': 2147483648, # 2GB
            'trt_builder_optimization_level': 3
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    
    try:
        # Load the model with TRT provider enabled
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # We need to run a dummy inference to trigger the engine build if it's dynamic
        # Get input names and shapes
        inputs = session.get_inputs()
        dummy_inputs = {}
        import numpy as np
        for inp in inputs:
            shape = inp.shape
            # Replace dynamic axes with 1
            shape = [dim if isinstance(dim, int) else 1 for dim in shape]
            dummy_inputs[inp.name] = np.zeros(shape, dtype=np.float32)
            
        logger.info("Running dummy inference to trigger engine build...")
        session.run(None, dummy_inputs)
        
        elapsed = time.time() - start_time
        logger.success(f"Successfully built TensorRT engine for {os.path.basename(onnx_path)} in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to build TensorRT engine for {onnx_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for ONNX models")
    parser.add_argument("--models", nargs="+", help="Paths to ONNX models", required=False)
    parser.add_argument("--cache-dir", default="./trt_cache", help="Directory to save TRT engine cache")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 (use FP32)")
    args = parser.parse_args()
    
    fp16 = not args.fp32
    
    models = args.models
    if not models:
        # Default models (InsightFace buffalo_s and AntiSpoofing)
        home = os.path.expanduser("~")
        buffalo_s_dir = os.path.join(home, ".insightface", "models", "buffalo_s")
        
        models = [
            os.path.join(buffalo_s_dir, "det_500m.onnx"),
            os.path.join(buffalo_s_dir, "w600k_mbf.onnx"),
            "models/anti_spoofing/minifasnet.onnx"
        ]
        
    for model_path in models:
        build_tensorrt_engine(model_path, args.cache_dir, fp16)

if __name__ == "__main__":
    main()
