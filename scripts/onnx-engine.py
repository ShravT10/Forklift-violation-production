import tensorrt as trt

ONNX_PATH = r"nanodet-v1\nanodet-v1.onnx"
ENGINE_PATH = r"nanodet-v1\nanodet-v1_fp16_pc2.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)  # EXPLICIT_BATCH is default in TRT 10+
parser = trt.OnnxParser(network, logger)

print(f"Parsing ONNX: {ONNX_PATH}")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

print("ONNX parsed successfully.")
print(f"Network inputs : {network.num_inputs}")
print(f"Network outputs: {network.num_outputs}")
for i in range(network.num_inputs):
    t = network.get_input(i)
    print(f"  Input  {i}: {t.name}  shape={t.shape}  dtype={t.dtype}")
for i in range(network.num_outputs):
    t = network.get_output(i)
    print(f"  Output {i}: {t.name}  shape={t.shape}  dtype={t.dtype}")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
# FP16 is automatically enabled in TRT 11+ when the GPU supports it (RTX A3000 does)

print("Building engine (this may take 5-15 minutes)...")
engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    raise RuntimeError("Engine build failed — check logs above.")

with open(ENGINE_PATH, "wb") as f:
    f.write(engine_bytes)

print(f"Engine saved to: {ENGINE_PATH}")