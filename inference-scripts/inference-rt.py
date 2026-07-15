import argparse
import ast
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import tensorrt as trt
from cuda.bindings import driver as cuda

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class _DotDict:
    """Recursive dot-access wrapper around a plain dict (mirrors CfgNode behaviour)."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, self._wrap(v))

    @classmethod
    def _wrap(cls, v):
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(i) for i in v]
        return v

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return f"_DotDict({self.__dict__})"


def load_config(cfg_obj: _DotDict, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(data)}")
    loaded = _DotDict(data)
    cfg_obj.__dict__.update(loaded.__dict__)


cfg = _DotDict({})


class SimpleLogger:
    def log(self, msg):
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="NanoDet TensorRT RTSP inference with ROI overlays")
    default_config = os.environ.get('config_file') or str(ROOT_DIR / 'config.yml')
    default_engine = os.environ.get('engine') or str(ROOT_DIR / 'nanodet-v1' / 'nanodet-v1_fp16_pc.engine')
    default_area = str(ROOT_DIR / 'Regions' / 'area.txt')
    default_exc = str(ROOT_DIR / 'Regions' / 'exc_points.txt')
    default_score = os.environ.get('score', 0.45)

    parser.add_argument("--config", default=default_config, help="NanoDet config .yml")
    parser.add_argument("--engine", default=default_engine, help="path to TensorRT .engine file")
    parser.add_argument("--rtsp", default=os.environ.get('RTSP_URL'), help="RTSP stream URL")
    parser.add_argument("--score_thres", type=float, default=default_score)
    parser.add_argument("--nms_thres", type=float, default=0.5)
    parser.add_argument("--display", nargs="?", const=True, default=None, type=parse_bool, help="show OpenCV window (supports True/False from env or CLI)")
    parser.add_argument("--save_dir", default=None, help="optional directory to save annotated frames periodically")
    parser.add_argument("--save_interval", type=float, default=5.0, help="seconds between saved snapshots")
    parser.add_argument("--reconnect_delay", type=float, default=3.0, help="seconds to wait before reconnecting RTSP stream")
    parser.add_argument("--quiet", action="store_true", help="disable per-frame console logging")
    parser.add_argument("--apply_sigmoid", action="store_true", help="apply sigmoid to class scores manually")
    parser.add_argument(
        "--violation_duration",
        default=os.environ.get('violation_duration'),
        type=float,
        help="seconds a forklift must remain outside the safe zone before alarm triggers",
    )
    parser.add_argument("--area", default=default_area, help="main ROI points file")
    parser.add_argument("--exc", default=default_exc, help="exclusion ROI points file")

    args = parser.parse_args()
    if args.display is None:
        env_display = os.environ.get("display") or os.environ.get("DISPLAY")
        args.display = parse_bool(env_display) if env_display is not None else False
    return args


def safe_get(obj, path, default=None):
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def open_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")
    return cap


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, score_thres, nms_thres):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        keep_mask = cls_scores >= score_thres
        if not np.any(keep_mask):
            continue

        cls_boxes = boxes[keep_mask]
        cls_scores = cls_scores[keep_mask]

        keep = nms(cls_boxes, cls_scores, nms_thres)
        for i in keep:
            final_dets.append([
                cls_boxes[i, 0],
                cls_boxes[i, 1],
                cls_boxes[i, 2],
                cls_boxes[i, 3],
                cls_scores[i],
                cls_id,
            ])

    if not final_dets:
        return np.zeros((0, 6), dtype=np.float32)

    return np.array(final_dets, dtype=np.float32)


def summarize_detections(dets, class_names, score_thres):
    if dets is None or len(dets) == 0:
        return "no detections"

    counts = {}
    for det in dets:
        score = float(det[4])
        cls_id = int(det[5])
        if score < score_thres:
            continue
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return "no detections"

    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def load_points(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROI file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "points" in data:
            return [(int(x), int(y)) for x, y in data["points"]]
        if isinstance(data, list):
            return [(int(x), int(y)) for x, y in data]
    except json.JSONDecodeError:
        pass

    candidate = text
    if "=" in candidate:
        candidate = candidate.split("=", 1)[1].strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        pass
    else:
        start = min(candidate.find("["), candidate.find("(")) if "[" in candidate and "(" in candidate else None
        if start is not None and start != -1:
            if candidate[start] == "[":
                end = candidate.rfind("]")
            else:
                end = candidate.rfind(")")
            if end > start:
                candidate = candidate[start:end + 1]
            else:
                raise ValueError(f"Could not parse ROI file {path}")
        else:
            raise ValueError(f"Could not parse ROI file {path}")

    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse ROI file {path}: {exc}") from exc

    if isinstance(parsed, dict) and "points" in parsed:
        parsed = parsed["points"]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Unsupported ROI format in {path}")

    return [(int(x), int(y)) for x, y in parsed]


def get_ground_points(dets, score_thres):
    results = []
    for det in dets:
        x1, y1, x2, y2, score, _ = det
        if score < score_thres:
            continue
        gx = int((x1 + x2) / 2)
        gy = int(y2)
        results.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "score": float(score),
            "ground_point": (gx, gy),
        })
    return results


def check_zone(ground_point, safe_zone, exclusion_zone):
    pt = (float(ground_point[0]), float(ground_point[1]))
    safe_poly = np.array(safe_zone, dtype=np.int32)
    exclusion_poly = np.array(exclusion_zone, dtype=np.int32)
    inside_safe = cv2.pointPolygonTest(safe_poly, pt, measureDist=False) >= 0
    inside_exclusion = cv2.pointPolygonTest(exclusion_poly, pt, measureDist=False) >= 0
    if inside_safe:
        return "safe"
    if inside_exclusion:
        return "excluded"
    return "violation"


def draw_detections(frame, dets, class_names, score_thres):
    out = frame.copy()
    for det in dets:
        x1, y1, x2, y2, score, cls_id = det
        if score < score_thres:
            continue
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        cls_id = int(cls_id)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        text = f"{label} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def _pipeline_to_list(pipeline):
    if pipeline is None:
        return []
    if isinstance(pipeline, list):
        return pipeline
    if isinstance(pipeline, _DotDict):
        return list(pipeline.__dict__.values())
    if isinstance(pipeline, dict):
        return list(pipeline.values())
    return []


def extract_normalize_params(cfg_obj):
    pipelines = []
    for path in ("data.val.pipeline", "data.train.pipeline"):
        pipelines.extend(_pipeline_to_list(safe_get(cfg_obj, path, None)))
    for step in pipelines:
        if not isinstance(step, (dict, _DotDict)):
            continue
        step_type = str(step.get("type", "")).lower()
        if step_type == "normalize":
            mean = step.get("mean", [103.53, 116.28, 123.675])
            std = step.get("std", [57.375, 57.12, 58.395])
            to_rgb = step.get("to_rgb", True)
            return mean, std, to_rgb
    return [103.53, 116.28, 123.675], [57.375, 57.12, 58.395], True


# ---------------------------------------------------------------------------
# TensorRT helpers  (uses cuda-python / cuda.bindings — no pycuda needed)
# ---------------------------------------------------------------------------

def _check(result):
    """
    Uniformly handle cuda.bindings return values.
    All driver functions return either:
      - A single CUresult  (e.g. cuInit, cuCtxSetCurrent, cuStreamSynchronize)
      - A tuple (CUresult, value)  (e.g. cuDeviceGet, cuMemAlloc, cuStreamCreate)
    We accept any of these forms and raise on non-zero error codes.
    """
    if isinstance(result, tuple):
        err = result[0]
        val = result[1] if len(result) > 1 else None
    else:
        err = result
        val = None
    # CUresult is an enum; error code is in .value
    code = err.value if hasattr(err, "value") else int(err)
    if code != 0:
        raise RuntimeError(f"CUDA error code {code}: {err}")
    return val


def load_engine(engine_path: str) -> trt.ICudaEngine:
    """Deserialise a serialised TensorRT engine from disk."""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialise TensorRT engine: {engine_path}")
    return engine


class _HostDeviceMem:
    """Thin wrapper pairing a numpy host buffer with a device pointer (int)."""

    def __init__(self, host_mem: np.ndarray, device_ptr: int, shape, dtype):
        self.host = host_mem          # numpy array (CPU)
        self.device = device_ptr      # integer device pointer (CUdeviceptr)
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"HostDeviceMem(shape={self.shape}, dtype={self.dtype})"


def _cuda_init():
    """Initialise the CUDA driver and return (device, context, are usable)."""
    _check(cuda.cuInit(0))
    dev = _check(cuda.cuDeviceGet(0))
    ctx = _check(cuda.cuDevicePrimaryCtxRetain(dev))
    _check(cuda.cuCtxSetCurrent(ctx))
    return dev, ctx


def allocate_buffers(engine: trt.ICudaEngine):
    """Allocate host/device memory for every engine binding using cuda-python."""
    dev, ctx = _cuda_init()

    inputs, outputs, bindings = [], [], []
    stream = _check(cuda.cuStreamCreate(0))

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        n_elems = int(np.prod(shape))
        nbytes = n_elems * np.dtype(dtype).itemsize

        host_mem = np.empty(n_elems, dtype=dtype)
        device_ptr = _check(cuda.cuMemAlloc(nbytes))

        bindings.append(int(device_ptr))
        hdm = _HostDeviceMem(host_mem, int(device_ptr), shape, dtype)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(hdm)
        else:
            outputs.append(hdm)

    return inputs, outputs, bindings, stream, ctx


def do_inference(context, engine, inputs, outputs, stream):
    """Copy inputs H→D, execute, copy outputs D→H (synchronous)."""
    # Set tensor addresses on the context (TRT 10+ API)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, inputs[0].device)
        else:
            context.set_tensor_address(name, outputs[0].device)

    for inp in inputs:
        _check(cuda.cuMemcpyHtoDAsync(
            inp.device,
            inp.host.ctypes.data,
            inp.host.nbytes,
            stream,
        ))

    context.execute_async_v3(stream_handle=int(stream))

    for out in outputs:
        _check(cuda.cuMemcpyDtoHAsync(
            out.host.ctypes.data,
            out.device,
            out.host.nbytes,
            stream,
        ))

    _check(cuda.cuStreamSynchronize(stream))
    return [out.host for out in outputs]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class TRTPredictor:
    def __init__(
        self,
        cfg_obj,
        engine_path,
        logger,
        score_thres=0.35,
        nms_thres=0.5,
        apply_sigmoid=False,
    ):
        self.cfg = cfg_obj
        self.logger = logger
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.apply_sigmoid = apply_sigmoid

        self.class_names = list(cfg_obj.class_names)
        self.num_classes = len(self.class_names)
        self.reg_max = int(
            safe_get(cfg_obj, "model.arch.head.reg_max",
                     safe_get(cfg_obj, "model.head.reg_max", 7))
        )
        self.strides = list(
            safe_get(cfg_obj, "model.arch.head.strides",
                     safe_get(cfg_obj, "model.head.strides", [8, 16, 32, 64]))
        )
        self.keep_ratio = bool(safe_get(cfg_obj, "data.val.keep_ratio", True))

        input_size = safe_get(cfg_obj, "data.val.input_size", [416, 416])
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_w = int(input_size[0])
        self.input_h = int(input_size[1])

        self.mean_vals, self.std_vals, self.to_rgb = extract_normalize_params(cfg_obj)
        # Pre-compute per-channel mean and inverse-std as numpy arrays (shape [3])
        self.mean = np.array(self.mean_vals, dtype=np.float32)
        self.inv_std = np.array([1.0 / max(float(s), 1e-6) for s in self.std_vals], dtype=np.float32)

        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self.center_priors, self.center_strides = self._generate_center_priors(
            self.input_w, self.input_h, self.strides
        )

        # Load TensorRT engine
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.trt_inputs, self.trt_outputs, self.bindings, self.stream, self._cuda_ctx = allocate_buffers(self.engine)

        self.logger.log(
            f"TensorRT engine loaded | engine={engine_path} "
            f"| input_size=({self.input_w},{self.input_h}) "
            f"| reg_max={self.reg_max} | strides={self.strides} "
            f"| num_inputs={len(self.trt_inputs)} | num_outputs={len(self.trt_outputs)}"
        )

    # ------------------------------------------------------------------
    def _generate_center_priors(self, input_w, input_h, strides):
        centers, stride_list = [], []
        for stride in strides:
            feat_w = int(math.ceil(input_w / stride))
            feat_h = int(math.ceil(input_h / stride))
            for y in range(feat_h):
                for x in range(feat_w):
                    centers.append([(x + 0.5) * stride, (y + 0.5) * stride])
                    stride_list.append(stride)
        return (
            np.array(centers, dtype=np.float32),
            np.array(stride_list, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def preprocess(self, frame):
        orig_h, orig_w = frame.shape[:2]

        if self.keep_ratio:
            scale = min(self.input_w / orig_w, self.input_h / orig_h)
            resized_w = int(round(orig_w * scale))
            resized_h = int(round(orig_h * scale))
        else:
            resized_w = self.input_w
            resized_h = self.input_h

        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        # Pad to input size
        canvas = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w] = resized

        # BGR → RGB if required
        if self.to_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Normalise: (pixel - mean) * inv_std  →  float32 NCHW
        img = canvas.astype(np.float32)
        img = (img - self.mean) * self.inv_std          # HWC, float32
        img = img.transpose(2, 0, 1)                    # CHW
        img = np.ascontiguousarray(img[np.newaxis, :])  # 1CHW

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "resized_h": resized_h,
            "resized_w": resized_w,
            "scale_x": resized_w / max(orig_w, 1),
            "scale_y": resized_h / max(orig_h, 1),
        }
        return meta, img

    # ------------------------------------------------------------------
    def _reshape_single_output(self, arr):
        expected_dim = self.num_classes + 4 * (self.reg_max + 1)
        arr = np.asarray(arr, dtype=np.float32)

        # Strip size-1 dimensions
        while arr.ndim > 2 and 1 in arr.shape:
            arr = np.squeeze(arr)

        if arr.ndim == 1:
            if arr.size % expected_dim != 0:
                raise RuntimeError(
                    f"Output size {arr.size} not divisible by expected dim {expected_dim}"
                )
            return arr.reshape(-1, expected_dim)

        if arr.ndim == 2:
            if arr.shape[1] == expected_dim:
                return arr
            if arr.shape[0] == expected_dim:
                return arr.T
            flat = arr.reshape(-1)
            if flat.size % expected_dim != 0:
                raise RuntimeError(
                    f"Unexpected 2D output shape {arr.shape}, expected one axis == {expected_dim}"
                )
            return flat.reshape(-1, expected_dim)

        if arr.ndim == 3:
            if arr.shape[-1] == expected_dim:
                return arr.reshape(-1, expected_dim)
            if arr.shape[0] == expected_dim:
                arr = np.transpose(arr, (1, 2, 0))
                return arr.reshape(-1, expected_dim)
            flat = arr.reshape(-1)
            if flat.size % expected_dim != 0:
                raise RuntimeError(
                    f"Unexpected 3D output shape {arr.shape}, cannot reshape to [N, {expected_dim}]"
                )
            return flat.reshape(-1, expected_dim)

        flat = arr.reshape(-1)
        if flat.size % expected_dim != 0:
            raise RuntimeError(
                f"Unexpected output ndim={arr.ndim}, size={flat.size}"
            )
        return flat.reshape(-1, expected_dim)

    # ------------------------------------------------------------------
    def postprocess(self, pred, meta):
        num_points = pred.shape[0]
        expected_points = len(self.center_priors)
        if num_points != expected_points:
            raise RuntimeError(
                f"Decoded rows mismatch: got {num_points}, expected {expected_points}. "
                "Check engine output shape / input size / strides."
            )

        cls_pred = pred[:, : self.num_classes]
        if self.apply_sigmoid:
            cls_pred = sigmoid(cls_pred)

        dis_pred = pred[:, self.num_classes:]
        dis_pred = dis_pred.reshape(num_points, 4, self.reg_max + 1)
        dis_pred = softmax(dis_pred, axis=2)
        dis_pred = np.sum(dis_pred * self.project[None, None, :], axis=2)
        dis_pred = dis_pred * self.center_strides[:, None]

        centers = self.center_priors
        x1 = centers[:, 0] - dis_pred[:, 0]
        y1 = centers[:, 1] - dis_pred[:, 1]
        x2 = centers[:, 0] + dis_pred[:, 2]
        y2 = centers[:, 1] + dis_pred[:, 3]
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] /= max(meta["scale_x"], 1e-6)
        boxes[:, [1, 3]] /= max(meta["scale_y"], 1e-6)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta["orig_w"] - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta["orig_h"] - 1)

        return multiclass_nms(
            boxes=boxes,
            scores=cls_pred,
            score_thres=self.score_thres,
            nms_thres=self.nms_thres,
        )

    # ------------------------------------------------------------------
    def inference(self, frame):
        meta, img = self.preprocess(frame)

        # Copy preprocessed image into the TRT input buffer
        np.copyto(self.trt_inputs[0].host, img.ravel())

        # Run inference
        raw_outputs = do_inference(
            self.context,
            engine=self.engine,
            inputs=self.trt_inputs,
            outputs=self.trt_outputs,
            stream=self.stream,
        )

        # Reshape and concatenate all output tensors
        preds = []
        for raw, hdm in zip(raw_outputs, self.trt_outputs):
            arr = np.array(raw, dtype=np.float32).reshape(hdm.shape)
            arr = self._reshape_single_output(arr)
            preds.append(arr)

        pred = np.concatenate(preds, axis=0)
        dets = self.postprocess(pred, meta)
        return meta, dets


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logger = SimpleLogger()
    load_config(cfg, args.config)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    area_points = load_points(args.area)
    exc_points = load_points(args.exc)
    rois = [area_points, exc_points]

    predictor = TRTPredictor(
        cfg_obj=cfg,
        engine_path=args.engine,
        logger=logger,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        apply_sigmoid=args.apply_sigmoid,
    )

    cap = open_stream(args.rtsp)
    window_name = "NanoDet TRT Detection + ROI"

    prev_time = time.time()
    last_save_time = 0.0
    needs_rendered = args.display or args.save_dir is not None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.log(f"RTSP read failed. Reconnecting in {args.reconnect_delay}s...")
                cap.release()
                time.sleep(args.reconnect_delay)
                cap = open_stream(args.rtsp)
                continue

            _, dets = predictor.inference(frame)
            forklift_data = get_ground_points(dets, args.score_thres)
            for fk in forklift_data:
                fk["status"] = check_zone(fk["ground_point"], area_points, exc_points)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if not args.quiet:
                timestamp = time.strftime("%H:%M:%S")
                logger.log(f"[{timestamp}]FPS:{fps:.1f} |  {summarize_detections(dets, cfg.class_names, args.score_thres)}")
                for fk in forklift_data:
                    logger.log(f"[{timestamp}]  ground_point={fk['ground_point']}  status={fk['status']}")

            if needs_rendered:
                out_frame = frame.copy()
                for idx, points in enumerate(rois):
                    if not points:
                        continue
                    pts = np.array(points, dtype=np.int32)
                    color = (0, 255, 0) if idx == 0 else (255, 165, 0)
                    cv2.polylines(out_frame, [pts], isClosed=True, color=color, thickness=2)
                    centroid = np.mean(pts, axis=0).astype(int)
                    cv2.putText(out_frame, str(idx + 1), (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for fk in forklift_data:
                    x1, y1, x2, y2 = fk["box"]
                    gx, gy = fk["ground_point"]
                    status = fk["status"]
                    color = (0, 255, 0) if status == "safe" else (0, 165, 255) if status == "excluded" else (0, 0, 255)
                    label = f"Forklift {fk['score']:.2f} [{status}]"
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out_frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.circle(out_frame, (gx, gy), 5, color, -1)

                out_frame = draw_detections(out_frame, dets, cfg.class_names, args.score_thres)
                cv2.putText(out_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if args.display:
                    cv2.imshow(window_name, out_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break

                if args.save_dir and (now - last_save_time) >= args.save_interval:
                    save_path = os.path.join(args.save_dir, f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(save_path, out_frame)
                    last_save_time = now

    except KeyboardInterrupt:
        logger.log("Stopped by user.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()