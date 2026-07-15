FROM forklift-gpu

ENTRYPOINT []

CMD ["python3","inference-scripts/inference-rt.py","--engine","nanodet-v1_fp16_pc2.engine"]