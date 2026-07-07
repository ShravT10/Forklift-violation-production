In this we are going to create our final ROI logic

Command to run the inference 

```
python inference-scripts/ncnn_inference.py --config nanodet-plus-m_416_custom.yml --param ncnn-model\nanodet.ncnn.param --bin ncnn-model\nanodet.ncnn.bin --rtsp "rtsps://admin:Kion%402024@10.102.10.230:554/video/live?channel=1&subtype=1" --display 
```

Command to run the inference with ROI logic 

```
python inference-scripts/inference-with-roi-from-files.py --display
```


Command to capture frame from feed 

```
python dataset-maker/snap.py --rtsp "rtsps://admin:Kion%402024@10.102.10.230:554/video/live?channel=1&subtype=1" --save_dir "dataset-maker/dataset" --duration 5 --save_interval 1 
```