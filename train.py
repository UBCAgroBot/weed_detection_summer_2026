from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data = "data.yaml",
    epochs = 50,
    imgsz = 640, # pixels
    batch = 16,
    project = "runs",
    name = "weed_v1"
)

metrics = model.val()
print("mAP50:", metrics.box.map50) # correct if the predicted box overlaps the real box by at least 50%
print("mAP50-95:", metrics.box.map) # correct if the predicted box overlaps the real box up to 95%

model.export(format="onnx")