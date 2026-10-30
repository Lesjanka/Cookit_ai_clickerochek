from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    result = model.train(
        data = 'data.yaml',
        epochs = 550,
        imgsz = 640,
        batch = 16,
        workers = 0,
        name ='cookie_detecror',
        patience = 150
    )
