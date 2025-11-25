from src.middleware.model import yolo_model
from src.middleware.config import YoloConfig
from src.modeling.hotpot_yolo.yolo_pipeline import train_yolo_model


def main():
    train_yolo_model(
        model=yolo_model,
        batch_size=YoloConfig().batch_size,
        epochs=YoloConfig().epochs,
        imgsz=YoloConfig().img_size,
        data_path=YoloConfig().data_path,
        workers=YoloConfig().workers,
        optimizer=YoloConfig().optimizer,
        lr0=YoloConfig().lr0,
        patience=YoloConfig().patience
    )
    
if __name__ == "__main__":
    main()