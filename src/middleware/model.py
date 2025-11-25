from src.modeling.hotpot_yolo.yolo_definition import create_yolo_model
from src.middleware.config import YoloConfig


yolo_model = create_yolo_model(
    model_path=YoloConfig().model_version,
    verbose=True,
    task="detect"
)