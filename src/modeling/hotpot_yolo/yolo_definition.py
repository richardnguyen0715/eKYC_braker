from ultralytics import YOLO


def create_yolo_model(model_path: str, verbose: bool, task: str) -> YOLO:
    """
    Create and return a YOLO model instance.

    Args:
        model_path (str): Path to the YOLO model weights or configuration.
        verbose (bool): Whether to enable verbose output.
        task (str): The task type for the YOLO model (e.g., 'detect', 'segment', 'classify', 'pose', 'obb').

    Returns:
        YOLO: An instance of the YOLO model.
    """
    model = YOLO(model_path, verbose=verbose, task=task)
    return model