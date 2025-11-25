from ultralytics import YOLO
from src.middleware.logger import model_builder_logger


def train_yolo_model(
    model: YOLO,
    batch_size: int,
    epochs: int,
    imgsz: int = 640,
    data_path: str = "",
    workers: int = 4,
    optimizer: str = "SGD",
    lr0: float = 0.01,
    patience: int = 10
    ) -> None:
    """
    Train the YOLO model using the specified training and validation data.

    Args:
        model (YOLO): An instance of the YOLO model to be trained.
    """
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience
    )
    model_builder_logger.info("YOLO model training completed.")


def evaluate_yolo_model(
    model: YOLO,
    data_path: str = "",
    batch_size: int = 16,
    imgsz: int = 640,
    workers: int = 4
    ) -> None:
    """
    Evaluate the YOLO model using the specified validation data.

    Args:
        model (YOLO): An instance of the YOLO model to be evaluated.
    """
    results = model.val(
        data=data_path,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers
    )
    model_builder_logger.info("YOLO model evaluation completed.")
    return results


def inference_yolo_model(
    model: YOLO,
    image_path: str
    ) -> None:
    """
    Perform inference using the YOLO model on the given image.

    Args:
        model (YOLO): An instance of the YOLO model for inference.
        image_path (str): Path to the input image for inference.
    """
    results = model.predict(source=image_path)
    model_builder_logger.info("YOLO model inference completed.")
    return results


def show_inference_results(
    results: YOLO,
    show: bool = True,
    save: bool = False,
    save_dir: str = "./inference_results"
    ) -> None:
    """
    Display or save the inference results.

    Args:
        results (YOLO): The results from the YOLO model inference.
        show (bool): Whether to display the results.
        save (bool): Whether to save the results to disk.
        save_dir (str): Directory to save the results if save is True.
    """
    for result in results:
        if show:
            result.show()
        if save:
            result.save(save_dir=save_dir)
    model_builder_logger.info("Inference results processed.")