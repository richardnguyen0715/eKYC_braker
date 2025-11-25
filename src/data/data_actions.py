import numpy as np
import cv2


def load_image(image_path: str, mode: str = "color") -> np.ndarray:
    """Load an image from the specified path with the given mode.

    Args:
        image_path (str): The path to the image file.
        mode (str): The mode to load the image in. Options are 'color', 'grayscale', or 'unchanged'.
    Returns:
        image (ndarray): The loaded image.
    """
    
    if mode == "color":
        flag = cv2.IMREAD_COLOR
    elif mode == "grayscale":
        flag = cv2.IMREAD_GRAYSCALE
    elif mode == "unchanged":
        flag = cv2.IMREAD_UNCHANGED
    else:
        raise ValueError("Invalid mode. Choose from 'color', 'grayscale', or 'unchanged'.")

    image = cv2.imread(image_path, flag)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    return image


def show_image(window_name: str, image: np.ndarray) -> None:
    """Display an image in a window.

    Args:
        window_name (str): The name of the window.
        image (np.ndarray): The image to display.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()