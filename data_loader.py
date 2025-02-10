import os
import yaml
import argparse
import numpy as np
import cv2
from PIL import Image

def load_images(image_dir, mask_dir, size):
    image_dataset, mask_dataset = [], []

    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.bmp')])

    assert len(images) == len(masks), "Mismatch in image and mask counts"

    for i in range(len(images)):
        # Load and preprocess image
        image = cv2.imread(os.path.join(image_dir, images[i]), 1)
        image = Image.fromarray(image).resize((size, size))
        image_dataset.append(np.array(image) / 255.0)  

        # Load and preprocess mask
        mask = cv2.imread(os.path.join(mask_dir, masks[i]), 0)
        mask = Image.fromarray(mask).resize((size, size))
        mask_dataset.append(np.expand_dims(np.array(mask), axis=3) / 255.0)  

    return np.array(image_dataset), np.array(mask_dataset)

if __name__ == "__main__":
    # Argument parser for config file
    parser = argparse.ArgumentParser(description="Load and preprocess dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract paths and parameters from config
    SIZE = config["Train"]["image_size"]
    train_image_dir = config["Data"]["train_image_dir"]
    train_mask_dir = config["Data"]["train_mask_dir"]
    val_image_dir = config["Data"]["val_image_dir"]
    val_mask_dir = config["Data"]["val_mask_dir"]
    test_image_dir = config["Data"]["test_image_dir"]
    test_mask_dir = config["Data"]["test_mask_dir"]
    dataset_path = config["Data"]["dataset_path"]

    # Load datasets using paths from YAML
    X_train, y_train = load_images(train_image_dir, train_mask_dir, SIZE)
    X_validation, y_validation = load_images(val_image_dir, val_mask_dir, SIZE)
    X_test, y_test = load_images(test_image_dir, test_mask_dir, SIZE)

    # Save dataset in .npz format
    np.savez_compressed(dataset_path, 
                        X_train=X_train, y_train=y_train, 
                        X_validation=X_validation, y_validation=y_validation, 
                        X_test=X_test, y_test=y_test)

    print(f"Data successfully loaded and saved at {dataset_path}!")

