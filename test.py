import argparse
import yaml
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from metrics_and_losses import dice_coef, accuracy, specificity, sensitivity, iou, total_loss  
from PIL import Image

def main():
    # Argument parser for config file
    parser = argparse.ArgumentParser(description="Evaluate DensePMSFNet Model on Test Data")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Load the test dataset from YAML path
    dataset_path = config["Data"]["dataset_path"]
    data = np.load(dataset_path)
    X_test, y_test = data["X_test"], data["y_test"]

    # Load the best model from checkpoint
    model_path = config["checkpoint_path"]  
    model = load_model(model_path, custom_objects={
        'dice_coef': dice_coef,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'iou': iou
    })

    # Evaluate the model on the test set
    results = model.evaluate(X_test, y_test, verbose=1)
    metrics_names = model.metrics_names

    # Print the metrics
    print("\n Model Evaluation Results on Test Set:")
    for name, value in zip(metrics_names, results):
        print(f"{name}: {value:.4f}")

    # Create an output directory for predictions
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save predictions
    print("\n Saving Predicted Masks...")

    for i in range(len(X_test)):
        pred_mask = model.predict(np.expand_dims(X_test[i], axis=0))[0]  
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  
        save_path = os.path.join(output_dir, f"prediction_{i}.png")
        Image.fromarray(pred_mask.squeeze()).save(save_path)

    print(f"\n Predictions saved in: {output_dir}")

if __name__ == "__main__":
    main()
