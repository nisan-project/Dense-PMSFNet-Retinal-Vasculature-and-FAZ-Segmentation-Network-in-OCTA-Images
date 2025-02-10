import time  
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from model import DensePMSFNet
from metrics_and_losses import dice_coef, accuracy, specificity, sensitivity, iou, total_loss

def main():
    # Argument parser for custom config file
    parser = argparse.ArgumentParser(description="Train DensePMSFNet Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Load dataset from YAML path
    dataset_path = config["Data"]["dataset_path"]
    data = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_validation, y_validation = data["X_validation"], data["y_validation"]

    # Model parameters
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = config["Train"]["image_size"], config["Train"]["image_size"], config["Train"]["img_channels"]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    batch_size = config["Train"]["batch_size"]
    epochs = config["Train"]["epochs"]
    learning_rate = config["Train"]["learning_rate"]
    seed = config["Train"]["seed"]
    patience = config["Train"]["patience"]
    checkpoint_path = config["checkpoint_path"]

    # Data Augmentation 
    train_datagen = ImageDataGenerator(
        rotation_range=config["Train"]["rotation_range"],
        horizontal_flip=config["Train"]["horizontal_flip"],
        shear_range=config["Train"]["shear_range"],
        fill_mode=config["Train"]["fill_mode"]
    )

    mask_datagen = ImageDataGenerator(
        rotation_range=config["Train"]["rotation_range"],
        horizontal_flip=config["Train"]["horizontal_flip"],
        shear_range=config["Train"]["shear_range"],
        fill_mode=config["Train"]["fill_mode"]
    )

    # Custom Generator
    def custom_generator(X, y, image_datagen, mask_datagen, batch_size, seed):
        gen_image = image_datagen.flow(X, batch_size=batch_size, seed=seed, shuffle=config["Train"]["shuffle"])
        gen_mask = mask_datagen.flow(y, batch_size=batch_size, seed=seed, shuffle=config["Train"]["shuffle"])
        
        for img_batch, mask_batch in zip(gen_image, gen_mask):
            yield img_batch, mask_batch

    # Model 
    model = DensePMSFNet(input_shape)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=total_loss,
                  metrics=[dice_coef, accuracy, specificity, sensitivity, iou])

    # Callbacks
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_dice_coef', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_dice_coef', patience=patience, mode='max', restore_best_weights=True, verbose=1)

    # TensorBoard Logging Setup
    log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")  
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Training
    steps_per_epoch = np.ceil(len(X_train) / batch_size)

    history = model.fit(
        custom_generator(X_train, y_train, train_datagen, mask_datagen, batch_size, seed),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_validation, y_validation),
        callbacks=[model_checkpoint, early_stopping, tensorboard_callback],  
        verbose=1
    )

    print("Training complete! Logs saved in:", log_dir)

if __name__ == "__main__":
    main()