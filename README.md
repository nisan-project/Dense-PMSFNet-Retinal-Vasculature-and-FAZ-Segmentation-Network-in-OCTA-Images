# **Dense-PMSFNet: Retinal Vasculature and FAZ Segmentation in OCTA Images**  
This repository contains the implementation of Dense-PMSFNet for accurate segmentation of retinal microvasculature (arteries, veins, capillaries) and the Foveal Avascular Zone (FAZ) in OCTA images.

### **File Descriptions**  

- **data_loader.py**: Preprocesses OCTA images and saves them as a compressed `.npz` file.  
- **train.py**: Trains the Dense-PMSFNet model using configurations from `config.yaml` and preprocessed data.  
- **test.py**: Evaluates the trained model on the test set and generates predictions.  
- **config.yaml**: Contains training and testing parameters (dataset paths, hyperparameters, model settings).  
- **requirements.txt**: Lists Python dependencies required for running the code.  
- **models/**: Folder with model architectures used for comparison.  
- **model.py**: Implementation of the Dense-PMSFNet model architecture for retinal vasculature and FAZ segmentation.
  
### **Clone the Repository**  
```bash
git clone github.com/nisan-project/Dense-PMSFNet-Retinal-Vasculature-and-FAZ-Segmentation-Network-in-OCTA-Images.git
```
### **Create a Virtual Environment & Install Dependencies**  
```bash
python -m venv env  
source env/bin/activate  # For Linux/Mac  
env\Scripts\activate     # For Windows  
pip install -r requirements.txt  
```
### **Preprocess & Save the Dataset**  
Run the following command to preprocess and save the dataset as a compressed NumPy file (`dataset.npz`):  
```bash
python data_loader.py --config config.yaml
```

## **Training the Model**  

### **Run Training Script**  
To train **Dense-PMSFNet**, use:  
```bash
python train.py --config config.yaml
```
## **Testing the Model**  
After training, evaluate the model on the test set and generate predictions using:  
```bash
python test.py --config config.yaml
```



