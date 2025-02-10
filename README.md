# **Dense-PMSFNet: Retinal Vasculature and FAZ Segmentation in OCTA Images**  

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



