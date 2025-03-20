# environmental sound classification with CNNs  

this project processes and classifies **environmental sounds** using a **cnn**. it extracts features from audio and trains a deep learning model to recognize different sound categories.  

## dataset  
uses the **esc-50 dataset** ([link](https://github.com/karolpiczak/ESC-50)) with **50 sound classes** like animals, nature, human, and urban sounds. each clip is **5 seconds, 44.1 kHz**.  

## methods  
- **audio processing**: noise reduction, mel spectrogram extraction  
- **model**: cnn with batch normalization, adaptive pooling  
- **training**: cross-entropy loss, adam optimizer  
- **evaluation**: confusion matrix, per-class accuracy  

## installation  
clone the repository and install dependencies  
```bash
pip install torch librosa numpy pandas scikit-learn seaborn matplotlib noisereduce
