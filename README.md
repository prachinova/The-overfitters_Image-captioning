

# Team - The Overfitters

Project Title : Image Captioning   
Team Mentor   : Amit Pandey

Team Members :  
Sakshi Warkari     : 0015  
Prachi Dhore     : 0004  
Sakshi Jiwtode   : 0006  
Himanshi Hatwar  : 0008  
Himanshu Katrojwar :0002 

# **Image Captioning using LSTM and Attention Mechanism**

This project generates captions for images by combining **computer vision** (feature extraction with VGG16) and **natural language processing** (sequence modeling with LSTMs and attention). The system learns to describe images in natural language by training on a large dataset of images and their corresponding captions.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## **Project Overview**
Image captioning bridges the gap between computer vision and natural language processing. Given an image, the model generates a meaningful caption describing its contents. This project leverages pre-trained feature extractors (VGG16), recurrent networks (LSTM), and an attention mechanism to generate high-quality captions.

---

## **Features**
- **Feature Extraction:** Pre-trained VGG16 to extract global image features.
- **Sequence Modeling:** LSTM for generating captions word by word.
- **Attention Mechanism:** Dynamically focuses on different parts of the image while generating captions.
- **Evaluation:** BLEU scores to assess the quality of generated captions.
- **Custom Data Generators:** Efficient handling of large datasets during training.

---

## **Technologies Used**
- **Python 3.8+**
- **TensorFlow/Keras**
- **NumPy, Pandas, Matplotlib, Seaborn**
- **NLTK (Natural Language Toolkit)**
- **TQDM**

---

## **Dataset**
We use the **Flickr8k Dataset**, which contains:
- 8,000 images, each annotated with 5 captions.
- Dataset splits:
  - Training set: 6,000 images
  - Validation set: 1,000 images
  - Test set: 1,000 images

**Preprocessing Steps:**
1. Resizing and normalizing images for VGG16 input.
2. Tokenizing captions and creating word-to-index mappings.
3. Padding sequences for consistent input size.

---

## **Model Architecture**
### **Feature Extraction (VGG16)**
- A pre-trained VGG16 model extracts a fixed-size feature vector for each image.
- The output is passed through a **Global Average Pooling layer** to reduce dimensionality.

### **Caption Generator (LSTM with Attention)**
1. **Embedding Layer:** Converts tokenized words into dense vector representations.
2. **LSTM:** Predicts the next word based on the current word and image features.
3. **Attention Mechanism:** Dynamically weights different parts of the image for each word generation step.

---

## **Training Procedure**
### **Custom Data Generator**
Efficiently supplies batches of image features and corresponding captions to the model.

### **Loss Function**
- Cross-entropy loss for word prediction.

### **Callbacks Used**
- **ModelCheckpoint:** Saves the best model based on validation loss.
- **EarlyStopping:** Stops training if validation loss doesn't improve for 5 epochs.
- **ReduceLROnPlateau:** Reduces learning rate when the model plateaus.

### **Hyperparameters**
- Epochs: 50
- Batch Size: 64
- Learning Rate: Adaptive

---

## **Evaluation**
The model is evaluated using **BLEU scores**, which measure the overlap between generated and reference captions:
- **BLEU-1:** Precision for unigrams (single words).
- **BLEU-2:** Precision for bigrams (two-word sequences).

---
## **Contributions**
This project was a collaborative effort. Each team member played a vital role in its successful completion. Below are the detailed contributions from each team member:

##1. Himanshi Hatwar
Responsible for creating an engaging and informative presentation (PPT) that effectively summarized the project's objectives, methodology, results, and future scope.

##2. Himanshu Katrojwar
Authored the research paper for the project.
The paper detailed the problem statement, methodologies, results, and conclusions, making it a valuable document for academic and research purposes.

##3.Sakshi Jiwtode
Focused on compiling the project report.
The report documented every phase of the project, including objectives, design, implementation, challenges faced, and conclusions.

##4.Sakshi Warkari
Primarily responsible for writing the code that powered the project.
This included implementing data preprocessing, the deep learning model (CNN + LSTM with attention mechanism), training scripts, evaluation metrics (BLEU scores), and the image caption generation script.

##5.Prachi Dhore
Took charge of building the machine learning model, including the integration of the VGG16-based encoder and the LSTM-based decoder with attention mechanism.
Managed the GitHub repository for the project, ensuring all scripts, datasets, and documentation were properly organized and accessible.
Contributed to debugging and testing to ensure smooth model operation.

Each member's contributions were essential to the project’s success, reflecting a collaborative and well-coordinated effort.







