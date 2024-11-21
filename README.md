

# Team - The Overfitters

Project Title : Image Captioning   
Team Mentor   : Amit Pandey

Team Members :  
Sakshi Warkari     : 0015  
Prachi Dhore     : 0004  
Sakshi Jiwtode   : 0006  
Himanshi Hatwar  : 0008  
Himanshu Katrojwar :0002 

This project focuses on generating captions for images by combining computer vision (feature extraction with VGG16) and natural language processing (sequence modeling with LSTMs and attention). The system learns to describe images in natural language by training on a large dataset of images and their corresponding captions.

Table of Contents
Project Overview
Features
Technologies Used
Dataset
Model Architecture
Training Procedure
Evaluation
How to Run
Results
Future Improvements
Acknowledgments
1. Project Overview
Image captioning bridges the gap between computer vision and natural language processing. Given an image, the model generates a meaningful caption describing its contents. This project leverages pre-trained feature extractors (VGG16), recurrent networks (LSTM), and an attention mechanism to generate high-quality captions.

2. Features
Feature Extraction: Uses pre-trained VGG16 to extract global image features.
Sequence Modeling: Employs LSTM for generating captions word by word.
Attention Mechanism: Dynamically focuses on different parts of the image while generating captions.
Evaluation: Uses BLEU scores to assess the quality of generated captions.
Custom Generators: Efficient handling of large datasets during training.
3. Technologies Used
Python 3.8+
TensorFlow/Keras
NumPy, Pandas, Matplotlib, Seaborn
NLTK (Natural Language Toolkit)
TQDM
4. Dataset
We use the Flickr8k Dataset, which contains:

8,000 images, each annotated with 5 captions.
Dataset splits:
Training set: 6,000 images
Validation set: 1,000 images
Test set: 1,000 images
Preprocessing Steps:

Resizing and normalizing images for VGG16 input.
Tokenizing captions and creating word-to-index mappings.
Padding sequences for consistent input size.
5. Model Architecture
a. Feature Extraction (VGG16):
A pre-trained VGG16 model extracts a fixed-size feature vector for each image.
The output is passed through a Global Average Pooling layer to reduce dimensionality.
b. Caption Generator (LSTM with Attention):
Embedding Layer: Converts tokenized words into dense vector representations.
LSTM: Predicts the next word based on the current word and image features.
Attention Mechanism: Dynamically weights different parts of the image for each word generation step.
6. Training Procedure
Custom Data Generator:
Supplies batches of image features and corresponding captions to the model.

Loss Function:
Cross-entropy loss for word prediction.

Callbacks Used:

ModelCheckpoint: Saves the best model based on validation loss.
EarlyStopping: Stops training if validation loss doesn't improve for 5 epochs.
ReduceLROnPlateau: Reduces learning rate when the model plateaus.
Hyperparameters:

Epochs: 50
Batch Size: 64
Learning Rate: Adaptive
7. Evaluation
The model is evaluated using BLEU scores, which measure the overlap between generated and reference captions:

BLEU-1: Precision for unigrams (single words).
BLEU-2: Precision for bigrams (two-word sequences).
8. How to Run
Prerequisites:
Install required libraries:

bash
Copy code
pip install tensorflow pandas numpy nltk tqdm matplotlib
Download the Flickr8k Dataset and extract it into the project directory.

Steps:
Preprocess Data:
Run the script to preprocess images and captions:

bash
Copy code
python preprocess_data.py
Train the Model:
Train the image captioning model:

bash
Copy code
python train_model.py
Evaluate the Model:
Evaluate performance using BLEU scores:

bash
Copy code
python evaluate_model.py
Generate Captions:
Generate captions for a specific image:

bash
Copy code
python generate_caption.py --image_path <path_to_image>
9. Results
Sample Captions:
Image: A dog playing in the grass.

Actual Captions:
A brown dog is running through the grass.
A dog is playing outdoors.
Predicted Caption:
A dog running in the grass.
BLEU Scores:
BLEU-1: 0.35
BLEU-2: 0.16
10. Future Improvements
Use larger datasets (e.g., MS COCO) for better generalization.
Implement transformers like BERT or ViT for enhanced performance.
Fine-tune attention mechanisms for better focus on image regions.
11. Acknowledgments
Flickr8k Dataset for providing annotated images.
The creators of TensorFlow/Keras and VGG16 for pre-trained models.
Inspiration from academic research on image captioning using deep learning.

