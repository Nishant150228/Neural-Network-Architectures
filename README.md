# Neural Network Architectures for Deep Learning Tasks

## Overview
This project involves constructing neural network models using both Sequential and Functional APIs, implementing models for handwritten digit recognition, and exploring various parameters in model compilation. Below are the details of each task undertaken:

---

### 1. Neural Network Construction (Sequential and Functional APIs)
#### Task
To build a neural network with the following architecture using both Sequential and Functional APIs:
- **H-Layer-1**: 512 neurons
- **H-Layer-2**: 512 neurons
- **H-Layer-3**: 1024 neurons
- **O-Layer**: 10 neurons (for multi-class classification)

#### Sequential API Model
- Constructed using TensorFlow's Keras library.
- Three hidden dense layers with ReLU activations.
- Output layer with a softmax activation for multi-class classification.
- Includes weight initialization (`he_normal`), bias initialization (`zeros`), and L2 regularization.
- Total trainable parameters: **849,930**.

#### Functional API Model
- Similar architecture and parameter settings as the Sequential API model.
- Utilizes Input, Model, and layers.Dense classes, enabling more flexibility.
- Total trainable parameters: **849,930**.

---

### 2. Handwritten Digit Recognition with Three Architectures
#### Dataset
- **Dataset Used**: MNIST (Handwritten Digits)
- Preprocessing: Normalized pixel values and one-hot encoding of labels.

#### Architectures and Performance
1. **Simple Dense Layers**
   - Baseline architecture with no additional regularization.
   - Achieved a test accuracy of **97.83%**.

2. **Model with Dropout**
   - Added dropout layers to prevent overfitting.
   - Test accuracy improved to **97.91%**.

3. **Model with Batch Normalization**
   - Added batch normalization layers for stable training.
   - Test accuracy: **97.90%**.

---

### 3. Exploring `model.compile()` Parameters
#### Task
To explore the following combinations of optimizer, loss, and metric parameters:

| Optimizer  | Loss Function          | Metric      |
|------------|------------------------|-------------|
| Adam       | Categorical Crossentropy | Accuracy    |
| SGD        | Binary Crossentropy    | Precision   |
| RMSprop    | Mean Squared Error     | Recall      |

#### Observations
1. **Adam with Categorical Crossentropy and Accuracy**
   - Standard choice for multi-class classification.
   - Balances speed and accuracy during optimization.

2. **SGD with Binary Crossentropy and Precision**
   - Focuses on binary classification tasks.
   - Precision metric evaluates how many positive predictions were correct.

3. **RMSprop with Mean Squared Error and Recall**
   - Suited for regression tasks but used here for experimentation.
   - Recall metric measures the proportion of actual positives correctly identified.

---

## Key Highlights
- The Sequential and Functional API models demonstrated equivalent capacity and complexity, emphasizing flexibility in model design.
- Dropout and batch normalization improved generalization and stabilized training for the handwritten digit recognition task.
- The choice of optimizer, loss function, and metric significantly impacts model performance and suitability for specific tasks.

## Repository Details
### Files Included
1. **Sequential and Functional API Model Construction**
2. **Handwritten Digit Recognition Models**
3. **`model.compile()` Parameter Exploration**

### Tools Used
- TensorFlow and Keras libraries for model construction and training.
- MNIST dataset for handwritten digit recognition.

---

### Acknowledgments
This project demonstrates the practical application of neural network design principles and performance optimization strategies for deep learning tasks. Suggestions for further improvements are welcome.

