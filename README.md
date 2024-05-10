
# Cat-Dog Image Classifier

## Overview
This project utilizes Keras to build a simple convolutional neural network for classifying images as either cats or dogs. It aims to provide a practical learning experience in understanding neural networks. The dataset used contains a small subset of 100 images for both training and testing, which accommodates the limitations of a slower computer. I am using ChatGPT to assist me throughout this project.

## Installation

Before running this project, ensure you have the following packages installed:

- TensorFlow
- Keras
- Pandas
- scikit-learn

You can install these packages via pip:

```bash
pip install tensorflow keras pandas scikit-learn
```

## Dataset

Please download the dataset for training from the following site: [Cat and Dog Images Dataset on Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog). Ensure you place the images in the respective directories (`./cat_dog/training_set` and `./cat_dog/test_set`) and adjust the folder structure as needed to be compatible with the project's path setup.

The dataset consists of images of cats and dogs located in separate directories under `./cat_dog/training_set` and `./cat_dog/test_set`. The model expects images in `.png`, `.jpg`, or `.jpeg` formats.

## Usage

To run the model:

1. Place your images in the respective directories.
2. Run the script to preprocess the images, train the model, and evaluate its performance.
3. Use the final section of the script to make predictions on individual images.

### Training the Model
The model is trained over 10 epochs with a batch size of 32 for training and 10 for validation.

### Evaluating the Model
After training, the model's performance is evaluated on the test dataset.

### Making Predictions
To make predictions, change the `img_path` variable to the path of your image. The script will process the image and provide a class prediction.

## Model Architecture

The model consists of the following layers:
- 2D Convolutional Layer (32 filters, 3x3 kernel)
- MaxPooling Layer (2x2 pool size)
- 2D Convolutional Layer (64 filters, 3x3 kernel)
- MaxPooling Layer (2x2 pool size)
- Flatten Layer
- Dense Layer (1 unit, sigmoid activation)

The model uses the Adam optimizer and binary crossentropy loss function.

## Results
Results are printed directly in the console, showing the accuracy of the model on the validation set and predictions for individual images.

## Contact
For support or to report issues, please reach out through the project repository issues section.# cat_dog_image_classification
