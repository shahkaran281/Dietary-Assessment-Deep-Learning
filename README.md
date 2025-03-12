# Dietary Assessment Using Deep Learning

## Overview

This project utilizes deep learning to classify food items from images and provides a dietary assessment. The system takes images of food, processes them using Convolutional Neural Networks (CNN), and returns nutritional information like calories, fat, carbohydrates, and protein. This project leverages **MobileNetV2** for food recognition, providing high accuracy with minimal computational overhead.

### Key Features
- **Image-Based Food Classification**: Recognizes multiple food items from images taken by the user.
- **Nutritional Assessment**: Provides detailed nutritional information such as calories, fats, proteins, and carbohydrates for each recognized food item.
- **Mobile-Friendly Model**: The MobileNetV2 model is lightweight and optimized for mobile devices, enabling fast and accurate classification.

## Technologies Used
- **Python**: The programming language used for implementing the deep learning model and processing images.
- **TensorFlow/Keras**: Frameworks for building and training the neural network.
- **OpenCV**: Used for processing and handling images.
- **NumPy**: Used for numerical operations on data.
- **Matplotlib**: Used for plotting graphs and visualizing results.
- **Pandas**: For data handling and analysis.
- **Google Colab**: Used for training the model due to the availability of free GPUs.

## Setup

To get started, clone this repository and set up the environment:

### Step 1: Clone the repository

```bash
git clone https://github.com:shahkaran281/Dietary-Assessment-Deep-Learning.git
cd Dietary-Assessment-Deep-Learning
```

### Step 2: Install dependencies

Create a virtual environment and install the necessary dependencies:

```bash
# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required libraries
pip install -r requirements.txt
```

### Step 3: Run the notebook

The project includes a Jupyter notebook (`Dietary-Assessment-Deep-Learning.ipynb`) that walks you through the process of training and testing the model. Open and run the notebook in a Jupyter environment or Google Colab.

```bash
jupyter notebook Dietary-Assessment-Deep-Learning.ipynb
```

### Step 4: Use the food classification model

Once the model is trained, you can use it to classify food images and obtain nutritional data by passing new images to the trained model.

## Dataset

The dataset used in this project is the **Food20 Dataset**, a collection of 20 popular food items, each with 100 images. For this project, we selected 10 food items for training and testing purposes.

- **Source**: [Food20 Dataset](https://www.kaggle.com/cdart99/food20dataset)

## Results

The model has achieved a classification accuracy of **92%** on the test set, and it can classify images with high precision. The results can be visualized using graphs showing the training and validation accuracy/loss, as well as the confusion matrix.

## Future Improvements

- **Expand the Dataset**: Add more food categories to improve the model’s accuracy and versatility.
- **Real-time Mobile App**: Develop a mobile app that utilizes the trained model for real-time food classification using the phone's camera.
- **Personalized Recommendations**: Incorporate user-specific dietary suggestions based on their food preferences and health metrics such as age, weight, and BMI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
