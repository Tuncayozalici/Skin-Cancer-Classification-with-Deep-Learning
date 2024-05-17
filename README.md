# 👩‍⚕️ Skin Cancer Classification with Deep Learning 🔬

# This project aims to classify skin lesions as benign or malignant using a Convolutional Neural Network (CNN). 🎯

## 📂 Project Structure

* **data/**
    * **train/**
        * **benign/** (contains images of benign skin lesions)
        * **malignant/** (contains images of malignant skin lesions)
    * **test/**
        * **benign/** 
        * **malignant/**
* **SkinCancer.ipynb**: Jupyter Notebook containing the complete code and analysis.

## 🚀 How to Use

Clone the repository:

```bash
git clone [repository link]
cd skin-cancer-classification
```
Prepare data: Place your skin lesion images in the data/ directory as described above. Ensure that the data is correctly labeled into "benign" and "malignant" folders.

Run the Notebook: Open and run the SkinCancer.ipynb notebook using Jupyter Notebook or JupyterLab. The notebook will:
- Load and preprocess images.
- Build, train, and evaluate the CNN model.
- Show results such as accuracy, loss, confusion matrix, and classification report.

## 🛠️ Key Steps in the Code

Image Loading: Images are loaded using the Python Imaging Library (PIL) and converted to NumPy arrays.

Data Preparation:
- Images are split into training and testing sets.
- Labels (benign or malignant) are assigned.
- Data is shuffled to introduce randomness during training.

Model Building: A CNN model is built with the following layers:
- Convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Dropout layers for regularization.
- Dense layers for classification.

Model Training: The model is trained using the Adam optimizer and binary crossentropy loss.

Model Evaluation: Performance is evaluated using accuracy and loss graphs. Confusion matrix and classification report are generated.


## 💡 Opportunities for Improvement
Data Augmentation: Apply techniques like rotation, flipping, and cropping to increase the diversity of training data.
Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and model architectures.
Pretrained Models: Consider using pretrained models (e.g., VGG, ResNet) for transfer learning.

## ⚠️ Important Notes
Disclaimer: This model is for educational purposes only and should not be used for real medical diagnosis. Always consult a qualified healthcare professional.


## Contributing 🤝

Contributions are welcome! Please open a Pull Request or report issues.

## License 📄

This project is licensed under the MIT License.

## Contact 📬

Tuncay Özalıcı - tuncay.ozalici@gmail.com - [GitHub](https://github.com/Tuncayozalici) - [LinkedIn](https://www.linkedin.com/in/tuncay-özalıcı)
