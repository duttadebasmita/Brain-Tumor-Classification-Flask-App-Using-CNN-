
# Brain Tumor Classification On Flask App



Built a Tumor detection And Classification model using a convolutional neural network in Tensorflow & Keras. The model with the best accuracy was then developed to a Keras App with Viable User Interface
Used a brain MRI images data founded on Kaggle.



# Data Info:


The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous.

# Need For Data Augmentation 

Augmentation provided better results of Accuracy on both the train And Validation Datasets. The data was split in a certain ration of test and validation that the model was created on the basis of training on the augmented images. The system on which this project was developed on doesnt have any Graphics Driver and as a result data preprocessing and augmentation took a long time. 

Since this is a small dataset, There wasn't enough examples to train the neural network. Also, data augmentation was useful in taclking the data imbalance issue in the data.

Further explanations are found in the Data Augmentation notebook.

Before data augmentation, the dataset consisted of:
155 positive and 98 negative examples, resulting in 253 example images.

After data augmentation, now the dataset consists of:
1085 positive and 980 examples, resulting in 2065 example images.

Note: these 2065 examples contains also the 253 original images. They are found in folder named 'augmented data'.


#  For Preprocessing The Data

For every image, the following preprocessing steps were applied:

Crop the part of the image that contains only the brain (which is the most important part of the image).
Resize the image to have a shape of (240, 240, 3)=(image_width, image_height, number of channels): because images in the dataset come in different sizes. So, all images should have the same shape to feed it as an input to the neural network.
Apply normalization: to scale pixel values to the range 0-1.
Data Split:
The data was split in the following way:

70% of the data for training.

15% of the data for validation.

15% of the data for testing.

# Understanding the architecture:
Each input x (image) has a shape of (64, 64, 3) and is fed into the neural network. And, it goes through the following layers:

A convolutional layer with 32 filters, with a filter size of (3, 3) and a stride equal to 1. 

The layer of filters was kept on increasing from 32 to 64 to 128, no hidden layer was used.

A batch normalization layer to normalize pixel values to speed up computation.

A ReLU activation layer.

A Max Pooling layer with f=2 and s=2.

A Max Pooling layer with f=2 and s=2, same as before.

A flatten layer in order to flatten the 3-dimensional matrix into 
a one-dimensional vector. 

A dropout of 50% is applied to remove overfitting.

Relu Activation function (binary classification) was applied to the flatten layer and a softmax function to pipeline the final output of the neural network.

Adam was used as an optimizer.

# Need For Using Such An Architecture

*Firstly, I applied transfer learning using a ResNet50 and vgg-16, but these models were too complex to the data size and were overfitting. Of course, you may get good results applying transfer learning with these models using data augmentation. But, I'm using training on a computer with 6th generation Intel i3 CPU and 8 GB memory*. So, I had to take into consideration computational complexity and memory limitations. As a result a custom CNN network was used.

# Model Performance And Inference


The model gave an excellent accuracy on both training and validation datasets of around 98%. The moidel is generated and stored in the categorical keras format and the same is used to deploy on the Flask App




## Run Locally

* Clone the project

```bash
  git clone https://github.com/duttadebasmita/Brain-Tumor-Classification-Flask-App-Using-CNN-.git
```

* Go to the project directory

```bash
  cd my-project
```
Paste the datasets from here https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection in a specified location inside my directory*

* For running augmenting the datasets run

```bash
python DataAugmentation.py
```
* For Running Without Augmentaton simply train on the datasets in the following manner 

```bash
python MainTrain.py
Python MainTest.py
```
For Starting the server

```bash
  python app.py
```
*The model is all set to detect the presence of tumorous cells in any BRAIN MRI IMAGE* 


## Reference 

* "Deep Learning-Based Classification for Brain Tumor Detection Using Transfer Learning" by Afshar, P., Plataniotis, K. N., Mohammadi, A.

* "A Comparative Analysis of Brain Tumor Detection Using ResNet, AlexNet, VGG, and InceptionV3 Models" by Akkus, Z., Galimzianova, A., Hoogi, A., Rubin, D. L., Erickson, B. J.

* "A Comparative Study on Brain Tumor Classification Using AlexNet, GoogLeNet, VGGNet, and ResNet" by Jain, A., Goyal, M., Alam, T., Goyal, D.

* "Automated Detection and Classification of Brain Tumors Using Convolutional Neural Networks" by Sivaramakrishnan, R., Sathiyabama, S.

* "Deep Learning-Based Brain Tumor Detection Using Transfer Learning and Data Augmentation" by Ghassemi, N., Gogate, M., Bennani, Y.

## Authors

- @DebasmitaDutta

