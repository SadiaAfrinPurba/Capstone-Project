# Data Scientist Nanodegree
## Capstone Project
Sadia Afrin Purba 
April 16, 2020

## I. Definition

### Project Overview

In the last decade, machine learning ( ML ) has become more popular thanks to very powerful computers that can handle lots of data in a reasonable amount of time. Machine learning concept was introduced by Arthur Samuel in 1959 so it is not new, but today we can use lots of its potential. One more reason for this is that today there is a lot of digitized data that we need to successfully implement good ML models. Dog breed identification problems are well known in the ML community. When I make this model I want to build a web app where users can input an image and obtain prediction from this model. This project will give me an opportunity to build and deploy ML models. One more reason for choosing this problem is because I believe this is a good path to understand how deep learning algorithms work.

### Problem Statement

The goal of this project is to build a pipeline that could be used in a web or mobile app to process real-world images. The pipeline must be able to:

Detect whether an image contains a dog or a human, and distinguish between the two.
- If given an image of a dog, the model predicts the canine’s breed with at least 60% accuracy.
- If given an image of a human, the model identifies the dog breeds the person most resembles.

### Metrics
The evaluation metric for this problem is simply the accuacy score. The accuracy is calculated as follows:
**Accuracy= (Number of items correctly classified) / (All classified items)**

## II. Analysis
]
### Data Exploration
The dataset is provided by Udacity. The dataset has images of dogs and humans. By exploring the dataset, I have deduced that these images are collected from various web sources and then they are preprocessed for further usages. 

The dog images dataset has- 

- 8,351 images
- 6,680 images for training
- 835 images for validation 
- 836 images for test 
- 133 different dog breeds

The human image dataset contains-

- 13,233 images 
- 5,750 different humans 
- image size is 250x250

Both dog and human images dataset are not balanced because the numbers of images provided for each breed or human varies.


### Exploratory Visualization

Example images from dog dataset-
![image](https://user-images.githubusercontent.com/36800937/79497600-c556f600-8049-11ea-8a6a-c5763919e71d.png)

Example images from human dataset-
![image](https://user-images.githubusercontent.com/36800937/79497763-10710900-804a-11ea-8e2c-f8ed51564379.png)

### Algorithms and Techniques
For performing this multiclass classification, we can use Convolutional Neural
 Network to solve the problem. A Convolutional Neural Network (CNN) is a Deep
 Learning algorithm which can take in an input image, assign importance (learnable
weights and biases) to various aspects/objects in the image and be able to
differentiate one from the other. The solution involves three steps. First, to detect
human images, we can use existing algorithm like OpenCV’s implementation of
Haar feature based cascade classifiers. Second, to detect dog-images we will use a
pretrained VGG16 model. Finally, after the image is identified as dog/human, we
can pass this image to an CNN model which will process the image and predict the
breed that matches the best out of 133 breeds.

### Benchmark
The CNN model created from scratch must have accuracy of at least 10%. This can
 confirm that the model is working because a random guess will provide a correct
 answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.


## III. Methodology

All the images are resized to 224*224, then normalization is applied to all images
(train, valid and test datasets). For the training data, Image augmentation is done
to reduce overfitting. The train data images are randomly rotated and random
horizontal flip is applied. Finally, all the images are converted into tensor before
passing into the model.
For building the classifier,the steps are:

Step 1: Import dataset and doing necessary image preprocessing
Step 2: Detect Dogs 
Step 3: Detect Humans
Step 4: Create a CNN model from scratch 
Step 5: Create a CNN model to classify Dog Breed using transfer learning
Step 6: Train the model to classify Dog Breeds 
Step 7: Evaluate the model's performance on the test images.



### Implementation
There are four convolutional layers and three linear layers. 

**CNN Architecture**

- (conv1): Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
  (activation1): ReLU
  (batch_norm1): BatchNorm2d(32)
  (pool1): MaxPool2d(kernel_size=2, stride=2)
  (dropout1): Dropout(p=0.25)
  
- (conv2): Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
  (activation2): ReLU
  (batch_norm2): BatchNorm2d(64)
  (pool2): MaxPool2d(kernel_size=2, stride=2)
  (dropout2): Dropout(p=0.25)
  
- (conv3): Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
  (activation3): ReLU
  (batch_norm3): BatchNorm2d(128)
  (pool3): MaxPool2d(kernel_size=2, stride=2)
  (dropout3): Dropout(p=0.25)
  
- (conv4): Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
  (activation4): ReLU
  (batch_norm4): BatchNorm2d(128)
  (pool4): MaxPool2d(kernel_size=2, stride=2)
  (dropout4): Dropout(p=0.25)
  
**Linear Layers**
- (linear1):  Linear(128 * 14 * 14, 512)
  (activation1): ReLU
  (dropout1): Dropout()

- (linear2):  Linear(512, 256)
  (activation2): ReLU
  (dropout2): Dropout()
  
- (linear3):  Linear(256, 133)

**Explanation**

Four conv layers are consist of kernel_size of 3 with stride 1, and this will not reduce input image. After each conv layer a maxpooling with stride 2 is placed and this will lead to downsize of input image by 2. after final maxpooling with stride 2, the total output image size is downsized by factor of 16 and the depth will be 128. I've applied dropout of 0.25 in order to prevent overfitting. I've applied batch normalization after each conv layer that enables the use of higer learning rates, greatly accelerating the learning process.

Fully-connected layer is placed and then, thrid fully-connected layer is intended to produce final num_classes which predicts classes of breeds.

### Refinement

The CNN created from scratch have accuracy of 16%, Though it meets the
benchmarking, the model can be significantly improved by using transfer learning.
To create CNN with transfer learning, I have selected the Resnet18 architecture
which is pre-trained on ImageNet dataset, the architecture is 18 layers deep. The
last convolutional output of Resnet18 is fed as input to our model. We only need
to add a fully connected layer to produce 133-dimensional output (one for each
dog category). The model performed extremely well when compared to CNN from
scratch. With just 10 epochs, the model got 62% accuracy


## IV. Results

### Model Evaluation and Validation

**Human Face detector:** The human face detector function was created using
OpenCV’s implementation of Haar feature based cascade classifiers. 98% of human
faces were detected in first 100 images of human face dataset and 17% of human
faces detected in first 100 images of dog dataset.

**Dog Face detector:** The dog detector function was created using pre-trained VGG16
model. 100% of dog faces were detected in first 100 images of dog dataset and 1%
of dog faces detected in first 100 images of human dataset.


CNN using transfer learning: The CNN model created using transfer learning with
ResNet18 architecture was trained for 10 epochs, and the final model produced an accuracy of 62% on test data. 


### Justification
I think the model performance is better than expected. The model created using
transfer learning have an accuracy of 62% compared to the CNN model created
from scratch which had only 16% accuracy


## V. Conclusion

### Free-Form Visualization

In Step-3 Haar Cascade Algorithm was used to detect the human face. 
![face_detection](https://user-images.githubusercontent.com/36800937/79497896-48784c00-804a-11ea-90e1-aa4cae25a767.png)

The final outputs of the dog breed classifier using transfer learning are-
![image](https://user-images.githubusercontent.com/36800937/79497985-6c3b9200-804a-11ea-82a5-489886ce32d8.png)


The model is not perfect. Some of the dog images could not be classified by this model. For example:
![image](https://user-images.githubusercontent.com/36800937/79498061-8a08f700-804a-11ea-8ea3-94e9d32622ed.png)


### Reflection
Some other image classification models such as ResNet50 or RestNet101 might perform better than ResNet 18. The model failed to detect dogs category due to an imbalance dataset of dog images. Some breeds have more images where some breeds have only one or two images. 


### Improvement
Three possible points for improvement are:
- A balanced datasets of dogs' images will improve training models.

- More image augmentations such as flipping vertically, move left or right, etc will improve performance on test data

- Ensembles of models
-----------
