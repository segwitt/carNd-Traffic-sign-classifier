# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image11]: ./examples/roadwork.jpg "Vis1"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ../data/web_images/1.jpg "Traffic Sign 1"
[image5]: ../data/web_images/2.jpeg "Traffic Sign 2"
[image6]: ../data/web_images/3.jpeg "Traffic Sign 3"
[image7]: ../data/web_images/4.png "Traffic Sign 4"
[image8]: ../data/web_images/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/segwitt/carNd-Traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

One example of image data is

![alt text][image11]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

As we can see that some classes have lots of examples and some have very few examples as compared to others.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because to reduce the number of dimensions which is quite faster for processing the images using CNN , also the color doesnt account for much.

As a last step, I normalized the image data so that the network converges faster.

I decided to generate additional data because more the data, more robust the network to small changes therefore I generated more and more data.

Data after augmentation:
Number of training samples = 54799
number of validation samples = 4410
number of test samples = 12630
image shape = (32,32,3)
number of classes = 43


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 kernel size, 2x2 stride, input = 28x28x6,  outputs 14x14x16 				|
|Convolution 5x5|1x1 stride, valid padding, input = 14x14x16 , output =10x10x16|
| RELU| |
| MaxPooling    | 2x2 kernel, 2x2 stride, input=10x10x16, ouputs=5x5x16      									|
|Flatten|input=5x5x16, output=400|
| Fully connected		| input=400, ouput=120        									|
|Relu| |
| Dropout				| keep probability=0.75        									|
|Fully connected| input=120, output=84|
|	RELU					|												|
|			Dropout			|	keep probability=0.75											|
 |Fully Connected|Input=84, Output=43|
|Softmax| |

model params
Batch Size = 32
epochs = 70
learning rate = 0.001
Mu = 0
sigma = 0.1
Dropout keep probability=0.75

after much testing , I came to the conclusion that a batch size of 32 or 64 was the best choice for the training of the model.
Lower epoch were leading to poor accuracy and much higher ecpochs could lead to overfitting so I chose 70 epochs for training the model.
In many cases I saw that  a learning rate of the order of 1e-3 is the best choice so I used that learning rate, using a higher learning rate prevents the model from converging so this is the best.
A dropout rate of .75 was chosen to prevent the model from overfitting.
Although other optimizers could have been used, I used the adam optimizer which gave the best results.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the LeNet architecture with the provided dataset and got an accuracy
of around 92% on validation set. I plan to change the architeture more and have also realised that to make the network better we need to provide more data for the classes with lesser amount of data.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 93.1 
* test set accuracy of 80

Different optimizers have different uses, but I found out the adam Optimizer to be the best for my use case.
Initially the model was not able to learn better so I added more layers and data augmentation, which led to slight overfitting therefore I had to introduce dropout layers.
Initially the accuracy was around 80% but augmentation and adding more layers shot it upto 90% but with overfitting.
Deciding the number of layers and tuning the hyperpaprameters was a challenge, had to experiment for lower number of epochs continously to get the right set of hyperparameters.
To preprocess the data , I converted it to grayscale because color information is not much useful in this case , also did normalisation so that the model converges faster.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


Perhaps different augementation techniques must be applied to account for classes with lower data and more preprocessing like histogram normalisation etc should be applied.
The pipeline was able to classify 4 out of 5 correctly.
The problem with the dataset picked from net might be different lighting conditions, angles zoomed out zoomed in,, watermarks etc.
I need to implement more complex architecture with more data, better data preprocessing and moreaugmentation

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| general caution      		| 1  									| 
| road work     			| 1 										|
| speed limit(20)					| 0.0025											|
| stop	   		| 1					 				|
| no entry			| 1      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The speed limit sign was unable to be classified properly , this probelem might be solvable by addding more data points and better data augmentation. This compares favourably to the accuracy of test set of 85%, which is obviously less because in case of real world contitions, lighting conditions and other factors can be quit e different.


## Top K values
the top k values of the model are
([18, 26,  0,  1,  2],
       [25, 31, 29, 30, 21],
       [ 1,  0,  3,  8,  4],
       [14,  7,  5,  1,  2],
       [17,  0,  1,  2,  3]],)
Out of which only the third is classified wrongly but else all the 5 values are classified correctly.
The model is quite good but I plan to conduct more experiments by adding more layers and different preprocessing techniques.