#Vehicle Detection Project

This is my solution to project 5 of Udacity self-driving car nanodegree. The goals of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction.
* Perform a Color Histogram feature extraction.
*	Perform a Spatial feature extraction.
*	Train a Linear SVM classifier with the features extracted from the images using a labeled training set of images 
*	Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
*	Estimate a bounding box for vehicles detected.

About the code:

All commented code can be found at:
* `Udacity - Self driving course.ipynb`: it contains the main program. 
* `VDTlibrary.ipynb`: it contains the functions used to run the code. Here we implement all the functions seen thought the project.


[//]: # (Image References)
[image1]: ./img-pruebas/car_notcar.png
[image2]: ./img-pruebas/HOG_example.png
[image3]: ./img-pruebas/sliding_windows.png
[image4]: ./img-pruebas/detected-cars.png
[image5]: ./img-pruebas/good.png
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second-fourth code cell of the IPython notebook called `Udacity - Self driving course.ipynb`, specifically in the second step. In this the second code cell we define the parameters to use in the feature extraction. Once we have defined the parameters, we call to the function `extract_features()` that is implemented and fully described in the IPython notebook called `VDTlibrary.ipynb` (also you can use `VDTlibrary.py`, that is the library here used).

I started by reading in all the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images. Here is an example of one of each of the vehicle and non-vehicle classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I found that the best combination was:
* color_space: HSV
* orientations: 10
* pixels_per_cell: 8
* cells_per_block: 2

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color histogram features and the resized image to (32,32,3). This is also done in the fourth code cell of `Udacity - Self driving course.ipynb`, in the seventh step. Previously to train the SVM classifier, we have to:
* Normalising the data
* Create a labeled vector 'y' in which '1' means car and '0' not car
* Creating the training and validation sets
* Create the classifier: SVC

And after train the classifier, I check the accuracy, and normaly I achieve around 99% of accuracy. That is pretty well!

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to prove different combinations of window sizes and start and stop positions, and I found that for this work the best positions and sizes are those compiled in `sampling_data` with a overlaping factor of 75% in x and y axis (`overlap`).

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Moreover, I optimized the performance of the classifier being more strict with the start and stop positions of the window sliding. Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](output.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This explication is totally specified in the Processing images cell of `Udacity - Self driving course.ipynb` file. Basically, I get a probability map from the spatial current state and a temporal probability map, in which I have into acount the previous frames for decide if the car has to be still there or no (`VDTlibrary.ipynb` - function `search_windows` - step 8). Moreover, to make more robust the system I have developed the next steps:
* Get the prediction from the SVM classifier.
* Pass the prediction thought a softmax function to normalize the data between 0 and 1.
* If the probability obtained before is over a threshold (0.85) then
* I flip horizontally the image and pass it again thought the SVM classifier.
* Get the probability by using the softmax function and if it is over a threshold (0.9) then it is consider as a 'true' positive.
* Moreover, I sum up the probability to the probability function, in order to build a probabilistic map and store the window into a list. If the probability obtained thought this process is less than the thresholds, then I sum up 0 to the probability matrix and do not store the window.

After doing this, I normalize the probabilities by dividing the probability matrix by the number of intersections per pixel. And then, I calculate the temporal probability and put to 0 those values of the matrix which are under a threshold (0.4). And finally, I use `scipy.ndimage.measurements.label()` to identify individual blobs in the temporal probability matrix.
Here's an example result showing the probabilistic maps from a series of frames of video, and the car positions.

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced with the false positive detections by using only a heatmap and a threshold. This is a non robust way to do the detection because you can have hot false amplitude that are equal to real ones, so by using a threshold it is impossible to avoid the false positives. 

My pipeline will fail in those zones in which the image has shadows or shapes very similar to a car. This is a consequence of the classifier, that it is not able to distinguis pretty well what is a car and what not. So my improvements would go in this way: use a more efficient and powerfull classifier: a combination of CNN, whose inputs are the resize images, and a NN in which we have as inputs the features of the image.


