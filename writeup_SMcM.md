##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Report_Images/ImagestSample.png
[image2]: ./Report_Images/ImageAnalysis.png
[image2p]: ./Report_Images/sub-samplng.png
[image3]: ./Report_Images/sub-sampling2.png
[image4]: ./Report_Images/pipeline.png
[image5]: ./Report_Images/heats.png
[image6]: ./Report_Images/IntegratedHeatmap.png
[image7]: ./Report_Images/LabeledIntHeatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how extracted HOG features were extracted from the training images.

The code for this step is contained in the first code cell of the jupyter notebook file called `TrafficDetect.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images from the .  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored different color spaces and set the different `skimage.hog()` parameters such as `orientations`, `cells_per_block`, and `pixels_per_cell`).  I then selected a vehicle image and a non vehicle image and ran the `skimage.feature` hog function on them

The following image illustrates the use of the `HSV` color space, in four instances: converted to grayscale and separated into its three constituent channels:  

![alt text][image2]

The HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and `visualise=True` were used.  The visualization image appears to show that the 'S' channnel (channel 2) offers the best potential to effectively and efficiently extract HOG features from a vehicle.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and in particular focussed on the color spaces, (RGB, HLS and HSV).  HSV looked the most promising and this view was reinforced by a paper I discoverd: 
Goto, Yuhi & Yamauchi, et al. (2013). *CS-HOG: Color similarity-based HOG* FCV 2013 - Proceedings of the 19th Korea-Japan Joint Workshop on Frontiers of Computer Vision. 266-271. 

I settled on the following parameters

|Parameter      | Value     |
|:-------------:|:---------:| 
|color_space    | 'HSV'     |
|orient         | 9         | 
|pix_per_cell   | 8         |
|cell_per_block |   2       | 
|hog_channel    | "ALL"     |
|spatial_size   | (16, 16)  |
|hist_bins      | 16        |    
|spatial_feat   | True      |
|hist_feat      | True      |
|hog_feat       | True      |

In retrospect, instead of using "ALL", processing time may have been saved by using channel 2 only.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In section 2 of `trafficDetect.ipynb` I trained a support vector machine (SVM) with radial basis function (rbf) kernel using the scikit learn svm function on the selected HOG features as follows:
`svc = svm.SVC(C=1.5, cache_size=250, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)`

I also tried C values of 0.5 and 1.1 but 1.5 seemed to give the highest prediction accuracy.

`Feature vector length: 6108
358.84 Seconds to train svm...
Test Accuracy of svm =  0.9963`

It was at this point that I discovered that the values I was obtaining were just random returns from using

`svc = LinearSVC()`

I had not commented out the above code line.  All my careful adjustments were meaningless.  I decided to use GridSearchCV to find the best algorithm.

`from sklearn.model_selection import GridSearchCV`

`parameters = {'kernel':('linear', 'rbf'), 'C':[.5, 1, 1.5]}`

`svc1 = svm.SVC()`

`svc = GridSearchCV(svc1, parameters)`

`svc.fit(X_train, y_train)`

`print(svc.best_params_)`

This revealed that C=0.5 and linear were the best parameters for the dataset.  Implementing these parameters yielded the following:

`Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
156.22 Seconds to train svm...
Test Accuracy of svm =  0.9918`

I decided to return to LinearSVC which appeared to give a better result and is a much faster linear SVM.  LinearSVC is an SVM implemented using the liblinear library. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Having an excellent prediction result, I expected to get a very good result from a window search with a 50% overlap.  Instead, I was very disappointed with the result. 

![alt text][image2p]

As can be seen, quite a number of vehicles were missed.

I then implemented a series of scaled sub sampling window searchs in code section five of `trafficDetect.ipynb` as suggested here : https://discussions.udacity.com/t/prediction-excellent-but-actual-result-poor/381167/2 and instead of overlap implemented cell stepping  

The following scale values of .75, 1, 1.25, 1.5 & 1.75 were used with a static window size of fulling horizontal width stating a height from the origin of 450 pixels and finishing at 700pixels from the origin .  Excluding false flags (of which there are a lot), this seems to identify most of the vehicles as shown below

![alt text][image3]

It is now a case of sorting the false detections fromt he real detections.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on seven scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  

The seven scales were as follows:

* .75
* 1.0
* 1.25
* 1.5
* 1.75
* 2.0
* 2.25

In order to optimize the classifier, I zoomed in on particular sections of the image window relating to the seven scales.

Including all the scales increased dramatically the likelihood of a vehicle discovery.

As you can see in the previous image, there were considerable false flags. In order to mitigate this, I used the add_heat function developed in the module.

```def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
```
I set a threshold of 16 for the heatmap, `heatt = apply_threshold(heat,16)` , meaning that only pixels that were identified more than 16 times were retained. 

This resulted in a marked improvment of vehicle recongition as can be seen in the figure below, with it's corresponding heatmap.
![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The positions of vehicle detections were recorded in each frame of the video.  A heatmap of the positive detections was generated.  As discussed in the previous section a threshold of 16 was set in order to identify vehicle positions and exclude false flags.  It was assumed that real vehicles would appear more than false flags.  Following this,  `scipy.ndimage.measurements.label()` was used to identify individual areas of interest in the heatmap and then each area was presumed to correspond to a vehicle.  Using the function `draw_labeled_bboxes`, boxes were drawn to cover the areas in which a vehicle was assumed to be present.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]

 ### Integrated heatmap from all six previous frames

 Although not added to the video generation section.  The integrated heatmap of the previous 6 images with a 
![alt text][image6]

### Resulting bounding boxes drawn onto the last frame in the series

The resulting integrated heatmap was passed to the function `draw_labeled_bboxes` and appears on the last frame as shown below.
![alt text][image7] 



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are a large number of false flags and also quite a large number of vehicles (on the other side of the road) that do not get detected.  The first thing to do would be to increase the size of the training set.  A running weighted moving average of the video would also help to smooth the detection window and remove false detections.

In addition, the sub-sampling scheme could be improved with a better focus on matching scales with specific window areas.

