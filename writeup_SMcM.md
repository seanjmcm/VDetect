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

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook of the file called `TrafficDetect.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and set the different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and in particular began to focus mainly on the color spaces, (RGB, HLS and HSV).  HSV looked the most promising and this view was reinforced by a paper I discoverd: 
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

In retrospect, instead of using "ALL", processing time may have been saved by using channel 3 only.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In section 2 of `trafficDetect.ipynb` trained a support vector machine (SVM) with radial basis function kernel using the scikit learn svm function as follows:
`svc = svm.SVC(C=1.5, cache_size=250, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)`

I also tried C values of 0.5 and 1.1 but 1.5 seemed to give the highest prediction accuracy.

`Feature vector length: 6108
358.84 Seconds to train svm...
Test Accuracy of svm =  0.9963`

It was at this point that I discovered that the values I was obtaining were just random returns from

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

I decided to return to LinearSVC which appeared to give a better result and is a faster linear SVM.  LinearSVC is an SVM implemented using the liblinear library. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Having an excellent prediction result, I expected to get a very good result from a window search with a 50% overlap.  Instead, I was very disappointed with the result. 

![alt text][image2p]

As can be seen, quite a number of vehicles were missed.

I then implemented a series of scaled sub sampling window searchs in code section five of `trafficDetect.ipynb` as suggested here : https://discussions.udacity.com/t/prediction-excellent-but-actual-result-poor/381167/2 and instead of overlap implemented cell stepping  

Using a range, I added values obtained at scale values of .75, 1, 1.25, 1.5 & 1.75.  Excluding false flags (of which there are a lot), this seems to identify most of the vehicles as shown below

![alt text][image3]

It is now a case of sorting the false detections fromt he real detections.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on seven scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  As you can see in the previous image, there were considerable false flags. In order to mitigate this, I used the add_heat function developed in the module.

```def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
```

This is a marked improvment as can be seen in the figure below, with it's corresponding heatmap.
![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

 ### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7] 



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are a large number of false flags and also quite a large number of vehicles (on the other side of the road) that do not get detected.  The first thing to do would be to increase the size of the training set.  A running weighted moving average of the video would also help to smooth the detection window and remove false detections.

In addition, the sub-sampling scheme could be improved with a better focus on matching scales with specific window areas.

