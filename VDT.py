#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from VDTlibrary import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import cv2
    
class Classifier():
    
    def __init__(self, training_process = False):
        self.training = training_process
        
    def get_svc(self):
        if self.training:
            print 'Training process activated ...'
            
            # 1) Reading the car and not car images
            cars, notcars = readimages('./big-set/', 'png')
            print 'Image file names loaded'
            
            # 1.1) Show a few samples of the dataset
            print 'Plotting some examples...'
            scar = mpimg.imread(cars[0])
            sncar = mpimg.imread(notcars[0])
            plt.figure()
            plt.subplot(121)
            plt.imshow(scar)
            plt.title('Car')
            plt.subplot(122)
            plt.imshow(sncar)
            plt.title('Not car')
            plt.show()
            
            # 2) Extraction of the features of the images
            print 'Extracting car features...'
            car_features = extract_features(cars, color_space = color_space, spatial_size = spatial_size,
                                           hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell,
                                           cell_per_block = cell_per_block, hog_channel = hog_channel, 
                                           spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)
            print 'Car features extracted'
            print 'Extracting not car features...'
            notcar_features = extract_features(notcars, color_space = color_space, spatial_size = spatial_size,
                                           hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell,
                                           cell_per_block = cell_per_block, hog_channel = hog_channel, 
                                           spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)
            print 'Not car features extracted'
            
            # 3) Normalising the data 
            print 'Normalising ...'
            data, scaler = normalize(car_features, notcar_features)
            
            # 4) Create a labeled vector 'y': 1 - car / 0 - not car
            print 'Creating labels ...'
            y = labeledvector(car_features, notcar_features)
            
            # 5) Creating the training and validation sets
            print 'Spliting and shuffle data set ...'
            x_train, x_val, y_train, y_val = split_shuffle(data, y)
            
            # 6) Create the classifier: SVC
            svc = LinearSVC(multi_class = 'crammer_singer', max_iter = 10000)
            
            # 7) Training the classifier
            print 'Training process begins ...'
            trainning(x_train, y_train, svc)
            print 'Training process is over !'
            
            # 8) Checking the accuracy of the classifier using the validation set
            print 'Calculating accuracy ...'
            accuracy(x_val, y_val, svc)

        else:
            # RUN THIS BLOCK IF YOU WANT TO USE THE PRE-TRAINED CLASSIFIER
            fd = open('SVC-scaler.dat', 'rb') # Open the file with the SVC objects
            clf_dictionary = pickle.load(fd) # Load the objects into a dictionary
            fd.close() # Close the file
            svc = clf_dictionary['svc-big'] # The classifier object
            scaler = clf_dictionary['scaler-big'] # The scaler object
            
        # return the classifier and the scaler
        return svc, scaler
        
class ProcessImage():
    
    def __init__(self, sampling_data, clf, scaler,
                       overlap, color_space, spatial_size, 
                       hist_bins, orient, pix_per_cell, 
                       cell_per_block, hog_channel, spatial_feat, 
                       hist_feat, hog_feat):
        self.sampling_data = sampling_data
        self.clf = clf
        self.scaler = scaler
        self.overlap = overlap
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
    
    #--------------------#
    def draw_heatmap(self, image, heatmap):
        #fig = plt.figure()
        plt.subplot(211)
        plt.imshow(image)
        plt.title('Car Positions')
        plt.subplot(212)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        #fig.tight_layout()
    
    #--------------------#
    def draw_detected_cars(self, image, labels, heatmap):
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)
        #fig = plt.figure()
        plt.subplot(211)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(212)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        #fig.tight_layout()
    
    #-------------------#
    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
        
    #-------------------#
    # Define a function to count the number of cars detected.
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img    
    
    #Function to detected cars#
    def car_zones(self, image, Hp):
        
        """
        Allow to get where car have been detected by using the function provided
        by Udacity SEARCH_WINDOWS.
        Input: - image
               - windows
        Return: - image with drawn boxes
                - detected boxes
        """
        # 1) Load data from the object
        windows_feat = self.sampling_data
        clf = self.clf
        scaler = self.scaler
        overlap = self.overlap
        color_space = self.color_space
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins
        orient = self.orient
        pix_per_cell = self.pix_per_cell
        cell_per_block = self.cell_per_block
        hog_channel = self.hog_channel 
        spatial_feat = self.spatial_feat
        hist_feat = self.hist_feat 
        hog_feat = self.hog_feat
        
        # 2) Create a copy of the image
        cimage = np.copy (image)
        # 3) Create a empty list where to store the detected windows
        detected_windows_list = []
        # 4) Loop into the windows 
        for window_feat in windows_feat:
            # 4.1) Get the window using sliding_window
            windows = slide_window(image, x_start_stop=window_feat[0],
                                          y_start_stop=window_feat[1], 
                                          xy_window=window_feat[2], 
                                          xy_overlap=overlap)
            # 4.2) Get the local probabilities
            hp, hot_windows = search_windows(cimage, windows, clf, scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat) 
            # 4.3) Store the probabilities into the Hp matrix
            Hp = add_probabilities(Hp, hp, windows)
            # 4.4) Store the detected boxes into the list
            detected_windows_list.extend(hot_windows)
            
        # 5) Return the detected windows and the probability matrix (Hp)
        return detected_windows_list, Hp
    
    #Function to process just one frame#
    def image_processing(self, img, Ht):
        # 1) Create two matrix in which we will store:
        Hc = np.zeros_like(img[:,:,0]).astype(np.float) # number of intersections per pixel
        Hp = np.zeros_like(img[:,:,0]).astype(np.float) # probability per pixel

        # 2) Rescale the image
        image = img.astype(np.float32)/255
        # 3) Get the detected boxes and the new probabilities
        detected_boxes, Hp = self.car_zones(image, Hp)
        # 4) Plot
        imgdetected = draw_boxes(image, detected_boxes)
        # 5) Normalize the probabilities --> Spatial probabilities
        Hc = calculate_intersections(Hc, detected_boxes)
        Hs = Hp/np.maximum(1, Hc)
        # 6) Check that probabilities are over a threshold
        # Get the Temporal probabilities --> Ht
        Ht = Hs * 0.5 + Ht * 0.5
        Ht[Ht < 0.4] = 0

        # 7) Draw the probabilities
        plt.subplot(221)
        plt.imshow(imgdetected)
        plt.title('Image')
        plt.subplot(222)
        plt.imshow(Hs)
        plt.title('Spatial probabilities')
        plt.subplot(223)
        plt.imshow(Ht)
        plt.title('Temporal probabilities')
        plt.pause(0.05)
        
        # 8) Return the detected boxes, the matrix of spatial and temporal probabilities 
        return detected_boxes, Hs, Ht
    
    # Function to process a video #    
    def video_processing(self, video):
        
        # 1) Create a figure
        plt.figure()
        # 1.1) Enable the real-time plotting
        plt.ion()
        
        # 2) Get the object of cv2 to capture frame per frame
        cap = cv2.VideoCapture(video)
        
        # 3) Capture first-frame
        ret, frame = cap.read()
        
        # 4) Create a zero matrix of probabilities
        Ht = np.zeros_like(frame[:,:,0]).astype(np.float)
        
        # If you want to go to a specific zone of the video, uncommend this lines. I presupose that we have 25 fps
        #for _ in xrange(25*26):
        #    ret, frame = cap.read()
        
        while ret:
            # 5) Process each image from the video
            detected_boxes, Hs, Ht = self.image_processing(frame, Ht)
            
            # 6) Label Ht to remark the car zone as an unique square
            labels = label(Ht)
            
            # 7) Plot the new values
            plt.subplot(224)
            draw_detected_cars(frame, labels, Ht)
            
            # 8) Capture frame-by-frame
            ret, frame = cap.read()            

# 1) Get the classifier. Put training_process = True if want to train            
clf = Classifier(training_process = False)
svc, scaler = clf.get_svc()
            
# 2) Define features analysis variables
hist_bins = 32 # Defining the number of bins to use in the color histogram analysis
spatial_size = (32, 32) # The size of the image 
orient = 10 # Number of angle divisions to work with in HOG analysis
pix_per_cell = 8 # Number of pixels per cells 
cell_per_block = 2 # Number of cells per block
color_space = 'HSV' # Color space in which we are going to work
hog_channel = 'ALL' # Number of channels of the image to use in order to calculate the HOG
spatial_feat = True # This flag equal to True means that we are going to use the spatial features of the image
hist_feat = True # This flag equal to True means that we are going to use the color histogram features of the image
hog_feat = True # This flag equal to True means that we are going to use the HOG features of the image

# 3) As we want to define different zones into the image, we need to define a 
# matrix in which we will have:
    # 1-2: x start and stop positions
    # 3-4: y start and stop positions
    # 5-6: the windows size
# In this case, we will have 3 different zones, so we will have 3 raws in the
# matrix
sampling_data = [# xstart  xstop  ystart  ystop  wxsize  wysize
                [   [300,  None],   [360,   None],   (128,   128)],
                [   [400,  None],   [360,   600],    (96,    96)],
                [   [500,  None],   [360,   560],    (80,    80)],
                [   [600,  None],   [360,   500],    (64,    64)]]

# 4) Also we will need to define a overlap factor.
overlap = (0.75,0.75) # It means that in each iteration it will occupy the 50% of the 
                      # previous image

# 5) Create the ProcessingImage object to process all the images
prcimg = ProcessImage(sampling_data, svc, scaler,
                       overlap, color_space, spatial_size, 
                       hist_bins, orient, pix_per_cell, 
                       cell_per_block, hog_channel, spatial_feat, 
                       hist_feat, hog_feat)

# 6) Process the video
#video = './test-images/test_video.mp4'
video = './test-images/project_video.mp4'
prcimg.video_processing(video)