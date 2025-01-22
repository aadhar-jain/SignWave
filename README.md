# Signwave: sign language detection using mediapipe

# Requirements :-
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later
* scikit-learn
* matplotlib 
# How to run :-
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options might need to be updated based on your system :-
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)


# File structure :-
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│    │  keypoint.csv
│    │  keypoint_classifier.hdf5
│    │  keypoint_classifier.py
│    │  keypoint_classifier.tflite
│    └─ keypoint_classifier_label.csv        
│          
└─utils
    └─cvfpscalc.py
</pre>

### keypoint_classification.ipynb
This is the model training script for hand sign recognition.

### model/keypoint_classifier
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)

### utils/cvfpscalc.py
For FPS measurement.

# Training
You can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
Press "s" to enter the mode to save key points
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv".

In the initial state, 2 types of learning data are included: open hand (class ID: 0) and close hand (class ID: 1).

If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.<br>

#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv".