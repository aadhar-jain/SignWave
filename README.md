
# **SignWave: Real-Time Sign Language Detection Using MediaPipe**

## **Overview**  
SignWave is a real-time sign language detection system designed to interpret hand gestures using a live webcam feed. Leveraging MediaPipe for hand tracking and TensorFlow for model training, this project provides a robust and customizable solution for sign language recognition.  
## Demo Video
[SignWave](https://youtu.be/zgZCIyAArHQ)
## **Requirements**  
Ensure the following dependencies are installed:  
* `mediapipe`  
* `OpenCV`  
* `TensorFlow`  
* `scikit-learn`  
* `matplotlib`  

Install the dependencies using pip:  
```bash
pip install mediapipe opencv-python tensorflow scikit-learn matplotlib
```

## **How to Run**  
To run the application and test sign detection using your webcam, execute the following:  
```bash
python app.py
```  

### **Optional Parameters**  
You may configure the following options when running the app:  
* `--device`: Specify the camera device number (default: `0`).  
* `--width`: Width of the camera capture (default: `960`).  
* `--height`: Height of the camera capture (default: `540`).  

Example:  
```bash
python app.py --device 1 --width 1280 --height 720
```  

---

## **File Structure**  
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

### **Key Files**  
#### **app.py**  
Main script to run the application and detect hand gestures in real-time.  

#### **keypoint_classification.ipynb**  
Notebook used for training the hand sign recognition model.  

#### **model/keypoint_classifier**  
Directory containing:  
* `keypoint.csv`: Training data for hand sign recognition.  
* `keypoint_classifier.hdf5`: Trained model in HDF5 format.  
* `keypoint_classifier.tflite`: Optimized TensorFlow Lite model.  
* `keypoint_classifier_label.csv`: Labels for each hand gesture class.  

#### **utils/cvfpscalc.py**  
Utility script for calculating FPS during real-time detection.  

---

## **Training the Model**  

### **Hand Sign Recognition Training**  
You can modify or extend the existing training dataset and retrain the model for additional hand gestures.  

#### 1. **Collect Training Data**  
* Press `s` to enter the data collection mode.  
* Press keys `0` to `9` to save the key points for the corresponding class ID.  
* Training data will be saved in `model/keypoint_classifier/keypoint.csv`.  


You can add more classes by adding new data or modify/delete existing data in the `keypoint.csv` file.  

#### 2. **Train the Model**  
1. Open `keypoint_classification.ipynb` in Jupyter Notebook.  
2. Update `NUM_CLASSES` to match the number of hand gesture classes.  
3. Modify `keypoint_classifier_label.csv` to include labels for new classes.  
4. Execute the notebook to train the model.  

---

## **Additional Information**  

### **Real-Time Detection**
This project uses MediaPipe's hand-tracking pipeline to extract key points for hand gestures in real-time. The extracted key points are then classified by the trained model to predict the gesture.  
![Screenshot 2025-01-23 015902](https://github.com/user-attachments/assets/f9ba442e-f4b4-4333-a93c-d6dd56e1dd55)

### **Customizing for New Signs**  
The project is designed to be flexible for adding new sign classes. You can update the dataset, retrain the model, and deploy it without changing the core application logic.  
![Screenshot 2025-01-23 015831](https://github.com/user-attachments/assets/b47f5243-7516-4e3f-a605-136befd986f8)
![Screenshot 2025-01-23 015936](https://github.com/user-attachments/assets/a3cfc7c0-94ea-46c6-b7ed-b406b2cc87ff)

### **Performance**  
The current model achieves an average accuracy of **95%** in medium lighting conditions.  

---

## **Future Enhancements**  
* Expand the dataset to include more sign language gestures.  
* Implement multi-language support for broader accessibility.  
* Improve real-time performance using optimized model architectures.  

---

## **Acknowledgments**  
This project uses:  
* [MediaPipe](https://google.github.io/mediapipe/) for efficient hand tracking.  
* TensorFlow Lite for lightweight model inference.  
* OpenCV for video frame processing.  

---
