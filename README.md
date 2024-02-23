# Contents of Table

* [Introduction](#introduction)
* [YoloV5 Features](#yolov5_features)
* [Class Diagram for Major Classes](#Class_Diagram_for_major_classes)
* [Data Flow](#Data_Flow)
* [Experiments](#Experiments)
  * [Task-Detect Traffic Signs](#Task-detect_traffic_signs)
  * [Data Exploratory Analysis](#Data_Exploratory_Analysis)
  * [Prepare Yolo Format Labels](#Prepare_Yolo_format_labels)
  * [Train Model](#Train_Model)
  * [Train and Val Results Analysis](#Train_and_Val_results_Analysis)
  * [Approaches to Improve](#Approaches_to_Improve)
* [Pros and Cons of Yolov5](#Pros_and_Cons_of_Yolov5)
* [References](#References)

# Introduction

Yolov5 is an anchor-based real-time object detection model.

In the Yolov5 architecture, there is 1 backbone, followed by 3 detection heads.

It can detect objects in 3 size scales.

# YoloV5 Features

1. Dynamic architecture
2. Data augmentation techniques
   1. Mosaic
   2. Copy-Paste
   3. Albumentations
   4. MixUp, Flip, HSV, Translate, Scale, etc.
3. Training stratigies
   1. Multiscale Training
   2. AutoAnchor
   3. Warmup and Cosine LR Scheduler
   4. Exponential Moving Average (EMA)
   5. Mixed Precision Training
   6. Hyperparameter Evolution
4. Computing loss strategies
   1. Weighted loss over Classes Loss (BCE loss), Objectness Loss (BCE Loss), and Location Loss
   2. Balanced objectness loss with [4, 1, 0.4] over small, medium, large size objects
   3. More grids and anchors assigned as ground truth targets to increase positive classes.
   4. Revised formula to predict the box coordinates, to reduce grid sensiticity and prevent predicting unbounded box dimensions.

# Class Diagram for Major Classes

![](http://www.plantuml.com/plantuml/svg/~1UDgDL5rhsp0KVVUlq5ktRlPBzh9G1asRGY65GX_sC8PGvEjKgomPITwMPVtlkvBiE7RaiaAeTUxNkVSUwO9moQmbjIv1seBhkjQgU6Yb4ol6Nq2Lj-20Ew7L8RiW_Y-uTjyv3IVgzWsyUz-17ofjVZ5J1k0LVyaI_dHupLAs6wr-7h99gjCZhLhhwys4BsRoFZq-tisNvCpD65VQEGaA-ClKOQ-bBeuJYZrbSJ1KAE4edJ3lZddV26jtDIZNmoLOpYJGni3HbKlHJFryRi6a-0DqAvW5UpwCG6s5jXDMSoomUUYnnkXA80MjW4gs1tFi8CuyV0jLwJw0DOtSpzIFepaC8cYTG5l979EQbM1iiS7_zpNRWJqSHJNCRa0Sc36Yu7VlqvT-A_krAZTiYypI2yBp3XnMgOUT-IHK-EWHKclB14z0dtkCQocP0rgfS-HJ6DKiuNjLULvmYzWcO8TnTr9RYtAjczR11h-xpMvR8KlgcC4kJjMWKButww_2FM_qJZWRGX8PxpGAhpDQd3LeVwU9CqmeX738oKU1NERv9GaI8qjI1ZWem0S4zAeeYB_GQ0uSepv8PNHDUBKZsX1KUwW3vJXgCJhwsplDYdTs14LUsVX6agBuqbDecB7GFM4fhqKA3PNJEJs1gSMWgIDoimYEmmEI9GH9D38_tpkovGQyQ0Pzu68JF58vbt5rXMbLk2BfxVcP3T773NPH9oeBTOuQGKTH4L1iAu5K7gLUHAE0FCU-fyh3GBtzWHKKM1jYFiQ78SOF1SBKfj7OnfHIgi99mZVwBaQvbiJxduytb_bVhD8_evXmuZ3IVnXvb7zYJV6sEMtuCutVmRP-kFu_xpKpSl_aTDC8jSjYBJ_bapwcYQxoV6JEGYPIV7iMPGdgEzQ9p_K_x9ogum00)

**train** is the entry point.

**ComputeLoss** measures the difference between the ground truth and predictions. The final loss is a weighted combination of Classes Loss (BCE Loss), Objectness Loss (BCE Loss), and Location Loss (CIoU Loss)

```
build_targets() is an important method to assign GT boxes to the right feature map grid and anchors.
```

**DetectionModel** predicts the prediction results. It initiates the Yolov5 model by using .cfg  or loading the pre-trained weights. Some classes are the building blocks of the Yolov5 model, like Conv, C3, SPPF, etc.

**LoadImagesAndLabels** and **InifiniteDataLoader** access and wrap the input images and labels for training.

# Data Flow

Assume we have

B: the number of anchor boxes = 3

C: class = 80

Example model: Yolov5s
![](http://www.plantuml.com/plantuml/svg/~1UDfjKR5Emp0GtVqLjv4YAgKfgIe61c0e4vKO683XaXnHLESSsOu4Vy_5XkAK3ddAlNT-xpcvS8UE1xJOmntMXzQQaPjxeuq5Rv6TICHU_MtLfXyC2_VilpI1fTeZjvgKEYgmjpHOKvDp7RS9FoLKyffGEy8c6H_8Ys52F8r-65FYaUVptZYVXAcWmz8kR-Sru0PsS0alygpLN0dBlzraRoqxc-iyeZeAOMtg2ofWc6DXERGgjd9WcqV8DDgM5cyCHRaHAhW9P3qr7V8abAj2k2AFaBaUjLwe7x-f_UnYab2KvRc4IFyFYFof_uchvCGlBFwlLMK-h-G7yuUWHkVUlm5UBgoO)

*Note that the 255 = B * (bbox + obj + cls) = 3 * (4 + 1 + 80)*

**Input** are images, with height * width * channels in dimensions. And they have ground truth bounding boxes in training.

**Preprocessing** includes

```
check_img_size()
letter_box()
dataloader()
```

**Backbone** extracts features at different spatial resolutions.

**Head** further refines the feature and prepares them for prediction.

**Detect** produces the final predictions, including bounding box coordinates, objectness scores, and class probabilities.

**Postprocessing** includes NMS in the detection phase.

NMS includes actions such as

1. filtering detections by confidence
2. calculating conf = obj_conf * cls_conf
3. sorting conf in descending order,
4. outputting the result predictions after filtering overlapping iou, it has an output number of the limit with max_nms = 300

# Experiments

**Task-Detect Traffic Signs**

Train a yolov5 model to detect 4 traffic signs: speed limit, crosswalk, traffic light, stop

**Data Exploratory Analysis**

877 road sign images, 4 class

<p align="center">
 <img src="https://github.com/GuilinXie/yolov5_project_report/blob/main/result_img/labels.jpg" width="500" height="500">
</p>

The top-left image shows the number of instances for each class. The most frequent class is speedlimit, with the number of about 570,  while the stop sign is the least frequent, with only around 90 instances. This indicates that, firstly, the classes are imbalanced. Secondly, we may need to collect more data. Some best practice recommends training over 1500 images per class, and more than 10, 000 instances per class, with 10% background images, to achieve a robust Yolov5 detection model and to reduce FP errors. But at this point, I will just work on this 877-tiny image dataset to show the Yolov5 workflow.

The top-right image shows that most of the bounding boxes are relatively small, so we need to detect smaller objects than larger ones. The bottom-right image confirms this, as most bounding boxes' height and width ratios are near (0, 0)

The bottom-left image shows that the instances are mostly located in the middle of the images.

**Prepare Yolo Format Labels**

The original labels are (x1, y1, x2, y2), which are the top-left and bottom-right positions in images, we convert these to (x0, y0, w, h) for yolo format, which are the bounding box's center position and the w, h ratio to the image size.

**Train Model**

Firstly, we configure:

```
yolov5/data/data.yaml 
yolov5/data/hyps/hyp.yaml
yolov5/models/yolov5s.yaml
```

1. *data.yaml* is for train/val/test images path, and classes' names
2. *hyp.yaml* is for the hyperparameters of the model, like lrf, mosaic, fl_gamma etc.
3. *yolov5s.yaml* is the .cfg file for the model, we can specify prior anchor boxes, nc(number of classes) in it.

Then, we train by using

```
python train.py --epochs=50 --weights yolov5s.pt --data yolov5/data/data.yaml  -- hyp yolov5/data/hyps/hyp.yaml --cfg yolov5/models/yolov5s.yaml --batch-size 32 --imgsz 640
```

**Train & Val Result Analysis**

train & val loss

![1708546857160](https://github.com/GuilinXie/yolov5_project_report/blob/main/result_img/results.png)

The loss results show that training loss and val loss are decreasing and precision, recall, and mAP metrics are increasing as expected.

The cls_clss and mAP_0.5 are converging to the flat point. But the box_loss, obj_loss, and mAP_0.5:0.95, and recall are still slightly decreasing or increasing over the number of epochs. This means that if we train more epochs, these metrics may get better.

**Precision-Recall Curve**

<p align="center">
 <img src="https://github.com/GuilinXie/yolov5_project_report/blob/main/result_img/PR_curve.png" width="550" height="500">
</p>
This figure shows that, for all classes, in the area near the point (Recall = 0.8, Precision = 0.93), the  model performs well on both Precision and Recall.

The model does not predict trafficlight as well as other classes.

**Confusion Matrix**

<p align="center">
 <img src="https://github.com/GuilinXie/yolov5_project_report/blob/main/result_img/confusion_matrix.png" width="550" height="500">
</p>

The confusion matrix shows that the model makes wrong predictions between (crosswalk, trafficlight, speedlimit) and the background.

So we can consider adding some background images during training.

**Val Predictions**

<p align="center">
 <img src="https://github.com/GuilinXie/yolov5_project_report/blob/main/result_img/val_batch0_pred_custom.jpg" width="500" height="500">
</p>

In this val result image, we can see that all the predictions are correct, but there are 4 trafficlight signs with low confidence scores like 0.3 and 0.4, as marked in red circles.

If we want to increase the correct predictions with a higher confidence score, we can try to modify the objectness loss function. In the source code of Yolov5, it calculates objectness loss using iou, we can change it to predict 1 instead.


<p align="center">
 <img src="https://github.com/GuilinXie/yolov5_project_report/blob/main/result_img/val_batch0_pred_custom.jpg" width="500" height="180">
</p>

These val results have a False Positive (FP) prediction as circled in red. It predicts a partial hidden sign as speedlimit with a very high confidence=0.7.

This happens mainly because we do mosaic data augmentation during training. Mosaic will crop training images randomly. This can lead to cut the GT boxes as well. Some instances are left with small parts. Even though it has a strategy to filter those partially left in the *box_candidates()* function. It is not good enough to avoid the FP in this case. 

The target object is already small, and cutting out most of it and still marking it as positive does not make sense. So we need to customize the *load_mosaic()* function to fit the small targets detection scenario as in this case.

To improve this, we need to mask out the partially left objects if they are too small as compared to a certain threshold. We need to mask out the objects in the image because we do not want to leave any partially left objects without labels.

**Approaches to Improve**

According to the previous result analysis, I figured out that we could do the following to improve the model's performance:

1. **From the dataset's view**

* Collect more traffic sign data, to match the best practice of 1500 images per class, 10, 000 instances per class
* Collect more data for traffic light
* Add 10% background images to reduce FPs

2. **From the training's view**

* Continue training for like 20 more epochs

3. **From data augmentation's view**:

* Customized mosaic augmentation to avoid small objects' FPs

4. **From model architecture's view**

* Add a 4th detection head in 160 * 160 feature map level, so that it can better detect small objects
* Add more anchor boxes in the small-scale feature map to detect small objects
* Try larger models like Yolov5l
* Trye newer models like Yolov8

# Reference:

[1] https://github.com/ultralytics/yolov5/issues/11299

[2] https://docs.ultralytics.com/yolov5/tutorials/architecture_description/#44-build-targets

[3] https://www.kaggle.com/datasets/andrewmvd/road-sign-detection

[4] https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843#:~:text=To%20achieve%20a%20robust%20YOLOv5,to%20reduce%20false%2Dpositives%20errors.

[5] https://github.com/ultralytics/yolov5/issues/6549

[6] Best practice for Yolov5: https://github.com/ultralytics/yolov5/issues/7794

[7] https://docs.ultralytics.com/yolov5/tutorials/architecture_description/
