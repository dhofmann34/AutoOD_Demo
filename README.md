# A Demonstration of AutoOD: A Self-Tuning Anomaly Detection System

## About AutoOD:
AutoOD is a self-tuning anomaly detection system to address the challenges of method selection and hyper-parameter tuning while remaining unsupervised. AutoOD frees users from the tedious manual tuning process often required for anomaly detection by intelligently identifying high likelihood inliers and outliers. AutoOD features a responsive visual interface allowing for seamless user interaction providing the user with insightful knowledge of how AutoOD operates.

AutoOD outperforms the best unsupervised anomaly detection methods, yielding results similar to supervised methods that have access to ground truth labels. 

This work has been accepted for publication at VLDB 2022 (48th International Conference on Very Large Databases) one of the most prestigious conferences in database systems.


### AutoOD Architecture:
![Alt text](https://github.com/dhofmann34/AutoOD_Demo/blob/main/screenshots/architecture.jpg "AutoOD Architecture")


### Input Interface:
![Alt text](https://github.com/dhofmann34/AutoOD_Demo/blob/main/screenshots/input.png "Input Interface")
Users can upload data, provide their own anomaly detection methods, specify the column of labels, and customize the expected percentage range of anomalies in their dataset.

### Data Analytics Display:
![Alt text](https://github.com/dhofmann34/AutoOD_Demo/blob/main/screenshots/results.jpg "Data Analytics Display")
Users can filter the chart based on metrics provided and interact with points by hovering over them to view summery statistics. Clicking on a point will provide that respective point's anomaly score for each unsupervised detector and attribute values from the input dataset. In addition, by moving the slider through each iteration, the user can watch the reliable object set change, and at any time select a point to view the contribution of each detector to its status.

## Instructions:
To run AutoOD, download the code and fill out the "database_temp.ini" file with the information required to connect to a local PostgreSQL database. Once completed rename the file "database.ini". Now running the command "python app.py" in a terminal at the root of the project directory will lunch AutoOD on a local webserver.

Python version: 3.9.12
Please see requirments.txt for libraries and their versions

A hosted web version of AutoOD is coming soon. The link will be provided here.
