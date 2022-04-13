# Detecting Parkinson’s Disease

Parkinson's disease is a long-term degenerative disorder of the central nervous system that mainly affects the motor system. Its effects on the central nervous system are both chronic (meaning they persist) and progressive (meaning the symptoms grow worse over time), so a model is created to detect the presence of the disease for individuals to seek treatment from physicians.

## Project Set-Up and Installation
Microsoft's Azure ML is used to solve the underlying problem. In this experiment, models were built using the python libraries, sci-kit-learn, NumPy, pandas, and xgboost, with environment details included in the myenv.yml file.
## 
![workflow](https://user-images.githubusercontent.com/48255327/162997271-9cf8c840-0e99-4380-bbf1-8bf88fd03477.png)

## Dataset

### Overview
This dataset is composed of a range of biomedical voice measurements from 
31 people, 23 with Parkinson's disease (PD). Each column in the table is a 
particular voice measure, and each row corresponds to one of 195 voice 
recordings from these individuals ("name" column). The main aim of the data 
is to discriminate healthy people from those with PD, according to "status" 
column which is set to 0 for healthy and 1 for PD.

The data is in ASCII CSV format. The rows of the CSV file contain an 
instance corresponding to one voice recording. There are around six 
recordings per patient, the name of the patient is identified in the first 
column. For further information or to pass on comments, please contact Max 
Little (littlem '@' robots.ox.ac.uk).

Further details are contained in the following reference -- if you use this 
dataset, please cite:
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 
'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', 
IEEE Transactions on Biomedical Engineering (to appear).

Citation Request:

If you use this dataset, please cite the following paper: 
'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', 
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 
BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

### Task
The project involves creating a model to predict the presence of Parkinson's Disease in an individual. The 'status' column is set as a target column and 
other parameters are set to create a model to solve the problem.

### Access
The data is uploaded to the workspace for AutoML and accessed from this [url](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/) for the hyperdrive process.

![Screenshot (105)](https://user-images.githubusercontent.com/48255327/163001751-bd650630-f21c-4f9e-bc9e-1c730eb41a95.png)


## Automated ML
An AutoML run was initiated with early stopping enabled, timeout minutes of fifteen, the primary metric of accuracy and featurization set to auto. The Parkinson's dataset was then selected with the label column 'status' chosen, and lastly, the task was set to classification.

### Results
After the experiment time had elapsed, a Voting Ensemble model registered the best accuracy of 95.4%. The column 'PPE' topped as the most important feature,
with 'spread1' and 'spread2' being the second and third respectively.

##
![feature_importance](https://user-images.githubusercontent.com/48255327/163023319-faddb933-3e19-4c8c-8baa-5f15d17b3623.png)

## 
Screenshots from the AutoML run:
![automl_run-details2](https://user-images.githubusercontent.com/48255327/163006808-15028d43-1523-4155-88e5-791dda8ac771.png)
![automl_model-run_id](https://user-images.githubusercontent.com/48255327/163006818-bd9d960a-59b7-4c35-b4bd-7f7f55ee597a.png)
![automl_run-details](https://user-images.githubusercontent.com/48255327/163006823-59827579-d8a8-4d18-9a0f-cf8f420e23fc.png)
##
Screeshot of Confusion Matrix:
![confusion_matrix](https://user-images.githubusercontent.com/48255327/163023130-c3e799e3-42b6-4318-8393-9c62fb0e4321.png)


## Hyperparameter Tuning
Logistic Regression was chosen because it is easier to implement, interpret, and very efficient to train on such a dataset. A random parameter search is then 
selected to reduce the time for computation. The run finished in approximately 8 minutes. The hyperparameters 'C' and 'max_iterations' were randomly 
sampled from '0.3, 0.6, 0.9, 1.2' and '40, 90, 140, 190, 240' choices respectively.

### Results
The best model was registered with an accuracy of 89.1% from hyperparameters, regularization Strength of choice of 0.3 and max iterations of 40. 
An improvement can be made by sampling randomly through the uniform distribution of a specific range. 

##
Screenshots from Hyperdrive run:
![hyper-model_param (2)](https://user-images.githubusercontent.com/48255327/163007179-72bbf73f-3122-4c00-ae77-0874e448e784.png)
![hyper_run-details](https://user-images.githubusercontent.com/48255327/163007188-11f82bbd-fa16-4727-8962-8fdb73c32baf.png)
![hyperdrive_3d](https://user-images.githubusercontent.com/48255327/163007192-3b2411e9-865d-45e1-b4fa-28b56cb08920.png)

##
Screenshot of the best model with its run id:
![hyper_run-id](https://user-images.githubusercontent.com/48255327/163024491-0b4138dc-185f-4110-96d1-933991c8b9e2.png)


## Model Deployment
The AutoMl model is selected for deployment after comparisons show it proved the better. The deployment is carried out with the help of the ACI
(Azure Container Instance). A request is then made to the endpoint, after the endpoint is deleted.

##
Screenshots of model deployment and Testing Endpoint:
![both_models](https://user-images.githubusercontent.com/48255327/163013586-88daf814-6a7c-463b-91fb-eb2354ca17ad.png)
![service-created](https://user-images.githubusercontent.com/48255327/163013900-6eda2b3c-f867-46ad-9824-8aebf53be17a.png)
![service_created](https://user-images.githubusercontent.com/48255327/163014003-985c5c74-91f2-4f04-a126-0257d12062e2.png)
![service_active](https://user-images.githubusercontent.com/48255327/163014185-62eac386-236a-4367-a8ec-26c85f72e56e.png)
![test_endpoints (2)](https://user-images.githubusercontent.com/48255327/163177858-24bed140-6deb-4a47-907b-30daee956db4.png)
![service-delete](https://user-images.githubusercontent.com/48255327/163014392-b97fdc06-f5b3-4187-872c-c9319dd12f8d.png)

## Screen Recording
[Link to Screencast](https://www.youtube.com/watch?v=2YEfKqpe0A4). Script can be found [here](https://github.com/KwabenaNyinaku/nd00333-capstone/blob/master/starter_file/script.md).
