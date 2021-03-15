# Heart Failure Prediction

In this project we create machine learning model to solve a classification problem using both Hyperdrive and AutoML. We then deploy the best model as a webservice. Below is a diagram demonstrating the steps taken in this project:

<p align="center">
  <img width="600" height="600" src="https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/workflow.png">
</p>

(Image taken from Udacity)

## Dataset

### Overview
We use the Heart Failure Prediction dataset that is publicly available on Kaggle https://www.kaggle.com/andrewmvd/heart-failure-clinical-data. This dataset contains various information on an individual such as sex, diabetes, high blood pressure etc. and if the cause of death was due to a heart failure.

### Task
Our goal is to develop a machine learning algorithm that can detect if a person is likely to die from a heart failure. This helps in diagnosis and early prevention. For this we are going to be using all 12 features in the dataset to develop an accurate model.

### Access
We access the data in automl by importing the dataset locally that was uploaded into the machine learning workspace. On the other hand with hyper drive, we use a URL to access the data directly from Kaggle.

## Automated ML
The settings that were used in automl were as follow:

automl_settings = {    
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'}
    
From the settings above it can be seen that the primary metric chosen to evaluate the performance of the models is Accuracy. The experiment was set to end after 20 mins as AutoML manages to test several models in a short time period. Max concurrent iterations are the maximum number of iterations that are allowed to be executed in parallel.
    
The automl configuration used were as follow:

automl_config = AutoMLConfig(
    task="classification",
    training_data= train_data,
    featurization="auto",
    enable_early_stopping=True,
    label_column_name= "DEATH_EVENT",
    n_cross_validations=4,
    compute_target=cpu_cluster,
    **automl_settings)
    
It can be seen from the configuration that we specifying the target column we are trying to predict which is the "DEATH_EVENT" column.

### Results

We use the RunDetails Widget to get the details of the AutoML experiment. Below are a set of screen shots from RunDetails():

![automl_1](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/automl_1.png)

![automl_2](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/automl_1.png)

![automl_3](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/automl_1.png)

Below is a screenshot of the parameters of the Voting Ensemble model

![best_model](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/best_model.png)

From the AutoML experiment we can see that the best model was a Voting Ensemble model with an accuracy of 86%. What is interesting as seen in the screen shot above, is that the weights of all the features in the model are the same (0.143). This is something that is to be investigated in detail as future work.

## Hyperparameter Tuning

We use a Logistic Regression model as they are well-known to be fast and fairly accurate. The two hyper parameters we choose to tune with their ranges are as follows:

 - "--C": choice(0.1,1,10)
 - "--max_iter": choice(50,100,150)

We use a RandomParameterSampling to randomly select hyperparamter values from the specified range above. This is much better than a grid sweep as it is not as computationally expensive and time-consuming and can choose parameters that achieve high accuracy. Random sampler also supports early termination of low-performance runs, thus saving on computational resources.

We also specifiy a BanditPolicy to terminate runs early if they are not achieving the same performance as the best model. This also adds to improving computational efficiency and saving time as it automatically terminates models with a poor performance.

### Results

We use the RunDetails Widget to get the details of the hyperdrive experiment. Below are a set of screen shots from RunDetails():

![hyperdrive_1](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/hyperdrive_1.png)

![hyperdrive_2](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/hyperdrive_1.png)

![hyperdrive_best](https://github.com/adhamalhossary/heart_failure_prediction_on_azure/blob/main/images/hyperdrive_best.png)

It can be seen above that the best model had parameters of C = 0.1 and max_iter = 50, and achieved an accuracy of 80%.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The best model was the Voting Ensemble model from the AutoML experiment. To deploy the model we did the following:

- Saved the model
- Used score.py in the inference configuration of the deployed model to answer the requests sent to the webservice
- Created a deployment configuration

After the model was deployed we interact with web service by using the REST API as follows:

`scoring_uri = service.scoring_uri # Rest Endpoint
headers = {'Content-Type':'application/json'}

test_data_1 = json.dumps({'data':[{
    'age':75,
    'anaemia':0,
    'creatinine_phosphokinase':582,
    'diabetes':0,
    'ejection_fraction':20,
    'high_blood_pressure':1,
    'platelets':265000,
    'serum_creatinine':1.9,
    'serum_sodium':130,
    'sex':1,
    'smoking':0,
    'time':4}
    ]
        })

test_data_2 = json.dumps({'data':[{
    'age':40,
    'anaemia':0,
    'creatinine_phosphokinase':321,
    'diabetes':0,
    'ejection_fraction':35,
    'high_blood_pressure':0,
    'platelets':265000,
    'serum_creatinine':1,
    'serum_sodium':130,
    'sex':1,
    'smoking':0,
    'time':198}
    ]
        })

response = requests.post(scoring_uri, data=test_data_1, headers=headers)

print("Result 1:",response.text)


response = requests.post(scoring_uri, data=test_data_2, headers=headers)

print("Result 2:",response.text)`

## Screen Recording

Link to video: https://youtu.be/LEsDIIhvDD0
