# Facial expression classification using Light CNN

## Team
This project was developed in collaboration with Illuin
 Technology in the form of a final-year project at
  CentraleSupélec.
The team members were :
* Corentin Carteau
* Riad Ghorra
* Arthur Lindoulsi


## Where to get the data
You must download the FER dataset here : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
If you want to use the FER+ dataset (recommended) you can find the new labels here : https://github.com/microsoft/FERPlus
or use the function we developed that builds FER+ using
 FER that can be found in ferplus/build_fer_plus.py
 
## Config file
The config file can be found at the root of the project and contains 
the parameters you can change in order to train the model.
In particular the parameters you should change are `path` and `current_best_model` if you want 
to test the model using a particular version. 

 
## How to train the model
We recommend using Google Colab or any form of GPU for a faster
training. To train the model you simply run 
```
python run train.py
``` 
And then
```
main_custom_vgg(start_from_best_model=False, with_data_aug=True)
```
Please note that you must have the dataset at the right path (as defined in the config file) with
the correct `data column` 

## How to test the model
You can test our model by using one og the two functions in model_testing.py : either 
by running `test_on_folder` (don't forget to specify folder path under `path_images` in the config file)
or by running `test_on_anotated_csv` which takes a csv containing images as an input and 
predicts their facial expression. 