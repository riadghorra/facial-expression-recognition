# Facial expression classification using Light CNN

This project was started by a final-year project at CentraleSupélec conducted in collaboration with Illuin Technology and aimed at making automatic and fast classification of facial emotions. To this end a Light CNN model has been implemented from scratch.

This project was then continued in order to make a comparative study of 3 different architectures of models capable of performing an emotions classification task:  the light Convolutional Neural Network, 3 Random Forest algorithm using SIFT descriptors as features and 2 versions of a hybrid SIFT-CNN architecture.
 
## Team
This project was firstly developed in collaboration with Illuin
 Technology in the form of a final-year project at
  CentraleSupélec.
The people who contributed to this project are :
* Corentin Carteau
* Riad Ghorra
* Arthur Lindoulsi
* Lucile Saulnier


## Where to get the data
You must download the FER dataset here : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
If you want to use the FER+ dataset (recommended) you can find the new labels here : https://github.com/microsoft/FERPlus

We created 4 data-sets from FER and the FER+ labels. Follow the instructions to build each of them.

1/ Original FER: ```python3 ferplus/split_fer.py```. That will split FER into train - dev - test by adding a column "attribution".

2/ FER with cropped faces using OpenCV: 
- replace the last line of pipeline.py with ```crop_csv_dataset("./fer_datasets/fer.csv", "./fer_datasets/fer_cropped.csv")```
- Run ```python3 pipeline.py```

3/ FER+: Run ```python3 ./fer_datasets/build_fer_plus.py```

4/ FER+ with cropped faces using OpenCV:
- replace the last line of pipeline.py with ```crop_csv_dataset("./fer_datasets/ferplustensor.csv", "./fer_datasets/ferplus_cropped.csv")```
- Run ```python3 pipeline.py```

5/ FEC data-set. 
- Download the data-set here: https://research.google/tools/datasets/google-facial-expression/
- At the top of the FEC creator script, replace "./fec/train.csv" by the path to the FEC csv for which you want to download the images
- Run ```python3 fec_creator.py```
- You can stop early as the scripts save regularly.



## Config file
The config file can be found at the root of the project and contains 
the parameters you can change in order to train and test the model.
- "path": path to the data set when training the model (see "How to train the model").
- "data_column": To use the original FER or FER+ data-sets, "data_column" must be set to "pixels" and
to use the cropped data-sets, it must be set to "face".
- "BATCH", "LR" & "epochs": batch size, starting learning rate value and number of epochs for training
- "quick_eval_rate": proportion of the validation set to use during training between epochs to quickly measure progress.
For example 0.2 will use 20% of the validation set.
- "path_images": use to change the image folder when testing on custom images (see How to test the model)
- "catslist": name of the categories, in order. Index i corresponds to emotion i in FER and to the i-th coordinate 
in the output of the model
- "current_best_model": During training, use to change where the model is saved.
During testing, use to choose the saved model version you want to test.
- "loss_mode": During training / testing, use BCE if the model is trained on FER+ and CE if the model is trained on FER.

 
## How to train the model
Before training the model, check the config values for 
path, data_column, BATCH, LR, epochs, current_best_model and loss_mode (as specified in "Config file" section)

We recommend using Google Colab or any form of GPU for a faster training. 
To train the model you simply run in Colab cells:

```
python run train.py
``` 

And then

```
main_custom_vgg(start_from_best_model=False, with_data_aug=True)
```

## How to test the model and visualise the results

Before testing, check the config values for data_column, current_best_model and loss_mode 
(as specified in "Config file" section).

### On FER, cropped FER, FER+ or cropped FER+ test set 

- In model_testing.py, replace the last line by ```test_on_fer_test_set(ferplus_cropped_path)``` 
with ferplus_cropped_path replaced by the path of the FER+ cropped data-set.
This is to be sure to use the exact same test set for all models, 
as some faces are removed from FER to FER+ and because of cropping.
- Run ```python3 model_testing.py```
This will run the chosen model for all images in the test set, print the results and plot the confusion matrices.

### On a custom image folder

- Choose a folder path and put it in the config for "path_images".
- In model_testing.py, replace the last line by ```test_on_folder()``` 
- Run ```python3 model_testing.py```
This will run the chosen model for all images in the folder after cropping the faces and save the results in "predictions.csv"
- Run ```python3 img_viewer.py```
- On the UI, click on Open folder and open the folder you chose at the previous step.
- Click on "Open predictions" and open "predictions.csv" generated at the previous step.
- You can now visualise all the images with the corresponding predictions.

### On an annotated csv

Create an annotated csv for an image folder:
- Run ```python3 img_viewer.py```
- Open an image folder
- Click on the corresponding emotion for each image
- You can quit at any time
The annotations are saved in annotations.csv

Evaluate the model on an annotated image folder:
- In model_testing.py, replace the last line by ```test_on_annotated_csv(annotations_csv_path)```
with annotations_csv_path the path to the annotation csv generated before.
- Run ```python3 model_testing.py```
This will run the model on all the annotated images, print the results and plot the confusion matrices.
We used this method to annotate ~600 images from the FEC dataset and evaluate the model on them.
