# Task

Build classifier to detect is there contact data in advertisement or not.

Dataset `train.csv` could be downloaded using script `./data/get_train_data.sh` or by clicking on 
[link](https://drive.google.com/file/d/1LpjC4pNCUH51U_QuEA-I1oY6dYjfb7AL/view?usp=sharing) 

As a model metric averaged `ROC-AUC` on each advertisement category is used.

# Run training

Source code to train model is located in `./lib/train_model.py`.
Baseline model score is 0.91 and must be overcome by a trained model.

For training python==3.8.6 is used.