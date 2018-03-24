# Gender and Race Classification with Face Images

Let's train a simple face attribute classification model! The dataset we are going to use contains 73 different face attributes, but we are only going to focus on gender and race. Here are some sample results of the trained model. (For now, the dataset I used have only "White," "Black," "Asian" faces. I will try to add more races in future.

![Ellen's Oscar Selfie](https://github.com/wondonghyeon/face-classification/blob/master/results/ellen-selfie.jpg?raw=true)
![Trump Rally](https://github.com/wondonghyeon/face-classification/blob/master/results/trump-rally.jpg?raw=true)
![Korean Protest](https://github.com/wondonghyeon/face-classification/blob/master/results/korean-protest.jpg?raw=true)


### Requirements
#### Packages
[dlib](http://dlib.net/)
[face_recognition](https://github.com/ageitgey/face_recognition/)
[NumPy](http://www.numpy.org/)
[pandas](https://pandas.pydata.org/)
[scikit-learn](http://scikit-learn.org/)

#### Dataset
[LFWA+ dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### Training
#### Data Preprocessing

##### Download Dataset
You can download LFWA+ dataset [here]((http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). You will see a Dropbox link for LFWA+ dataset in this page. Please unzip the lfw.zip file after you download.
##### Feature Extraction
You can run feature_extraction.py or just run the jupyter notebook (feature-extracion.ipynb) step by step.
```bash
python feature_extraction.py --data_dir LFWA/ --save_feature feature.csv --save_label label.csv
```
#### Model Training
You can run train.py or just run the jupyter notebook (train.ipynb) step by step.
```bash
python train.py --feature feature.csv --label label.csv --save_model face_model.pkl
```

#### Prediction
Classify face attributes for all images in a directory using pred.py
```bash
python pred.py --img_dir test/ --model face_model.pkl
```
### More Examples!
![blm](https://github.com/wondonghyeon/face-classification/blob/master/results/blm.jpg?raw=true)
![women's march](https://github.com/wondonghyeon/face-classification/blob/master/results/womens-march.jpg?raw=true)
![g20](https://github.com/wondonghyeon/face-classification/blob/master/results/g20-2016.jpg?raw=true)
![grammy](https://github.com/wondonghyeon/face-classification/blob/master/results/grammy-selfie.jpg?raw=true)
![volleyball](https://github.com/wondonghyeon/face-classification/blob/master/results/korean-womens-volleyball.jpeg?raw=true)
![baskettball](https://github.com/wondonghyeon/face-classification/blob/master/results/london-olympics-basketball-men.jpg?raw=true)
![obama](https://github.com/wondonghyeon/face-classification/blob/master/results/obama-selfie.png?raw=true)
![real madrid](https://github.com/wondonghyeon/face-classification/blob/master/results/realmadrid-ucl.jpg?raw=true)
![sanders](https://github.com/wondonghyeon/face-classification/blob/master/results/sanders-rally.jpg?raw=true)



### Future Works
- [ ] Add more data (Google Image Search)
- [ ] Use a better face detection model
