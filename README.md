# Gender and Race Classification with Face Images

Let's train a simple face attribute classification model! The dataset we are going to use contains 73 different face attributes, but we are only going to focus on gender and race. Here are some sample results of the trained model. (For now, the dataset I used has only "White," "Black," and "Asian" faces. I will try to add more races in future)

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

### Usage
#### Data Preprocessing

##### Download Dataset
You can download LFWA+ dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). You will see a Dropbox link for LFWA+ dataset in this page. Please unzip the lfw.zip file after you download.
##### Feature Extraction
```bash
python feature_extraction.py --data_dir LFWA/ --save_feature feature.csv --save_label label.csv
```
You can run feature_extraction.py like the above command or just run the jupyter notebook (feature-extracion.ipynb) step by step.

We are going to use the face embedding function _face_encodings_ in [face_recognition](https://github.com/ageitgey/face_recognition/) package, which is based on [dlib](http://dlib.net/)'s _face_recognition_model_v1_. This function takes a face image as input and outputs a 128-dimensional face embedding vector. This function is basically an implementation of [Face-Net](https://arxiv.org/abs/1503.03832), a deep CNN trained with triplet loss. Please refer to the [original paper](https://arxiv.org/abs/1503.03832) for more details

For face detection, I just used [face_recognition](https://github.com/ageitgey/face_recognition/)'s  _face_locations_. This is a histogram of oriented gradient (HOG), which is quite outdated, by default. You can use a CNN face detector by changing input parameters in line 49 of feature_extraction.py and line 20 of pred.py like the follwing
```python
# line 49 of feature_extraction.py
X_faces_loc = face_locations(X_img, model = "cnn")
```
```python
# line 20 of pred.py
locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE, model = "cnn")
````

#### Model Training
```bash
python train.py --feature feature.csv --label label.csv --save_model face_model.pkl
```
You can run train.py as above or just run the jupyter notebook (train.ipynb) step by step.

I used _MLPClassifier_ in [sklearn](http://scikit-learn.org/) package for the classifier. You can change it to the classifier of your flavor but I like _MLPClassifier_ since it is really simple!


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
I love Obama.. Please don't get me wrong!
![real madrid](https://github.com/wondonghyeon/face-classification/blob/master/results/realmadrid-ucl.jpg?raw=true)
![sanders](https://github.com/wondonghyeon/face-classification/blob/master/results/sanders-rally.jpg?raw=true)



### Future Works
- [ ] Add more data (Google Image Search)
- [ ] Use a better face detection model (faster rcnn)
