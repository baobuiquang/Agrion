<br>
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="assets/logo-black.png" alt="Logo" height="100">
  </a>
  <p align="center">
    An awesome README template to jumpstart your projects!
    <br>
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br>
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents
- [Hackathon Submissions](#)
- [About AGRION Project](#)
- [AGRION VISION - The Core AI Function](#)
  - [How AI was applied](#)
  - [Approach](#)
  - [Demo on mangoes](#)
  - [Test this demo](#)


## Hackathon Submissions
Content | Link
------- | -------------
Github  | [Link to Github](https://github.com/AgrionTeam/Agrion) (You're at here)
Video   | [Link to video on Vimeo](https://github.com/AgrionTeam/Agrion)
Slide   | [Link to slide](https://github.com/AgrionTeam/Agrion)
App Prototype | [Link to Mobile App Prototype on Figma](https://github.com/AgrionTeam/Agrion)
Business Model Canvas | [Link to AGRION's BMC](https://github.com/AgrionTeam/Agrion)


## About AGRION Project
blah blah


## AGRION VISION - The Core AI Function

### How AI was applied
The core AI function - Agrion Vision - is an application of **Image Classification** in **Computer Vision** technology.

We use a technique called **Transfer Learning**, namely **MobileNets** - a pre-trained Convolutional Neural Networks - to train the machine learning model to recognize the agricultural products are in good or bad condition and predict how long they remain usable from photos that users uploaded.

The model we train is **Tensorflow** model that can be exported into 2 types: **Tensorflow Lite** to implement on mobile apps (in collaboration with Flutter) and **Tensorflow.js** to implement on the web.

In the Agrion Vision demo for this hackathon, we have gathered and trained the model using a dataset including approximately 1470 images of mangoes with “in-good-condition” label and 1380 images of mangoes with “in-bad-condition” label. The accuracy of this model is pretty high, even in different backgrounds, it still recognizes which mango is in good or bad condition.


### Approach

#### Machine Learning
Machine Learning is an application of AI that provides computers the ability to learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of programs that can access data and use it to learn for themselves without human intervention or assistance and adjust actions.

The process of learning begins with observations of data in order to look for patterns, make better decisions based on the data that we provide. Once we have trained a machine learning model, we can use it to reason over data that it hasn't seen before, and make predictions about those data.

#### TensorFlow
TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in Machine learning and developers easily build and deploy ML-powered applications.

#### Teachable Machine
Teachable Machine is a tool that makes creating machine learning models fast and easy. The models trained with Teachable Machine are real TensorFlow.js models.

Teachable Machine is using a technique called [**Transfer Learning**](https://www.tensorflow.org/tutorials/images/transfer_learning). There is a neural network that was previously trained on a large dataset, typically on a large-scale image-classification task. This model is called MobileNet and the training data for this model is called ImageNet.

There are papers about MobileNets and ImageNets.
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
- [ImageNet: A Large-Scale Hierarchical Image Database](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

MobileNet doesn't necessarily know whether a mango is in good condition or not, but it's learned how to boil the essence of images down into a bunch of different numbers, to then retrain it with our own images. If we were training an image classification model from scratch without this base model, we probably would need a much larger data set.


### Demo on mangoes
#### Recognize a mango is in good or bad condition
*(If you don't see any gif here, please reload the page)*
![png](assets/demo.gif)
*Realtime Screen Recording*

### Test this demo
1. Clone this repository
    ```
    git clone https://github.com/AgrionTeam/Agrion
    ```
2. Run `index.html` on local host
3. Allow the web to access camera
4. Test with mangoes 🥭

#### Why demo on web?
The model we have trained in this hackathon is [Tensorflow.js](https://www.tensorflow.org/js) model that work anywhere javascript runs, plays nice with [P5.js](https://p5js.org/), [ML5.js](https://ml5js.org/) & more, so it's easier to test and have a demo quickly.