# Amharic Speech-to-Text engine
## Introduction
<p> Speech recognition technology allows for hands-free control of smartphones, speakers, and even vehicles in a wide variety of languages. Companies have moved towards the goal of enabling machines to understand and respond to more and more of our verbalized commands. There are many matured speech recognition systems available, such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice assistants work for limited languages only. </p>

<p>The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database. </p>

<p>Our responsibility was to build a deep learning model that is capable of transcribing a speech to text in the Amharic language. The model we produce will be accurate and is robust against background noise.</p>

## Code
The code of our analysis can be found in the **notebooks** folder. The data preprocessing and visualization, and model training parts can be found in the **Amharic_STT_preprocessing.ipynb** jupyter notebook. This notebook can be run in google colab. The **Amharic_Speech_To_Text.ipynb** contains a modularized version of the first notebook. The **scripts** folder contains the data loading and preprocessing functions. The trained models will be stored in the **models** folder.

## Models
<p>We attempted two types of models. In our first attempt, we first performed the preprocessing and augmentation. We then performed feature extraction by plotting mel-spectrograms, and mfcc-spectrograms. We saved these plots as images, and we later loaded and used them to train our model. When we applied our first approach using only 100 audios, we obtained good results, but when we used 2000 audio files, it consumed too much memory, and the accuracy was bad. 
</p>
<p>
In our second attempt, we kept the same architecture as the first model, but we added one more layer, called the log melgram layer. This layer takes the sampled audio as input and computes the spectrogram for each audio on the fly. Not only did adding this layer decrease the memory consumption, it also boosted the performance of our model. 
</p>
<p>
The best model we have trained so far is saved as **new_model_v1_8500.h5**. You can load and test this model by running the **test_model.py** file from the **scripts** directory.
</p>

## Dependencies
To install the necessary dependencies, execute the command 
```$ pip install -r requirements.txt"```
