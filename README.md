# Sales Data Analysis
## Introduction
<p> Speech recognition technology allows for hands-free control of smartphones, speakers, and even vehicles in a wide variety of languages. Companies have moved towards the goal of enabling machines to understand and respond to more and more of our verbalized commands. There are many matured speech recognition systems available, such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice assistants work for limited languages only. </p>

<p>The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database. </p>

<p>Our responsibility was to build a deep learning model that is capable of transcribing a speech to text in the Amharic language. The model we produce will be accurate and is robust against background noise.</p>

## Code
The code of our analysis can be found in the **notebooks** folder. The data preprocessing and visualization, and model training parts can be found in the **Amharic_STT_preprocessing.ipynb** jupyter notebook. This notebook can be run in google colab. The **Amharic_Speech_To_Text.ipynb** contains a modularized version of the first notebook. The **scripts** folder contains the data loading and preprocessing functions. The trained models will be stored in the **models** folder.

## Dependencies
To install the necessary dependencies, execute the command 
```$ pip install -r requirements.txt"```
