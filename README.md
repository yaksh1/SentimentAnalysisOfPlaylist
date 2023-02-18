# Sentiment Analysis on Songs Playlist

We created a web-app using `streamlit` library to perform sentiment analysis on spotify songs playlist entered by the user using various machine learning algorithms. 

## Dataset Information

We use and compare various different methods for sentiment analysis on songs (a binary classification problem). The training dataset is a `csv-file` from a research paper which contains artist names and song names along with pre-determined `sentiment` which contains `positive` or `negative`.  

## Requirements

There are some general library requirements for the project:  
* `numpy`
* `pandas`
* `scikit-learn`
* `scipy`
* `nltk`
* `streamlit`
* `matplotlib`

## Usage

1. Run `main.ipynb` to train the models or add the models of your choice to perform sentiment analysis.
2. Downlaod `music.xlsx` for a pre-determined sentiment of songs dataset.
3. Place all the saved models in `models/` folder with `.pkl` extension.
4. Run `main.py` using `streamlit run main.py`.

## Information about files

* `music.xlsx`: List of songs and artists with playlist.
* `models/`: List of all the saved models.
* `main.ipynb`: training file.
* `main.py`: file to run for a web app solution.
