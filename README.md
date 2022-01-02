# cs682-humor-offense
Using pre-trained model ensembles with LSTMs to classify and rate humor and offense

## Steps
1. First clone the repository on Colab. Navigate to destination folder on your machine and run the following in a Colab Notebook

```
!git clone https://github.com/philipgeorge94/cs682-humor-offense.git
```

2. If you're not running this on google colab, dependencies will have to be installed on terminal/conda/PyCharm venv etc
These include, but are not limited to, the following four.
```
$ pip install transformers --quiet
$ pip install datasets transformers[SentencePiece] --quiet
$ pip install pyter3 --quiet
$ pip install torchmetrics --quiet
```

3. Run the cells in main.ipynb. If you're using the free version of Colab you might have to modify the 'tasks' and 'checkpoints' parameters, to ensure that you dont time-out in the middle of training.
4. You might also have to change some of the file paths in **main.ipynb** and **code/utils.py**