# :video_game: League of Legends Match Predictor

A machine learning model with the goal of predicting the outcome of a League of Legends match. 
Genetic algorithms are also used to tune the model.

## :gear: Setup

The program is created using [Python 3.11](https://www.python.org/downloads/). It can be run after setting up `venv` as follows:
```
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

Once the environment is activated (command prompt has a `(venv)` prefix), a program can be run using:
```
python <script>.py
```

### :pick: Collecting Data

Most of the Data Collection functionality lies in the `data_collector.py` script.
Collecting mass amounts of data will be time-consuming, as there are rate limits for queries.
```
python data_collector.py
```

### :dna: Optimizing Hyperparameters

The optimal hyperparameters can be discovered using the Genetic Algorithm in the `main.py` script.
```
python main.py
```

### :8ball: Predicting a Future Match

It is helpful to find the team on [Leagepedia](https://lol.fandom.com/wiki/) to get the full team name.
If an abbreviated version is used, the script will fail. 
Ensure that quotes are used for teams with a space in their name.
```
python predict_match.py "Gen.G" "G2 Esports"
```

## :book: Libraries, APIs, and Databases

Below are some APIs and databases used, as well as some links to documentation and guides.
* [Leaguepedia API](https://lol.fandom.com/wiki/Help:Leaguepedia_API)
* [scikit-learn](https://scikit-learn.org/stable/index.html)
  * [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  * [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
* [PyGAD](https://pygad.readthedocs.io/)
