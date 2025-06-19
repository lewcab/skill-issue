"""
Main script for training and evaluating machine learning models on match data.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from constants import SEED
from data_processor import DataProcessor


def main():
    print("skill issue ...")

    data_files = [
        "data/match-data.csv",
    ]
    data = DataProcessor(data_files)

    n_samples = len(data.x_train) + len(data.x_test)
    n_features = data.x_train.shape[1] if data.x_train.size else 0

    print(f"Data loaded and processed. ({n_samples}, {n_features})", end='\n\n')
    print(data)

    # Setting up the Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        max_features='sqrt',
        random_state=SEED,
    )

    # Fit the model to the training data
    gbc.fit(data.x_train, data.y_train)

    # Score/test the model on the training and testing data
    print("Gradient Boosting Classifier trained.", end='\n\n')
    print("Training score:", gbc.score(data.x_train, data.y_train))
    print("Testing score:", gbc.score(data.x_test, data.y_test), end='\n\n')

    # Setting up the Multi-layer Perceptron Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        random_state=SEED,
    )

    # Fit the model to the training data
    mlp.fit(data.x_train, data.y_train)

    # Score/test the model on the training and testing data
    print("Multi-layer Perceptron Classifier trained.", end='\n\n')
    print("Training score:", mlp.score(data.x_train, data.y_train))
    print("Testing score:", mlp.score(data.x_test, data.y_test), end='\n\n')


if __name__ == '__main__':
    main()
