import matplotlib.pyplot as plt
import pandas as pd

GBC_PATH = 'solutions/gbc_solutions.csv'
NN_PATH = 'solutions/nn_solutions.csv'


def main():
    analysis_gbc()
    analysis_nn()


def analysis_gbc():
    print("+--- Analysis on Gradient Boosting Classifier (GBC) ---+")
    solutions = parse_file(GBC_PATH)
    solutions.sort_values("accuracy", ascending=False, inplace=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(solutions[:15])
    print(solutions.describe())

    plot('GBC', solutions, 'scatter', 'learning_rate', scale='log')
    plot('GBC', solutions, 'scatter', 'n_estimators')
    plot('GBC', solutions, 'scatter', 'subsample')
    plot('GBC', solutions, 'scatter', 'max_depth')
    plot('GBC', solutions, 'scatter', 'min_samples_split')
    plot('GBC', solutions, 'scatter', 'min_samples_leaf')
    plot('GBC', solutions, 'scatter', 'max_features')
    print()


def analysis_nn():
    print("+--- Analysis on Neural Network (NN) ---+")
    solutions = parse_file(NN_PATH)
    solutions.sort_values("accuracy", ascending=False, inplace=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(solutions[:15])
    print(solutions.describe())

    plot('NN', solutions, 'scatter', 'learning_rate_init', scale='log')
    plot('NN', solutions, 'scatter', 'alpha', scale='log')
    plot('NN', solutions, 'scatter', 'max_iter')
    plot('NN', solutions, 'scatter', 'hidden_layer_sizes')
    plot('NN', solutions, 'scatter', 'activation')
    plot('NN', solutions, 'scatter', 'solver')
    plot('NN', solutions, 'scatter', 'learning_rate')
    print()


def parse_file(filename: str) -> pd.DataFrame:
    """
    Parses a CSV file and returns a DataFrame.
    :param filename: The name of the file to parse.
    :return: A DataFrame containing the parsed data.
    """
    try:
        df = pd.read_csv(filename)
        print(f"File {filename} parsed successfully.")
        return df
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return pd.DataFrame()


def plot(model: str, solutions: pd.DataFrame, kind: str, x: str, y: str = 'accuracy', scale: str = None) -> None:
    ax = solutions.plot(
        x=x, y=y, kind=kind,
        title=f'{model}: {x} vs. {y}',
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if scale:
        ax.set_xscale(scale)
    plt.savefig(f'figures/{model.lower()}-{x}_{y}.png')


if __name__ == '__main__':
    main()
