import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import selfies as sf
from tqdm.auto import trange
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


FILE_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(FILE_DIR, '..', '..', 'data')
LOGS_DIR = os.path.join(FILE_DIR, '..', '..', 'logs')
GOLD_PATH = os.path.join(DATA_DIR,
                         'reagent-pred',
                         'smiles',
                         'atom',
                         'x1',
                         'tgt-test.txt')
PRED_PATH = os.path.join(LOGS_DIR,
                         'reagent-pred',
                         'smiles',
                         'atom',
                         'x1',
                         'from-scratch',
                         'test_predictions.txt')
FIG_PATH = os.path.join(FILE_DIR, 'fig4.tiff')
NUM_PRED = 10  # how many predictions per sample
MAX_REAGENTS = 12  # up to how many reagents per reaction the analysis goes


def do_plot():
    # Load predictions and true labels
    predictions = load_reagents(PRED_PATH, NUM_PRED)
    labels = load_reagents(GOLD_PATH)

    # Cluster the instances by number of true reagents
    clusters = cluster_by_num_reagents(labels, predictions)

    # Compute accuracy as a function of number of true reagents
    # and number of prediction considered
    matrix = topn_accuracy_matrix(clusters, NUM_PRED)
    save_accuracy_matrix_as_heatmap(matrix.iloc[:MAX_REAGENTS, :], FIG_PATH)
    print('- Plotted figure 4 at %s!' % FILE_DIR)


def standardize_molecules(sample_line: str, fmt: str = 'smiles') -> List[str]:
    """
    Clean and standardize molecules.

    Parameters:
        sample_line (str): The string representation of a molecule.
        fmt (str): The format of the input molecule string. 
            'smiles' (default) or 'selfies'

    Returns:
        A list of standardized and cleaned molecules.
    """
    # Removing all whitespace characters from the input line
    sample_line = ''.join(sample_line.strip().split())
    # If format is selfies, convert the input to SMILES
    if fmt == 'selfies':
        sample_line = create_smiles_from_selfies(sample_line)
    # Canonicalize the SMILES representation
    sample_line = canonicalize_smiles(sample_line)
    # Return the standardize molecules
    return sorted(sample_line.split('.'))


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert a SMILES string to its canonical form.

    Parameters:
        smiles (str): A SMILES string.

    Returns:
        The canonical form of the SMILES string.
    """
    try:
        # Using RDKit library to canonicalize the smiles
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        # Returning the input smiles in case of any exception
        return smiles


def create_smiles_from_selfies(selfies: str) -> Union[str, List[str]]:
    """
    Convert a Selfies string to its SMILES representation.

    Parameters:
        selfies (str): A Selfies string.

    Returns:
        The SMILES representation of the Selfies string. 
        If the input is a single line, it will return a string
        If the input is multiple line, it will return list of strings
    """
    try:
        # Using Selfies library to decode the selfies
        return sf.decoder(selfies)
    except sf.DecoderError:
        # If there is any error with the input. It will split the selfies by '.'
        selfies_mols = selfies.split('.')
        smiles_mols = []
        for selfies_mol in selfies_mols:
            try:
                # Again using selfies library to decode each molecule
                smiles_mol = sf.decoder(selfies_mol)
            except sf.DecoderError:
                # If the above decode fails then it will append '?'
                smiles_mol = '?'
            smiles_mols.append(smiles_mol)
        return smiles_mols


def load_reagents(path: str,
                  num_pred_per_instance: int = 1,
                  standardize: bool = True) -> List[List[str]]:
    """
    Load reagent strings from a file and chunk them into a list of lists
    of equal size.

    Parameters:
        path (str): The path to the file containing the reagent strings.
        num_pred_per_instance (int): The number of reagent strings to include
        in each sublist. Default one.
        standardize (bool): If returned reagent should be standardized or not.
        Default True.

    Returns:
        A list of lists of SMILES strings, where each sublist contains 
        `num_pred_per_instance` SMILES strings.
    """

    # Open the file at the specified path in read mode
    with open(path, "r") as p:
        # Read all the lines
        lines = p.read().splitlines()
    # Chunk the lines into sublists with num_pred_per_instance lines per sublist
    if standardize:
        return [[standardize_molecules(j)
                for j in lines[i:i + num_pred_per_instance]]
                for i in trange(0,
                               len(lines),
                               num_pred_per_instance,
                               desc='Loading data for figure 4',
                               leave=False)]
    else:
        return [lines[i:i + num_pred_per_instance]
                for i in trange(0,
                                len(lines),
                                num_pred_per_instance,
                                desc='Loading data for figure 4',
                                leave=False)]


def cluster_by_num_reagents(leading_lists: List[List[List[str]]],
                            subordinate_lists: List[List[List[str]]])\
                            -> Dict[int, List[Tuple[List[List[str]],
                               List[List[str]]]]]:
    """
    Cluster a nested list of lists by the number of elements in the leading list

    Parameters:
        leading_lists (List[List[List[str]]]): A nested list of lists of list
        of leading strings
        subordinate_lists (List[List[List[str]]]): A nested list of lists of
        lists of subordinate strings

    Returns:
        A dictionary where keys are the number of elements in the leading list,
        and the values are the lists of tuples where tuple includes the leading
        lists and subordinate lists that match that number of elements.
    """
    # Create a defaultdict to store the clusters
    clusters = defaultdict(list)
    # Iterate over the leading and subordinate lists
    for leading_sublist1, subordinate_sublist1 in zip(leading_lists,
                                                      subordinate_lists):
        # Iterate over the leading sublists in the outer list
        for leading_sublist2 in leading_sublist1:
            # Get the length of the leading_sublist2
            subsublist_len = len(leading_sublist2)
            # Append the tuple of leading and subordinate sublists
            # to the appropriate cluster
            clusters[subsublist_len].append(
                (leading_sublist1, subordinate_sublist1))
    # Sort the clusters by key and return as a dictionary
    return dict(sorted(clusters.items()))


def is_in_top(y_true: List[List[str]], y_hat: List[List[str]], n: int) -> bool:
    """
    Check if the true value is in the top n predictions.

    Parameters:
        y_true (List[List[str]]): A list of true values.
        y_hat (List[List[str]]): A list of predictions.
        n (int): Number of top predictions to check.

    Returns:
        A boolean indicating if the true value is in the top n predictions.
    """
    # Check if the true value is in the top n predictions
    return y_true[0] in y_hat[:n]


def is_in_top_lenient(y_true: List[List[str]],
                      y_hat: List[List[str]], n: int) -> bool:
    """
    Check if the true value is in the top n predictions using lenient match
    i.e. we do not penalize the excess of reagents.

    Parameters:
        y_true (List[List[str]]): A list of true values.
        y_hat (List[List[str]]): A list of predictions.
        n (int): Number of top predictions to check.

    Returns:
        A boolean indicating if the true value is in the top n predictions.
    """
    # Check if the true value is in the top n predictions
    return sum(all(elem in y_hat_i for elem in y_true[0])
               for y_hat_i in y_hat[:n]) >= 1


def topn_accuracy(cluster: List[Tuple[List[List[str]], List[List[str]]]],
                  n: int) -> float:
    """
    Compute the top-n accuracy for the given cluster
    of true values and predictions.

    Parameters:
        cluster (List[Tuple[List[List[str]], List[List[str]]]]): A list of
        tuples of true values and predictions.
        n (int): Number of top predictions to consider.

    Returns:
        A float indicating the top-n accuracy of the predictions.
    """
    # Initialize a variable to store the number of correct predictions
    correct = 0
    # Iterate over the true values and predictions in the cluster
    for y_true, y_hat in cluster:
        # Increment the correct counter if the true value is
        # in the top n predictions
        correct += is_in_top(y_true, y_hat, n) * 1
    # Compute the accuracy
    accuracy = correct / len(cluster)
    return accuracy


def topn_accuracy_lenient(cluster: List[Tuple[List[List[str]],
                                              List[List[str]]]],
                          n: int) -> float:
    """
    Compute the top-n accuracy for the given cluster of true values and
    predictions using lenient match i.e. we do not penalize
    the excess of reagents.

    Parameters:
        cluster (List[Tuple[List[List[str]], List[List[str]]]]): A list of
        tuples of true values and predictions.
        n (int): Number of top predictions to consider.

    Returns:
        A float indicating the top-n accuracy of the predictions.
    """
    # Initialize a variable to store the number of correct predictions
    correct = 0
    # Iterate over the true values and predictions in the cluster
    for y_true, y_hat in cluster:
        # Increment the correct counter if the true value is
        # in the top n predictions
        correct += is_in_top_lenient(y_true, y_hat, n) * 1
    # Compute the accuracy
    accuracy = correct / len(cluster)
    return accuracy


def topn_accuracy_matrix(clusters: dict, max_n: int) -> pd.DataFrame:
    """
    Create a DataFrame with top-n accuracy for all clusters, with the number
    of reagents and the accuracy as rows and columns respectively

    Parameters:
        clusters (dict) : Dictionary with the cluster of values and predictions
        max_n (int) : Maximum number of top predictions to consider as correct

    Returns:
        pd.DataFrame : DataFrame with top-n accuracy for all clusters, with
        the number of reagents and the accuracy as rows and columns respectively
    """
    # Create an empty DataFrame
    df = pd.DataFrame(index=[f"# Reagents {i}" for i in range(1, max_n + 1)],
                      columns=[f"Top {i}" for i in range(1, max_n + 1)],
                      dtype="float")

    for key, value in clusters.items():
        for i in range(1, max_n + 1):
            # Compute the top-n accuracy for the current cluster
            accuracy = topn_accuracy(value, i)
            # Set the corresponding value in the DataFrame
            df.at[f"# Reagents {key}", f"Top {i}"] = accuracy
    return df


def topn_accuracy_matrix_lenient(clusters: dict, max_n: int) -> pd.DataFrame:
    """
    Create a DataFrame with top-n accuracy for all clusters, with the number
    of reagents and the accuracy as rows and columns respectively using lenient
    match i.e. we do not penalize the excess of reagents.

    Parameters:
        clusters (dict) : Dictionary with the cluster of values and predictions
        max_n (int) : Maximum number of top predictions to consider as correct

    Returns:
        pd.DataFrame : DataFrame with top-n accuracy for all clusters,
        with the number of reagents and the accuracy
        as rows and columns respectively
    """
    # Create an empty DataFrame
    df = pd.DataFrame(index=[f"# Reagents {i}" for i in range(1, max_n + 1)],
                      columns=[f"Top {i}" for i in range(1, max_n + 1)],
                      dtype="float")

    for key, value in clusters.items():
        for i in range(1, max_n + 1):
            # Compute the top-n accuracy for the current cluster
            accuracy = topn_accuracy_lenient(value, i)
            # Set the corresponding value in the DataFrame
            df.at[f"# Reagents {key}", f"Top {i}"] = accuracy
    return df


def save_accuracy_matrix_as_heatmap(data: pd.DataFrame,
                                    path: str = '_.tiff',
                                    colorbar: bool = False,
                                    figsize: tuple = (8, 8),
                                    fontsize: int = 14) -> None:
    """
    This function plots a heatmap of the input accuracy matrix.
    The x-axis shows the 'top N' accuracy, and the y-axis shows
    the number of reagents.

    Parameters
    ----------
    data : pandas.DataFrame
        The matrix to be plotted.
    colorbar : bool, optional
        Show colorbar or not, by default False
    figsize : tuple, optional
        The size of the figure
    fontsize : int, optional
        Fontsize of the ticklabels and colorbar labels, by default 14

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    pc = ax.pcolormesh(data)
    if colorbar:
        plt.colorbar(pc)
    ax.set_xlabel('Top N accuracy', fontsize=fontsize+2)
    ax.set_ylabel('Number of reagents', fontsize=fontsize+2)
    ax.set_xticks(np.linspace(1, 10, 10)-0.5)
    ax.set_xticklabels(np.linspace(1, 10, 10, dtype=int), fontsize=fontsize)
    ax.set_yticks(np.linspace(1, 12, 12)-0.5)
    ax.set_yticklabels(np.linspace(1, 12, 12, dtype=int), fontsize=fontsize)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j + 0.5,
                    i + 0.5,
                    np.round(data.iloc()[i, j], 2),
                    va='center',
                    ha='center',
                    color='white',
                    fontsize=fontsize,
                    fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300)


if __name__ == '__main__':
    do_plot()
