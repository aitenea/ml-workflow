import pubchempy as pcp
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import warnings
from mlworkflow.sanitation import check_str, check_strs
import pandas as pd
import json
import importlib.resources
from tqdm import tqdm
import logging

import pandas as pd
import scipy.constants.constants as cte
from numpy import exp

def name_to_smile(name):
    """
    Transform a compound name into a SMILE string
    :param name: the name of the compound
    :return: the canonical SMILE string
    """
    check_str(name)

    try:
        res = pcp.get_compounds(name, 'name')[0].canonical_smiles
    except IndexError:
        res = None
        warnings.warn('No compound named "' + name + '" found in the database.')

    return res


def calc_tanimoto(c1, c2, fingerprint='rdk'):
    """
    Calculate the Tanimoto index between two compounds in SMILE form.
    On SMILES not recognized by RdKit, this function returns 0. This is a limitation of RdKit because it is unable to
    parse that specific SMILE into a mol, either due to a poorly constructed SMILE or to a coordinate bond in the
    SMILE that RdKit doesn't support.
    :param c1: The SMILE from the first compound
    :param c2: The SMILE from second compound
    :param fingerprint: The fingerprint descriptor used. Either "rdk" or "morgan"
    :return: The calculated Tanimoto index between c1 and c2
    """
    check_strs(c1, c2, fingerprint)

    m1 = Chem.MolFromSmiles(c1)
    m2 = Chem.MolFromSmiles(c2)

    fingerprint = fingerprint.lower()

    if m1 is None or m2 is None:
        res = 0
    elif fingerprint == 'rdk':
        fp1 = Chem.RDKFingerprint(m1)
        fp2 = Chem.RDKFingerprint(m2)
        res = DataStructs.TanimotoSimilarity(fp1, fp2)
    elif fingerprint == 'morgan':
        fp1 = rdMolDescriptors.GetMorganFingerprint(m1, 2)
        fp2 = rdMolDescriptors.GetMorganFingerprint(m2, 2)
        res = DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        raise ValueError('Fingerprint descriptor not recognized')

    return res


def find_k_tani(df, col, c, k=10, fingerprint='rdk'):
    """
    Find the k nearest compounds to c inside column col of dataframe df in terms of Tanimoto index.
    :param df: a pandas dataframe with the SMILES of the compounds
    :param col: the name of the column with the SMILES
    :param c: the SMILE of the compound to check
    :param k: the number of compounds to return
    :param fingerprint: The fingerprint descriptor used. Either "rdk" or "morgan"
    :return: a dataframe with the SMILES of the k most similar compounds and their calculated Tanimoto index
    """
    df['tani_idx'] = df[col].map(lambda x: calc_tanimoto(c, x, fingerprint))
    res = df.sort_values(by='tani_idx', ascending=False)[[col, 'tani_idx']].iloc[0:k]
    del df['tani_idx']

    return res


def print_cond(condition, text):
    if condition:
        logging.info(text)


def eyring_eq(e_barrier):
    T = 298
    k = ((cte.Boltzmann * T) / cte.Planck) * exp(-e_barrier / (cte.gas_constant * T))

    return k


def eyring_ratio(alfa, beta):
    alfa_ey = eyring_eq(alfa * 4184)
    beta_ey = eyring_eq(beta * 4184)

    return alfa_ey / beta_ey