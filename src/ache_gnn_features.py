import torch

from rdkit import Chem
from torch_geometric.data import Data


def one_hot_with_unknown(value, choices):
    if value not in choices[:-1]:
        value = "Other"

    return [int(value == choice) for choice in choices]


elements = [
    "C", "N", "O", "F", "P", "S",
    "Cl", "Br", "I",
    "B", "Si", "Se", "Na",
    "Other"
]

degrees = [0, 1, 2, 3, 4, "Other"]

formal_charges = [-1, 0, 1, "Other"]

hybridizations = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    "Other"
]

hydrogen_counts = [0, 1, 2, 3, "Other"]

bond_types = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
    "Other"
]


def get_atom_features(atom):
    return (
        one_hot_with_unknown(atom.GetSymbol(), elements)
        + one_hot_with_unknown(atom.GetDegree(), degrees)
        + one_hot_with_unknown(
            atom.GetFormalCharge(), formal_charges
        )
        + one_hot_with_unknown(
            atom.GetHybridization(), hybridizations
        )
        + [int(atom.GetIsAromatic())]
        + one_hot_with_unknown(
            atom.GetTotalNumHs(), hydrogen_counts
        )
    )


def get_bond_features(bond):
    return (
        one_hot_with_unknown(bond.GetBondType(), bond_types)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
    )


def smiles_to_graph(smiles, target=None):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    node_features = [
        get_atom_features(atom)
        for atom in mol.GetAtoms()
    ]

    x = torch.tensor(
        node_features,
        dtype=torch.float32
    )

    edge_list = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        features = get_bond_features(bond)

        edge_list.append([i, j])
        edge_features.append(features)

        edge_list.append([j, i])
        edge_features.append(features)

    if edge_list:
        edge_index = torch.tensor(
            edge_list, dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features, dtype=torch.float32
        )
    else:
        edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        edge_attr = torch.empty(
            (0, 7), dtype=torch.float32
        )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    if target is not None:
        data.y = torch.tensor(
            [target], dtype=torch.float32
        )

    return data
