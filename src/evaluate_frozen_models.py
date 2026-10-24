
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "ache_ic50_clean.csv"
SPLITS_PATH = PROJECT_DIR / "configs" / "ache_frozen_splits.npz"

RF_PATH = PROJECT_DIR / "models" / "ache_rf_frozen.joblib"
SVR_PATH = PROJECT_DIR / "models" / "ache_tanimoto_svr_frozen.joblib"


def evaluate(y_true, y_pred):
    return np.array([
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
    ])


def main():

    print("AChE Bioactivity ML")
    print("Frozen-model test evaluation")
    print("-" * 45)

    # Load data and frozen splits
    df = pd.read_csv(DATA_PATH)

    with np.load(SPLITS_PATH) as splits:
        train_idx = splits["train_idx"].copy()
        test_idx = splits["test_idx"].copy()

    assert len(train_idx) == 4099
    assert len(test_idx) == 1256
    assert len(np.unique(test_idx)) == len(test_idx)

    train_smiles = df.iloc[train_idx]["canonical_smiles"]
    test_smiles = df.iloc[test_idx]["canonical_smiles"]

    y_test = df.iloc[test_idx]["pIC50"].to_numpy()

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    def get_molecule(smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        return mol

    # --------------------------------------------------
    # Random Forest: Morgan count fingerprints
    # --------------------------------------------------

    print("\nEvaluating Random Forest...")

    rf = joblib.load(RF_PATH)

    X_test_rf = np.asarray([
        generator.GetCountFingerprintAsNumPy(
            get_molecule(smiles)
        )
        for smiles in test_smiles
    ])

    rf_predictions = rf.predict(X_test_rf)
    rf_metrics = evaluate(y_test, rf_predictions)

    # --------------------------------------------------
    # Tanimoto-SVR: Morgan binary fingerprints
    # --------------------------------------------------

    print("Evaluating Tanimoto-SVR...")

    svr = joblib.load(SVR_PATH)

    train_fps = [
        generator.GetFingerprint(get_molecule(smiles))
        for smiles in train_smiles
    ]

    test_fps = [
        generator.GetFingerprint(get_molecule(smiles))
        for smiles in test_smiles
    ]

    K_test = np.asarray([
        DataStructs.BulkTanimotoSimilarity(
            fp, train_fps
        )
        for fp in test_fps
    ], dtype=np.float64)

    assert K_test.shape == (1256, 4099)
    assert svr.shape_fit_ == (4099, 4099)

    svr_predictions = svr.predict(K_test)
    svr_metrics = evaluate(y_test, svr_predictions)

    # --------------------------------------------------
    # Report and verify results
    # --------------------------------------------------

    expected = {
        "Random Forest": np.array([
            0.565433, 0.815533, 0.696993
        ]),
        "Tanimoto-SVR": np.array([
            0.547969, 0.788877, 0.716477
        ]),
    }

    results = {
        "Random Forest": rf_metrics,
        "Tanimoto-SVR": svr_metrics,
    }

    print("\nHeld-out test set:", len(test_idx), "molecules")
    print("-" * 45)

    all_passed = True

    for model_name, metrics in results.items():

        matches = np.allclose(
            metrics,
            expected[model_name],
            atol=1e-6,
            rtol=0,
        )

        all_passed = all_passed and matches

        print(f"\n{model_name}")
        print(f"  MAE:  {metrics[0]:.6f}")
        print(f"  RMSE: {metrics[1]:.6f}")
        print(f"  R²:   {metrics[2]:.6f}")
        print(f"  Matches reported metrics: {matches}")

    print("\n" + "-" * 45)

    if all_passed:
        print("PASS: Both frozen models reproduced.")
    else:
        raise RuntimeError(
            "FAIL: At least one model did not reproduce "
            "the reported test metrics."
        )


if __name__ == "__main__":
    main()

