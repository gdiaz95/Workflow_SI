import numpy as np
import pandas as pd
import pytest
import sys
import os

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.npgc import NPGC


def _build_mixed_dataframe(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "int_col": rng.integers(0, 20, size=rows),
            "float_col": rng.normal(loc=10.0, scale=2.5, size=rows),
            "cat_col": rng.choice(["A", "B", "C"], size=rows, p=[0.5, 0.3, 0.2]),
        }
    )

    # Inject NaNs in every supported kind (int/float/categorical)
    nan_rows = rng.choice(rows, size=rows // 5, replace=False)
    df.loc[nan_rows[: rows // 15], "int_col"] = np.nan
    df.loc[nan_rows[rows // 15 : 2 * rows // 15], "float_col"] = np.nan
    df.loc[nan_rows[2 * rows // 15 :], "cat_col"] = np.nan
    return df


def test_fit_rejects_non_dataframe_and_empty_dataframe():
    synth = NPGC(epsilon=None)

    with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
        synth.fit([1, 2, 3])

    with pytest.raises(ValueError, match="Data is empty"):
        synth.fit(pd.DataFrame())


def test_mixed_tabular_sampling_handles_nan_integer_continuous_and_categorical():
    train = _build_mixed_dataframe(rows=300)
    synth = NPGC(epsilon=None)
    synth.fit(train)

    sampled = synth.sample(num_rows=400, seed=123)

    assert list(sampled.columns) == ["int_col", "float_col", "cat_col"]
    assert len(sampled) == 400

    # Integer support: every non-null value should stay integer-like.
    int_non_null = sampled["int_col"].dropna().to_numpy()
    assert np.allclose(int_non_null, np.round(int_non_null))

    # Categorical support: no unexpected labels
    sampled_labels = set(sampled["cat_col"].dropna().unique())
    assert sampled_labels.issubset({"A", "B", "C"})

    # NaN handling: at least one NaN appears in sampled output.
    assert sampled.isna().any().any()


def test_sampling_is_reproducible_for_same_seed_on_same_fitted_model():
    train = _build_mixed_dataframe(rows=250, seed=15)
    synth = NPGC(epsilon=None)
    synth.fit(train)

    sample_a = synth.sample(num_rows=150, seed=999)
    sample_b = synth.sample(num_rows=150, seed=999)

    pd.testing.assert_frame_equal(sample_a, sample_b)


def test_sample_before_fit_raises_runtime_error():
    synth = NPGC(epsilon=None)

    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        synth.sample(num_rows=10, seed=1)


def test_save_and_load_keep_model_usable(tmp_path):
    train = _build_mixed_dataframe(rows=120, seed=22)
    model_path = tmp_path / "models" / "npgc.pkl"

    synth = NPGC(epsilon=None)
    synth.fit(train)
    synth.save(str(model_path))

    loaded = NPGC(epsilon=None)
    loaded.load(str(model_path))

    sampled = loaded.sample(num_rows=50, seed=17)

    assert sampled.shape == (50, 3)
    assert set(sampled.columns) == {"int_col", "float_col", "cat_col"}


def test_single_value_continuous_column_generates_mostly_same_value():
    repeated_value = 42.5
    train = pd.DataFrame({"only_continuous": [repeated_value] * 500})

    synth = NPGC()
    synth.fit(train)

    sampled = synth.sample(num_rows=500, seed=21)

    assert list(sampled.columns) == ["only_continuous"]
    close_ratio = np.mean(np.isclose(sampled["only_continuous"], repeated_value, atol=1e-8))
    assert close_ratio >= 0.95


if __name__ == "__main__":
    print("Running NPGC tests...")
    exit_code = pytest.main([__file__, "-q"])
    print(f"Test run completed with exit code: {exit_code}")
    sys.exit(exit_code)
