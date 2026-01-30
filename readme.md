# Non-Parametric Gaussian Copula Synthesizer

## Setup (using Poetry)

If you already have Python and Poetry installed and this folder contains
`pyproject.toml` and `poetry.lock`, run:

```bash
poetry install
```

This will create the virtual environment and install all dependencies.

To activate the environment:

```bash
poetry env activate
```
Poetry will print a command in the terminal.
Copy and paste that command and run it to activate the environment. After activation, you can run Python normally.


## Synthesizer Parameters

### enforce_min_max_values (bool, default=True)

Intended to enforce the minimum and maximum values observed in the training data during sampling.

---

### epsilon (float or None, default=1.0)

Controls the level of differential privacy noise applied during training.

- If epsilon is None or ≤ 0 → no noise is added (non-private mode).
- Smaller epsilon → more noise → stronger privacy → lower fidelity.
- Larger epsilon → less noise → weaker privacy → higher fidelity.

---

## Method Parameters

### fit(data, epsilon=None)

- data: pandas DataFrame containing the real table.
- epsilon: optional override of the model’s epsilon value for this fit call only.
  - If None, the model uses the epsilon defined during initialization.

---

### sample(num_rows, seed=None)

- num_rows: number of synthetic rows to generate.
- seed: optional integer for reproducible sampling.
  - Same seed + same fitted model → identical synthetic output.

---

### save(filepath)

Saves the fitted model as a .pkl file.  
The directory is created automatically if it does not exist.

---

### load(filepath)

Loads a previously saved model from a .pkl file into the current instance.


## Example

Simple example of how to use the synthesizer with a pandas DataFrame.


```python
import pandas as pd
from source.non_parametric import NonParamGaussianCopulaSynthesizer

# 1) Create or load your table (must be a pandas DataFrame)
df = pd.read_csv("your_table.csv")   # or create your own DataFrame

# 2) Create the synthesizer
synth = NonParamGaussianCopulaSynthesizer(epsilon=1.0)  
# epsilon=None disables noise

# 3) Fit the model to the real data
synth.fit(df)

# 4) Generate synthetic data
synthetic_df = synth.sample(num_rows=1000, seed=42)

# 5) Save synthetic data if needed
synthetic_df.to_csv("synthetic_data.csv", index=False)

# Optional: save the trained model
synth.save("npgc_model.pkl")

# Optional: load a saved model
# synth2 = NonParamGaussianCopulaSynthesizer()
# synth2.load("npgc_model.pkl")
# synthetic_df = synth2.sample(1000)
