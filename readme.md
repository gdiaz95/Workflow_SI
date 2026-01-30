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
Copy and paste that command and run it to activate the environment. After activation, you can run Python normally


Simple example of how to use the synthesizer with a pandas table.


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
