## Data format
- The notebook expects a single CSV table where each row is a feature vector and includes at least these columns:
  - participant (person id or name)
  - session (session id or name)
  - label (target class)
  - many feature columns (numeric)

- Example row: [feature1, feature2, ..., participant, session, label]

## What to edit in `run_ML.ipynb`
Open the notebook and change these cells near the top:

- Data file: change the CSV read path in the cell that contains:

```python
df = pd.read_csv('ecg_features_final_standard.csv')
```

- Column names: update the variables that control column names
- If your participant/session/label columns have different names, set them here.

```python
ID_col = "participant"
session_col = "session"
target_col = 'label'
drop_cols = [ID_col, session_col]
```


- Label encoding: the notebook runs `LabelEncoder()` on `participant`. If your `participant` column is already numeric, you can skip or remove that block.

- Binary target: the notebook sets the `label` to binary automaticly with:

```python
df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
```

- If you need to change the `string` labels for numerical:
```
label_mapping = {'REST': 0, 'Stroop': 1, 'Reaction': 2, 'N-Back': 3}
df['label'] = df['label'].map(label_mapping)
```

- Drop non-feature columns: **ensure `drop_cols` contains all non-feature columns** (IDs, session, label and any metadata). The code used for training does `X = df.drop(columns=[target_col])` and expects you to pass `drop_cols` to the `train_loso` function so only feature columns are used.

## Model configs and tuning
- The cell `model_configs` contains the list of models and their parameter grids. Edit this if you want different models or hyperparameter ranges.

## Running model
- Function `scores = tr.train_loso(...)` will automaticlly apply Leave-one-subject-out (LOSO) cross validation
- return scores dictionary which include *(['person_id', 'model', 'evaluation'])*
- Finally, function `df_train, df_val = tr.summarize_scores(scores)` transforme the `scores` into separate dataframes and calculate mean ± std

**Notice**: This repository does not accept public pull requests. Any submitted PRs will be automatically closed.
