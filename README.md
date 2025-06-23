# KMeansClustering
A Custom K-Means clustering algorithm implementation using python3 and basic python types only.

## main.py
Contains the complete implementation of the `Point` class i.e., an n-dimensional vector and `KMeansClustering` class to implement the K-Means clustering model.

## KMeans.ipynb
Contains cells to implement the `Point` and `KMeansClustering` class and run a comparision between the custom and sklearn.cluster.KMeans model with random initialization.

## how to run commands
(using astral-sh's uv python version manager)
```bash
uv sync
uv run main.py
```
(using pip)
```bash
pip install -r "requirements.txt"
python3 ./main.py
```
