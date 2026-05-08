# MountainCar Browser Demo

## Run

```bash
cd /home/mochan/workspace/is2/js
npm start
```

Then open `http://localhost:8000`.

## Modes

- `Collect Data`: runs random/mixture behavior policy and tracks live metrics + visitation heatmap.
- `Run Trained Policy`: loads a Q-table JSON and runs greedy policy in the browser environment.

## Q-table JSON format

Accepted:

1. Raw 3D array:

```json
[[[0,0,0], ...], ...]
```

2. Object payload:

```json
{
  "q_table": [[[0,0,0], ...], ...],
  "n_bins": 20
}
```

`q_table` shape must be `[n_bins][n_bins][3]`.
