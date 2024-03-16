# Double Moon

This binary classification task involves categorizing points in a 2D plane that belong to two sets resembling intertwined moons.

## Usage Notes

The folder `generate_resources` contains the scripts used to generate the data and the indices splits for this datamanager.

## DataManager Versions

Default is `v1`.

| Version | Nickname              |
|---------|-----------------------|
| `v1`    | Double Moon           |
| `v2`    | Noisy Double Moon     |

### Data Versions

| Version | Description                          |
|---------|--------------------------------------|
| `v1`    | noise = 0                            |
| `v2`    | noise = 0.16                         |

### Indices Splits Versions

| Version | Description                          |
|---------|--------------------------------------|
| `v1`    | Nested k-folds CV (5 outer, 5 inner) |
| `v2`    | A copy of `v1`                       |