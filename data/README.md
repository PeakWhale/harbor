# Demo data

## california_housing.csv

The default Harbor demo. One row is one California district (a census
block group) in 1990. The target column is `median_house_value`.

* Rows: 20,640. Columns: 10 (9 numbers, 1 category).
* SHA-256: `8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e`
* Downloaded from:
  <https://raw.githubusercontent.com/ageron/handson-ml2/7b7e23e7267356f8355580877eff98c43cda1bd0/datasets/housing/housing.csv>

Where it comes from:

* Original data: R. Kelley Pace and Ronald Barry, "Sparse Spatial
  Autoregressions", *Statistics and Probability Letters*, 33 (1997) 291-297.
  They built it from the 1990 U.S. Census. U.S. Census data is a work of the
  U.S. government and is in the public domain.
* This copy comes from Aurélien Géron's
  [handson-ml2](https://github.com/ageron/handson-ml2) repository, which is
  released under the Apache License 2.0. Géron made two small changes: he
  removed 207 values from `total_bedrooms` (to show missing data) and added
  the `ocean_proximity` column (to show a text category).
* The same data, without those two changes, ships with scikit-learn as
  `sklearn.datasets.fetch_california_housing`.

Things to know about this data:

* It is from 1990. It does not describe today's prices.
* `median_house_value` is capped at 500,001 and `housing_median_age` is
  capped at 52. The model cannot predict above these caps.
* `median_income` is in tens of thousands of US dollars (3.5 means $35,000).
* Each row is a whole district, not a single house.

`california_housing.columns.json` holds the short column descriptions that
the web form shows.

## Why not the Boston housing data?

Earlier versions of Harbor used the Boston housing data. That data set is
retired: one of its columns was built from the share of Black residents in
each town, and scikit-learn removed it in version 1.2. Harbor no longer uses
it.

## ../examples/equipment_repairs.csv

A small, made-up (synthetic) data set that shows Harbor works on data that is
not about housing. It is created by `scripts/make_example_data.py` with a fixed
random seed, so anyone can create the same file again. It contains no real
machines, companies or people. It is released under the same license as this
repository (Apache-2.0).
