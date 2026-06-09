
# Response Variability and Stability in Human Reasoning

This is the companion repository for the CogSci 2026 article "Response Variability and Stability in Human Reasoning" by C. Bombach, R. Lakshmanan and M. Ragni.

## Overview

All relevant scripts for the analysis and the data preparation are contained in the folder `analysis`:

* `K-meansClustering.ipynb`: contains code for the clustering analysis.
* `compare_distances.py`: Data preparation for the distance comparison (Q2)
* `model_comaprison.R` Comparison of GLMMS  for Q1
* `plots.py` CDF plot (Q2)
* `prepare_data.py` Data preparation
* `sanity_check.R`Sanity check with alternate (one-hot) encoding (Q1)
* `syllog_encoding.py` Classes and helper functions for encoding syllogisms and calculating distances/energy
* `test_retest_distance.R` Test-Retest Distance comparison for Q2.

The data can be found in the folder `analysis\data`:

* `dames.csv` Cleaned data from Dames, H., Klauer, K. C., & Ragni, M. (2022). The stability of syllogistic reasoning performance over time. Thinking & Reasoning, 28(4), 529–568
* `dames_with_energy.csv` Dames dataset with Energy measure
* `distances_retest.csv`Distances between reasoners on retest
* `distances_test.csv`Distances between reasoners on test
* `distances_test_retest.csv`Distance between test and retest
* `pivot_table.csv` Data in wide format (used for Q1)
