# Diffusion based Non-Negative Matrix Factorization
A modified version of non-negative matrix factorization that accounts for a diffusion process between features in the data. Non negative matrix factorization consists of splitting a data matrix into two matrices representing both the basis vectors and the coefficients needed for their linear composition. If diffusion is assumed amongst the features of the data then the data might be split further, giving instead a matrix of basis vectors, a sparse matrix of coefficients, and a third matrix which captures the diffusion process and factors it out of the matrix of coefficients. For more information see the paper (Need link or to include that here).

This repository contains the tools needed to implement Diffusion-NMF, for use please consult the examples or the documentation included in the following files:

- **diff_nmf.py** 
Is a full implementation of the D-NMF algorithm. For use please consult the documentation provided in the file

- **nmf.py**
Is a straight-forward implementation of NMF under the traditional update rules outlined by [Lee and Seung](https://papers.nips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)

- **data_generator.py**
Contains tools for generating synthetic data and for preparing real data for testing. Note that this implementation of NMF and D-NMF allows for hiding data in order to do training/testing procedures -- a result of the work done [here](https://www.guangtunbenzhu.com/nonnegative-matrix-factorization)

- **grid_search.py**
Can used for testing the algorithm for hyper-parameter selection. In particular the algorithm requires both a rank parameter and a parameter used to describe the extent of the spread in the diffusion process. Selecting appropriate values for such requires trying different combinations and selecting the ones with appropriate magnitude and minimal error.

- **examples/**
Contains a few jupyter notebooks with detailed examples on how we used the code for analyzing COVID-19 data.

To download all requirements via Anaconda (replacing <env> with desired name):
```
conda create --name <env> --file conda_requirements.txt
```

A less intensive install can be done with:
```
pip install -r requirements.txt
```
However this will NOT download the geopandas package required for creating maps, as it seems to cause problems with Windows. To download geopandas afterwards please look to https://geopandas.org/en/stable/getting_started/install.html

## COVID-19 Data

Also included are prepared COVID-19 datasets. All Covid case data is supplied by [Johns Hopkins](https://github.com/CSSEGISandData/COVID-19). For each of the country-wide, US state-wide, and US county-wide data sets we've included data for the regions cumulative COVID-19 case counts since April 2020 (still being updated at irregular intervals), data about the region's populations, the laplacian matrix corresponding to the region's adjacency graph, and the geo-json files needed for creating visual maps of the region. This data was obtained from various sources: 
- [county adjacency](https://www.census.gov/geographies/reference-files/2010/geo/county-adjacency.html) 
- [county populations](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html)
- [state adjacency](https://data.world/bryon/state-adjacency)
- [state population](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html) (aggregate of counties)
- [county and state geo-json](https://eric.clst.org/tech/usgeojson/)
- [country adjacency](https://github.com/geodatasource/country-borders) (roughly)
- [country population](https://data.worldbank.org/indicator/SP.POP.TOTL)
- [country geo-json](https://datahub.io/core/geo-countries#resource-countries)


