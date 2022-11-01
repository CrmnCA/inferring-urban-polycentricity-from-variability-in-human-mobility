Download and run the Jupyter notebook named 'Inferring_urban_polycentricity_from_the_variability_in_human_mobility_patterns.ipynb' in order to learn more about how far London and Seoul are from being monocentric cities.

This is done using smart travel card data from London's and Seoul's public transport network.

A nucleus for each city is chosen. The code first tests whether the observed travelling behaviour agrees with the hypothesised travelling behaviour if this nucleus was the only centre for the city.

Then, departures from the hypothesised travelling behaviour corresponding to the monocentric city are analysed. This is done through the use of Poisson mixture models, which allow us to capture the variability in the data.

The analysis is perforemed for different choices of nucleus.

The Jupyter notebook automatically loads the data and the module named 'urban_structure_human_mobility.py'. No need to download them for the notebook to work. 
