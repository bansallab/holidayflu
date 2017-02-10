# Age-specific metapopulation model of disease spread 
This is the source code for the mathematical model described in the following paper:

[Ewing, Anne, Elizabeth C. Lee, Cecile Viboud, and Shweta Bansal. (2016). "Contact, travel, and transmission: The impact of winter holidays on influenza dynamics in the United States." The Journal of Infectious Diseases. doi:10.1093/infdis/jiw642](https://doi.org/10.1093/infdis/jiw642)

These Python scripts implement a Susceptible-Infected-Recovered deterministic mathematical model for influenza transmission across U.S. cities among children and adults. In addition to modeling disease transmission, temporary holiday-associated behavioral changes mimicking "school closure" and "travel" may be switched on/off in the model.

email: ecl48@georgetown.edu, shweta.bansal@georgetown.edu

Please cite the paper above if you use our code in any form or create a derivative work.

---
## File and Folder Descriptions
simulation_main_code.py
* This script includes functions to import school closure and travel parameters, import the non-holiday and holiday travel networks, and run the disease simulations.
* The bottom of the script includes the model main code. 

experiment_functions.py
* This script defines the temporary changes to the contact matrix that are associated with "school closure" interventions.

population_parameters.py
* This script imports the POLYMOD contact matrix from Germany, which includes small age bins, and aggregates the information into a 2x2 child and adult contact matrix, adjusted by the age structure of the U.S. population.

intervention_settings.py
* This script switches on/off holiday-associated behavioral changes and indicates which specific school closure or travel interventions should be used. See "Changing intervention settings" below.

model_inputs/contact_matrix_data/
* This folder includes POLYMOD contact matrix data ([Mossong, et al. 2008](http://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074)) and the Germany and U.S. population data to standardize the contact matrices.
* Please consult the README file and Supporting Information for additional details.

model_inputs/metro_travel_data/
* This folder includes undirected flight networks weighted by average number of passengers for the holiday and non-holiday periods. Flight data were obtained from the U.S. Bureau of Transportation Statistics ([data available here](http://www.transtats.bts.gov/)).
* Please consult the README file and Supporting Information for additional details.

---
## Usage
Keep the same folder structure as indicated in the repository and run in the Terminal:
```
$ python simulation_main_code.py
```

### Dependencies
Use of this model requires installation of the the following Python packages:
* NumPy ([download here](http://www.scipy.org/scipylib/download.html))
* NetworkX ([download here](https://networkx.github.io/))

Beginning Python users may choose alternatively to install the [Canopy](https://store.enthought.com/downloads/#default) or [Anaconda](https://www.continuum.io/downloads) Python distributions, which enable easy installation of these Python packages.

### Changing intervention settings
Users may change the holiday timing and holiday-associated behavioral changes by editing the values in intervention_settings.py. The settings currently implement the full holiday intervention (school closure and travel changes among both adults and children) during "typical" holiday timing relative to the influenza season. 

The following settings would run the models in the four simulations plotted in Figure 4:

```python
# baseline
experiment = 'no' # all other settings will be ignored

# travel
experiment = 'yes'
disease_intervention = 'none' 
travel_intervention = 'swap_networks' 
timing = 'actual' 

# school closure
experiment = 'yes'
disease_intervention = 'red_C_all' 
travel_intervention = 'none' 
timing = 'actual' 

# holiday
experiment = 'yes'
disease_intervention = 'red_C_all' 
travel_intervention = 'swap_networks' 
timing = 'actual' 
```

### Changing additional simulation parameters
Users may set additional simulation parameters by editing main code values in simulation_main_code.py:
* the transmission (beta) and recovery (gamma) parameters in the Susceptible-Infected-Recovered disease model (lines 698-699)
* the proportion of travelers in each state (Susceptible, Infected, or Recovered) (lines 702-704)
* the number of initial child and adult seeds for the simulation (lines 708-709)

---
## Output
The model outputs a CSV file of the following structure:

| metro_zero  | time_step | metro_id  | age | currently_infected | total_infected |
| ----------- | --------- | --------- | --- | ------------------ | -------------- |
| 1           | 0         | 1         | C   | 1                  | 1              |
| 1           | 0         | 1         | A   | 0                  | 0              |
| 1           | 0         | 2         | C   | 0                  | 0              |
| 1           | 0         | 2         | A   | 0                  | 0              |

Column descriptions:
* metro_zero: unique identifier for the metro area where the epidemic simulation was seeded
* time_step: simulation time step (epidemics are seeded in time_step 0)
* metro_id: unique identifier for the metro area whose infections are recorded in this row
* age: indicates whether the row is a record of infection for children (C) or adults (A)
* currently_infected: number of active infections in 'time_step' for 'metro_id' for 'age' in the simulation seeded in 'metro_zero'
* total_infected: cumulative number of infections by 'time_step' for 'metro_id' for 'age' in the simulation seeded in 'metro_zero'

Please note that a single model's output file will require ~2GB of storage space.
