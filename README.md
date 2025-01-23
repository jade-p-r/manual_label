# Rationale
Physiological signals are often divided in two categories : 
- High frequency waveforms (100-500Hz)
- Low frequency numerics (1 Hz)

Getting a glance at some fraction of the data often allows for better understanding of the potential artefacts.
This script was designed to quickly label a fraction of the dataset of interest and save it locally

Labelling tool to manually label some time series and numerics 
# How to 
Place your data as a folder with patient level parquet files.

Run main.py within the directory.
The data gets saved as a .npz file locally