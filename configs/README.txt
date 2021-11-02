This folder contains the config files
used for different experiment setups.

For more information on each folder, look
at comments on files and corresponding
templates.

basic/: 
-------
Standard experiments for training unrolled 
DL-recon on mridata.org knee dataset


dl-recon-subsampled/: 
---------------------
Experiments regarding training standard dl-recon 
framework with subsampled amounts of data.
In this framework, it is unclear how to handle
data without fully sampled references 
(i.e. ground-truth data). As a result, these
all scans without reference data are discarded.


tests/: 
---------------------
Configs for basic tests.