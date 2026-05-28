# Template Information

This template should be an exemplary structure for our replication study.
You can copy this folder when starting to work on your own project, and just template this file (*template_info.md*).

It consists of 10 items that are outlined below:

## Item 01: `./Readme.md`
This should be a readme file for the reviewer of the repository and reader of our reproducibility structure. 
It should outline the structure of the repository, and contain explanations on commands to install and run the replication study. 

## Item 02: `./requirements.txt`
This file should contain the Python packages that are required for the replication (in case Python is used for the replication study).
Generally, Python environments can be easily setup based on this `requirements.txt` using:
```
pip install -r requirements.txt
```

## Item 03: `./0_original_papers/`
This folder should contain the original paper(s) that we aim to replicate.
Furthermore, you can also upload an annotated PDF version or further documents, notes and comments you created when reading the original paper(s). 

## Item 04: `./1_original_repository/`
This folder should contain the original code / data/ repository, if provided by the authors.

## Item 05: `./1_code_produce/`
This folder should contain script(s) that produce some replication result data into the folder `2_data_produced`. It might be, that some source data is required for that, that should be loaded from `1_data_source`.

## Item 06: `./1_data_source/`
This folder should contain data that is used by the script(s) in `1_code_produce`, if required for the replication study.

## Item 07: `./2_data_produced/`
This folder should contain the results (replicated data, log files) produced by the script(s) in `1_code_produce`.

## Item 08: `./3_code_visualization/`
This folder should contain script(s) that analyse the produced, replicated data from `2_data_produced`, in order to produce, numbers, figures, tables, that were claimed / presented in the original papers. The results of this script(s) should be stored in `3_data_visualization`.

## Item 09: `./3_data_visualization/`
This folder should contain the produced numbers, figures, tables, that were claimed / presented in the original papers.

## Item 10: `./4_others/`
Furthermore, you might want to add further material, such as scripts that systematically compare the produced / replicated results with the original results.