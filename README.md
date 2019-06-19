# Assignment 3

## Participants
1.
2.
3.
4.


## Description

The project is splitted into two main parts. The one is the jupyter notebook that contains
the models formations and initializations, it lives under the */notebooks* folder. The other
one lives under */app* and contains resusable code for the data preprocessing the analysis
of the models and the visualization of their performance.

All the reusable data sources or execution logs lives under the 
*/data* folder 

## Setup Instructions

   1. Create or load an existing conda environment
   2. Install the packages from the *requiremets.txt* file
   3. Download the vocabulary that will be used in the project and place it under the */data* folder [Download here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)
        
        #### Linux environments
        * !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
        * !gzip -d cc.en.300.bin.gz
   4. Download the **stackoverflow-posts** dataset and place i under the */data* folder. [Download dataset here](https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv)
   
## Runtime Instructions

   1.