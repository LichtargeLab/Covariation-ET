# Evolutionary Trace and Covariation Prediction

This project re-implements in Python3 a large existing code base built for the Evolutionary Trace algorithm (see [citations](##References)) in C with a significant amount of documentation. This new project focuses on Evolutionary Trace in the context of covariation predictions and makes different distance matrice, tree construction method, and many more metrics of covariation prediction available. The project seeks to make this code base accessible in Python as well as increase hardware usage/throughput through the use of vectorization and multiprocessing.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

There is one main pre-requisites for using this code: Anaconda

1. Anaconda
    * To install Anaconda (on Ubuntu 18.04) please follow the link from DigitalOcean in the [Acknowledgements](##Acknowledgements).

*Caveats* If you are installing this code base on a different version of Ubuntu or a different operating system altogether these steps will not be the same.

### Installing

1. Create the Anaconda environment for this project by navigating to the project directory and running:
    ```
    conda env create -f environment.yml
    ```
    *Caveats* * Using Anaconda environments allows for greater portability, but some operating systems use different channels meaning it may not be possible to set up the environment as described in this step, consider using other channels or looking for different versions of the same packages.
2. Activate the environment to use the code in this package.
    ```
    conda activate PyET
    ```
3. (Optional) If using the DCAWrapper.py module Julia is needed as well. This has caused issues when included in the base environment, it can be installed by running the command:
    ```
    conda install -c conda-forge julia=1.0.3
    ```
   with the PyET environment activated.
### Usage
To use the faster version of CovET, first clone this "faster" branch, then reinstall the PyET environment. The faster version is only implemented for CovET. The faster version runs about 5 times faster and doesn't produce large intermediate files. But due to the usage of R and C++, there might be issues during the environment setup.

To use this code please navigate to the 'src' directory and use the following command:
```
python CovET.py --query <Sequence identifier for the target sequence.> --alignment <Path to a fasta alignment> --output_dir <Directory to which results will be written.> --processors <How many processors are available to this tool while making predictions.>
```

## Authors
* **Daniel Konecki** - *Python Libraries and adaptation of C code base. Further development of ET-MIp method. Optimization of algorithm implementation and evaluation. Development of new ET covariation method penalizing transitions in pairs of columns which do not represent concerted variation or conservation.* - [dkonecki](https://github.com/dkonecki)
* **Spencer Hamrick** - *Further development of the CovET method. Current maintainer of the Covariation-ET repository.*
* **Chen Wang** - *Optimization of the CovET work flow to speed up the calculation. Current maintainer of the Covariation-ET repository.*
* **Benu Atri** - *Original work on ET-MIp continuation, identification of the fact that a limited traversal could achieve the same, and sometimes better covariation predictions*
* **Jonathan Gallion** - *Assessment of covariation by structural clustering and enrichment methods and frequent discussion of important factors for the improvement of covariation prediction.*
* **Angela Wilkins** - *Original developer of the ET-MIp method, who made original suggestion on how it might be possible to improve the method.*

## References
1. [An evolutionary trace method defines binding surfaces common to protein families.](https://www.ncbi.nlm.nih.gov/pubmed/8609628)
2. [A family of evolution-entropy hybrid methods for ranking protein residues by importance.](https://www.ncbi.nlm.nih.gov/pubmed/15037084)
3. [Evolutionary and Structural Feedback on Selection of Sequences for Comparative Analysis of Proteins](https://www.ncbi.nlm.nih.gov/pubmed/16397893)
4. [Accounting for epistatic interactions improves the functional analysis of protein structures](https://www.ncbi.nlm.nih.gov/pubmed/24021383)
5. [Combining Inference from Evolution and Geometric Probability in Protein Structure Evaluation](https://www.ncbi.nlm.nih.gov/pubmed/12875851)
6. [Sequence and structure continuity of evolutionary importance improves protein functional site discovery and annotation](https://www.ncbi.nlm.nih.gov/pubmed/20506260)

## License
MIT license

## Acknowledgments

Thanks to:
* **Angela Wilkins** - *Original development of C libraries adopted in this project. Additional development of the ET (pair interaction ET) method and methods for evaluating residue ranking (structural clustering weighting z-score).*
* **Benu Atri** - *Additional development of the ET (pair interaction ET) method and contributions to ET-MIp development.*
* **Rhonald Lua** - *Additional development of the ET (real valued ET) method and methods for evaluating residue ranking (structural clustering weighting z-score).*
* **David Marciano** - *Additional development of the ET (pair interaction ET) method.*
* **Eric Venner** - *Additional development of the ET (pair interaction ET) method.*
* **Serkan Erdin** - *Additional development of the ET (pair interaction ET) method and methods for evaluating residue ranking (structural clustering weighting z-score).*
* **Ivana Mihalek** - *Additional development of the ET (real valued ET) method and methods for evaluating residue ranking (structural clustering weighting z-score).*
* **Ivica Res** - *Additional development of the ET (real valued ET) method and methods for evaluating residue ranking (structural clustering weighting z-score).*
* **Hui Yao** - *Development of methods for evaluating residue ranking (structural clustering weighting z-score)*
* **R. Mathew Ward** - *Development of methods for evaluating residue ranking (structural clustering weighting z-score)*
* **Olivier Lichtarge** - *Creator of original Evolutionary Trace (integer valued ET) method*
* [theskumar](https://github.com/theskumar/python-dotenv) for creating and maintaining the python-dotenv package
* [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for the tools to manage environments like the one in this project.
* Sphinx and Google for their description [stylesheet](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
* [Billie Thompson](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) for this readme template.
* [DigitalOcean](https://www.digitalocean.com) for tutorials on:
    * [Installing Anaconda3](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)