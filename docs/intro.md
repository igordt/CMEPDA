(intro)=
## Introduction

This is the documentation page for the exam project of the course "Computational Methods for Experimental Physics and Data Analysis".

The authors of this project are Luca Callisti, Marco Carotta, Igor Di Tota.

The main purpose of this project is to implement a lossy compression, using flow-based generative models.
The realization is obtained using Affine Autoregressive Flows, an example of [Normalizing Flows](https://arxiv.org/abs/1912.02762). As an application of Normalizing Flows, it was also shown how new data can be generated from the Gaussian distributions into which the original data are mapped.

## Motivation

The goal of this project was to implement a machine learning algorithm to analyze some HEP data. This gave us a chance to learn how to use GitHub and try writing documentation.

Also, since this work was inspired by the development of the [Baler](https://arxiv.org/abs/2305.02283) tool, which achieves lossy compression using an autoencoder, the datasets used are the same, so a comparison could be made.

## Project Structure

In this project, compression with two different models was studied, so the files `original_model1.ipynb` and `original_model2.ipynb` were created.

The document `Description_of_the_project_Callisti_Carotta_DiTota` contains the more detailed explanation, along with the results obtained, and is available in this repo.

 ## Table of Contents

 Here is an automatically generated Tabel of Contents:

 ```{tableofcontents}
 ```

 [github]: https://github.com/readthedocs-examples/example-jupyter-book/ "GitHub source code repository for the example project"
 [tutorial]: https://docs.readthedocs.io/en/stable/tutorial/index.html "Official Read the Docs Tutorial"
 [jb-docs]: https://jupyterbook.org/en/stable/ "Official Jupyter Book documentation"
