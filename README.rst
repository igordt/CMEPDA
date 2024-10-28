======
CMEPDA
======
.. image:: https://readthedocs.org/projects/cmepda-callisticarottaditota/badge/?version=latest
    :target: https://cmepda-callisticarottaditota.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


**Repository for the Computing Methods for Experimental Physics and Data Analysis course exam project.**

Luca Callisti, Marco Carotta, Igor Di Tota

Introduction
=================
The main purpose of this project is to implement a lossy compression, using flow-based generative models.
The realization is obtained using Affine Autoregressive Flows, an example of `Normalizing Flows <https://arxiv.org/abs/1912.02762>`_. As an application of Normalizing Flows, it was also shown how new data can be generated from the Gaussian distributions into which the original data are mapped.

Motivation
=================
The goal of this project was to implement a machine learning algorithm to analyze some HEP data. This gave us a chance to learn how to use GitHub and try writing documentation.

Also, since this work was inspired by the development of the `Baler <https://arxiv.org/abs/2305.02283>`_ tool, which achieves lossy compression using an autoencoder, the datasets used are the same, so a comparison could be made.

Project Structure
=================
In this project, compression with two different models was studied, so the files ``original_model1.ipynb`` and ``original_model2.ipynb`` were created.

The document ``Description_of_the_project_Callisti_Carotta_DiTota`` contains the more detailed explanation, along with the results obtained, and is available in this repo.
