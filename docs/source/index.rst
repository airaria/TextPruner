.. TextPruner documentation master file, created by
   sphinx-quickstart on Thu Jul 22 17:00:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TextPruner's documentation
======================================

.. image:: ../../pics/banner.png
   :width: 500
   :align: center


**TextPruner** is a toolkit for pruning pre-trained transformer-based language models written in PyTorch. It offers structured training-free pruning methods and a user-friendly interface.

The main features of TexPruner include:

* **Compatibility**: TextPruner is compatible with different NLU pre-trained models. You can use it to prune your own models for various NLP tasks as long as they are built on the standard pre-trained models.
* **Usability**: TextPruner can be used as a package or a CLI tool. They are both easy to use.
* **Efficiency**: TextPruner reduces the model size in a simple and fast way. TextPruner uses structured training-free methods to prune models. It is much faster than distillation and other pruning methods that involve training. 

TextPruner currently supports the following pre-trained models in 
`transformers <https://github.com/huggingface/transformers>`_:

* BERT
* Albert
* Electra
* RoBERTa
* XLM-RoBERTa

Installation
------------

.. code-block:: bash
   
   pip install textpruner


.. note::

   This document is under development.

 
..
      Usage
      Examples
      How Does It Works


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   APIs/Configurations
   APIs/Pruners
   APIs/Utils

