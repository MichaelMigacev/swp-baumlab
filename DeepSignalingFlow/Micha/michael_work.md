# Introduction

As I have copied changed and written many files I will provide here a brief explanation of what I did where.

## Table of Contents

-   [Data](#data)
-   [Preprocessing](#preprocessing)
-   [Running The Model](#running-the-model)
-   [Running The Explanations](#running-the-explanations)
-   [Running The Visualizations](#running-the-visualizations)

## Data

Unfortunately it is impossible to reproduce some parts from the raw data so I downloaded their data from the github:
[DeepSignalingFlow](https://github.com/FuhaiLiAiLab/DeepSignalingFlow)

## Preprocessing

In the Preprocessing folder we can see 3 files:

preprocessing.ipynb

-   here we examine the raw data and look for duplicates in the filtered data
    create-data.ipynb
-   this looks at how our train and test splits are generated
    using_data.ipynb
-   here we look at how our data looks like in our data structure after preprocessing

## Running The Model

to run the model you can use the following command:

```bash
python geo_tmain_webgnn.py
```

I have modified the code to run for the specified fold for the nci-dataset.

## Running The Explanations

The GNN Explainer and Analyze Method was taken from Olha and was modified to work for the nci dataset.
analyze_model_nci.ipynb
explain_model_nci.ipynb
To replicate the analysis graphs from the paper I modified the functions from the paper.
For that first run:
webgnn_acc_analysis.py
webgnn_cell_dec_analysis.py
webgnn_edge_analysis.py
and then:
analysis_bindbi_net-c.py

## Running the Visualizations

To view the visualizations I used graphia. You can use their web-based interface to view the graphs.
[Graphia](https://graphia.app/)
All of them contain some custom merging to generate them so replicating that is not straightforward.
But:
gnn_viz.ipynb
contains all code to generate the tables used by graphia as well as some failed attempts to use plotly.

Additionally you can open the graph in "wiki-mode" in graphia to get automatic real time wikipedia search the moment you click on a node ^^
