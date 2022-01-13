
# e4750_2021Fall-Project-SSAA-ala2197-sls2305
Implementing Parallel Non-Negative Matrix Factorization

<img width="909" alt="Screen Shot 2021-12-19 at 5 23 39 PM" src="https://user-images.githubusercontent.com/50375261/146692944-891bd196-1524-46a3-9f33-d70461856329.png">

Non-Negative matrix factorization (NMF) is an unsupervised machine learning technique often used in recommendation systems.

In this project we investigate and compare various serial and parallel implementations of NMF. See `E4750_2021Fall_SSAA_sls2305_ala2197.report.pdf` for a detailed report of the project. 

### Repository Index

`data` - dataset files used for implementing NMF including the NYT dataset </br>
`dev` - unpolished code that was used for developing our NMF implementations on the way to the final versions in `src` </br>
`figures` - plots generated in `Plotting.ipynb` and `Demonstration.ipynb` get saved here </br>
`kernels` - CUDA (.cu) kernel files </br>
`src` - final polished versions of our NMF implementations and various helper functions. Code contains extensive comments and documentation. </br>
`Demonstration.ipynb` - Jupyter notebook to demonstrate how to use our code </br>
`Measurement.ipynb` - Jupyter notebook for measuring execution times of NMF implementations </br>
`Plotting.ipynb` - Jupyter notebook for generating plots from measurements </br>
`execution_times.csv` - grid execution parameters and runtimes </br>
