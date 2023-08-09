# Installation

If you are going to work with python packages (for example Dosepy), it is a best practice to create virtual environments for your different applications, to avoid python-packages dependency conflicts. To do that, first we need to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (a python package manager). Once installed, open *Anaconda Prompt* (a black window) and write the following commands:


```python
conda create -n myenv python=3.10
conda activate myenv 
pip install dosepy
```

1. The first line is used to create an environment named "myenv", with a specific version of Python: 3.10
2. The second line activates the created environment
3. The last one to install dosepy

For more information about environments, [see this guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)

## Open Dosepy

To open Dosepy first start a python interpreter, for example running the python command:

```python
python
```

Once you see the symbol >>>, run the next command:

```python
>>> import Dosepy.GUI
```

The main window should be displayed

![Portada_Dosepy](../assets/main_window.PNG)
