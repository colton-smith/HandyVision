# HandyVision
ECE 8410 - Computer Vision Project

Colton Smith  
Devin McLoughlin  
Jillian Power  
Matt Crewe  

# Requirements
- Python 3.8+

# Project Structure Overview
```
.
├── lib: handyvision python package     | handyvision python package
├── README.md                           | Information
├── .gitignore                          | Tell git, "don't track transient generated files"
├── requirements.txt                    | Python package dependencies
└── setup.sh                            | Install dependencies
```

# How to Run 
After performing python setup below:
```
python game.py
```

# Python Setup 
Create a python virtual environment. This will seperte all the packages in this project from system packages. We will track all package dependencies in `requirements.txt` so that we can easily install all required packages and avoid a breaking change in a package version change from breaking our application. This way, we know exactly what combination of packages and versions our application is known to run on.    

## Unix
---
Create a virtual environment: 
```
python3 -m venv .venv
```

Activate environment (you will need to do this in every shell, or your system python will be used instead)
```
source .venv/bin/activate
```

Should see something like (depends on your terminal config):
```
(.venv) ~/dev/HandyVision (master) $
```

Now, you can install python dependencies:
```
pip install -r requirements.txt
```

Also, we keep our python code in our own package to solve the problem of terrible horrible relative imports in python. This allows us to define submodules for various subsystems, and just general logical seperation of code. The `-e` flag enables you to install a local package in **development mode**, meaning you can make changes to the package without having to re-install to see the changes.  
```
cd lib 
pip install -e .
```

You can verify you have the package installed (make sure you have the environment activated)
```
pip freeze | grep handyvision
>> -e git+ssh://git@github.com/colton-smith/HandyVision.git... blah blah blah
```

Finally try and import the handyvision.confirm_install subpackage from a python interpreter.
```
python3 
>>> from handyvision import confirm_install
EXAMPLE PACKAGE IMPORTED - INSTALLATION VALID
```

Now you can go ahead and import anything from handyvision library, perhaps like:
```
import handyvision as hv
```
