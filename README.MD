Steps :-
1) Create a python environment. Run below command in terminal
    python -m venv venv

2) Activate the environment by running activate script.

3) Install requuired python packages using the requirement.txt file. Run below command
    pip install -r requirements.txt

4) Install jupter kernel for virtual environment
    ipython kernel install --user --name=venv

5) Open Notebook.ipynb and change the kernel to venv from default.

6) Run all cells.

7) Execute test_script.py, to test model with user input.
./venv/Scripts/python.exe test_script.py
