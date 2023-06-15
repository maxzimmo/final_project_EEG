#   .py Files needed to run the model:
#   get_data (incl.Internal_funcs)
#   Preproc
#   RNN_Model
#   processflow

# RNN Preproc process:

##  1) Fit the scaler on fullDF()
##  1b Fit the Scaler on the full dataset. MinMax scaler 
##  1c Fit the PCA
##  2) Transform the scaler
### 2a rnn_df(). Returns a ist with 16 np.arrays (13-74,310)
### 2b Transform each set  with the previously fitted scaler 
##  3) Padding
### 3a Apply padding() to Pad the arrays to (74,310) shape ok to RNN
##  4) y-Preproc
### 4a run y_unique() ndarray (720, 1) with a single value for each trial 
### 4b OHE hot-encode the y: y-values need to be one-hot-encoded for the RNN.


# Data analysis
- Document here the project: dummy_project
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for dummy_project in github.com/{group}. If your project is not set please add it:

Create a new project on github.com/{group}/dummy_project
Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "dummy_project"
git remote add origin git@github.com:{group}/dummy_project.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
dummy_project-run
```

# Install

Go to `https://github.com/{group}/dummy_project` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/dummy_project.git
cd dummy_project
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
dummy_project-run
```
