# Data analysis
- Neural network analysing EEG brain data
- Description: In this project, we built a recurrent neural network which analyzes patterns in EEG brain data in order to determine emotional states of the participants.
- Data Source: SEED Dataset V. Provided from the Department of Computer Science at Shanghai Jiao Tong University
- Type of analysis: Recurrent Neural Network, Support Vector Machines (Base Model)

# Install

Go to `https://github.com/maxzimmo/final_project_EEG/` to see the project, manage issues and get further information.

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone https://github.com/maxzimmo/final_project_EEG/ <new_directory>
cd <new_directory>
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
new-directory-run
```
# Data procurement
In order to run use the application, it is necessary to obtain EEG brain data as npy files. As mentioned earlier, we obtained the EEG brain data from Shanghai Jiao Tong University.

# Run project
- In order to run the project, it is possible to connect the repository to a streamlit webpage. The app.py file within the repository offers a detailed application, on which it is possible to upload npy files in order to analyze the given brain data for emotional states. In order to do so, simply connect the streamlit webpage with the app.py file.
- A second possibility is to take advantage of the scripts within the script folder, which preprocess, model and predict.
