from final_project_EEG.
from final_project_EEG import Preproc

# architecture of the RNN 
from tensorflow.keras.layers import Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW, Nadam
from tensorflow.keras import layers
from tensorflow.keras.metrics import Recall
from tensorflow.keras.losses import CategoricalHinge