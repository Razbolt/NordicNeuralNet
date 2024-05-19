                       TRANSLATION FROM ENGLISH TO SWEDISH

Instructions file for INM706 Translation Project for Erdem Baha Arslan and Grigorios Vaitsas
Github Repository: https://github.com/Razbolt/NordicNeuralNet
 
#Structure of the project folder
The folder contains the following files:

setup_hyperion.sh --> Sets up a pyenv environment in Hyperion and installs all the required packages from requirements.txt

requirements.txt  --> Requirements file with list of required packages

INM706_Inference-2.ipynb  --> Jupyter notebook that can be used for inference testing
instructions on how to run this are included in the Notebook itself

The project also contains 4 folders:

oldcode/ --> Contains the code of our initial attempts where we created the vocabularies and did the tokenization ourselves **For reviewing purposes only, code does not run

seq2seq/ --> Contains our sequence to sequence model files
transformer/ --> contains the transformer model files
t5model/ --> contains the t5 model files

**Important Before training please add your wandb API key to the location in runjob.sh file:
export WANDB_API_KEY=

To start training of any of the models one needs to use the runjob.sh file as:
>sbatch runjob.sh

All 3 models contain a data/ folder where we have included a small sample dataset with:
english_100k_clean.txt
swedish_100k_clean.txt

that can be used to test that the training starts. 

The full dataset can be downloaded from this loacation:
https://www.statmt.org/europarl/
by clicking on the link called:
parallel corpus Swedish-English, 171 MB, 01/1997-11/2011





