# Setup the environmental variables for English to Swedish Translation coursework
# and load the requirements file

source /opt/flight/etc/setup.sh
flight env activate gridware
module add gnu
pyenv virtualenv 3.9.5 translate
echo translate > TRANSLATE/.python-version
cd TRANSLATE
which python
python --version
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt

