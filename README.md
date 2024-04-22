# ATiCS_practical1
* First install the environment ATiCS_P1.yml
conda create –ATiCS_P1.yml

Activate the environment
conda activate ATiCS_P1

Run the training file (insert model of choice) after the model_type
PS. There are some optional arguments:
checkpoint_path:  to alter the path to save the best model checkpoint
--lr: to alter the learning rate
 --lr_threshold: to alter the learning rate threshold for early stopping
-epochs: to alter the number of epochs
If not explicitly defined, they'll be the defaultvalues from Conneau et al. (https://aclanthology.org/D17-1070.pdf)

python -u path_to_file/practical1.py baseline
python -u path_to_file/practical1.py udlstm
python -u path_to_file/practical1.py bilstm
python -u path_to_file/practical1.py bilstm-max





* When working on snellius, first put files on snellius
rsync -av source scur___@snellius.surf.nl:~/destination

Then run the installation of the environment file*
sbatch install_environment.job

Activate the environment
conda activate ATiCS_P1

Alter the run_P1.job file to run the model_type of your choice (see options below)
PS. There are some optional arguments:
checkpoint_path:  to alter the path to save the best model checkpoint
--lr: to alter the learning rate
 --lr_threshold: to alter the learning rate threshold for early stopping
-epochs: to alter the number of epochs
If not explicitly defined, they'll be the defaultvalues from Conneau et al. (https://aclanthology.org/D17-1070.pdf)
srun python -u path_to_file/practical1.py baseline
srun python -u path_to_file/practical1.py udlstm
srun python -u path_to_file/practical1.py bilstm
srun python -u path_to_file/practical1.py bilstm-max

Run the training file
sbatch run_P1.job





* If after installing the environment we get the following error:

Traceback (most recent call last):
File "practical1.py", line 16, in <module>
train_dataset = load_dataset('stanfordnlp/snli', split='train')
File "/home/scur0234/.conda/envs/ATiCS_P1/lib/python3.8/site-packages/da$
ds = builder_instance.as_dataset(split=split, verification_mode=verifi$
File "/home/scur0234/.conda/envs/ATiCS_P1/lib/python3.8/site-packages/da$
raise NotImplementedError(f"Loading a dataset cached in a {type(self._$
NotImplementedError: Loading a dataset cached in a LocalFileSystem is not $
srun: error: gcn62: task 0: Exited with exit code 1
srun: Terminating StepId=5958687.0


Please update datasets:
1. conda activate AtiCS_P1
2. pip install -U datasets
or
2. conda install -c huggingface -n ATiCS_P1 datasets




