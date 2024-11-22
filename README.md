

Welcome to the code repository for our research paper, Targeted Therapy in Data Removal: Object Unlearning Based on Scene Graphs.



## Setup

We have tested it on Ubuntu 22.04.4 LTS with Python 3.7.4 and PyTorch 1.3.1+cu117.

### Setup code
You can setup a conda environment to run the code like this:


create a conda environment and install the requirments
```conda create --name ousg python=3.7 --file requirements.txt 
conda activate ousg        # activate virtual environment ```
install pytorch and cuda version as tested in our work
```conda install pytorch==1.3.1 torchvision==0.4.0 cudatoolkit=11.7 -c pytorch```
more installations
```pip install opencv-python tensorboardx grave addict```


### Setup Visual Genome data
(instructions from the <a href="https://github.com/google/sg2im"> sg2im </a> repository)

Download and unpack the relevant parts of the Visual Genome dataset:

```bash
bash scripts/download_vg.sh
```


Preprocess data:

```bash
python scripts/preprocess_vg.py
```

This will create files `train.h5`, `val.h5`, `test.h5`, and `vocab.json` in the directory `datasets/vg`.


## Training

Train the model (i.e., develop original model):
```
python train_ft.py
``` 


Please set the dataset path `DATA_DIR` in `train_ft.py` before running.


## Unlearning and Verification

To develop fine-tuned model:
```
python unlearn_finetune_{method_name}.py
``` 

To develop redacted-based model (e.g., Obj_IF) and evaluate the methods:
```
python unlearn_ft.py
``` 

Please set the checkpoint path (in command-line arguments) in `unlearn_finetune_{method_name}.py` and `unlearning_ft.py` before running. The default path is ```<exp_dir>/<experiment>_model.pt```.




## Acknoledgement

The data preprocessing and SG2I components are based on the <a href="https://github.com/he-dhamo/simsg"> sg2im repository. For detailed information about scene graph data processing, the SG2I model, and its underlying principles, please refer to this repository. 

