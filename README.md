# CS231_Assignments_2018
Here are my answers for Stanford cs231 assignments 2018 version http://cs231n.stanford.edu/.

I have also included my answers to inline questions based on my understanding, welcome any discussion on those questions.

Some notes for setting up google cloud environment for cs231n 2018-2019:
1) General update compare to cs231n 2017-2018:
  -previous imaging procedure has been digarded; 
  -quota scheme introduced by google cloud to restrict usage of GPU/CPU;
  
2) key setting-up steps:
  - setup vm machine (with following installed)
    ~TensorFlow 1.13 (Intel optimized ...) to PyTorch 1.0 + fast.ai
    ~Install NVIDIA GPU driver automatically on first startup?
    ~Enable access to JupyterLab via URL instead of SSH. (Beta)
  - Configure networking
  - Access gcloud VM
  - gcloud setup script
  - Getting a Static IP Address to run Jupyter on local browser (important!)
  
3) Start
  - ssh to gcloud VM (copy and paste the gcloud command from the VM page)
  - transfer scripts (git clone)
  - transfer dataset (cd cs231n/datasets     ./get_datasets.sh)
  - ssh into VM, cd to assignment folder, run jupyter notebook
