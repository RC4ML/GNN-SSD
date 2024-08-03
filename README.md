# Hyperion is a Cost-efficient Out-of-core GNN Training System Based on Legion
```
$ git clone https://github.com/RC4ML/Legion.git
$ git checkout -b Hyperion
```

## 1. Hardware 
### Hardware Recommended
All platforms are bare-metal machines.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | SSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 768GB | PCIe 4.0x16| 80GB-PCIe-A100 | Intel P5510, Sansumg 980 pro |


## 2. Software 
Hyperion's software is light-weighted and portable. Here we list some tested environment.

1. Nvidia Driver Version: 515.43.04

2. CUDA 11.7

3. GCC/G++ 11.4.0

4. OS: Ubuntu(other linux systems are ok)

5. Intel PCM(according to OS version)
```
$ wget https://download.opensuse.org/repositories/home:/opcm/xUbuntu_18.04/amd64/pcm_0-0+651.1_amd64.deb
```
6. pytorch-cu117, torchmetrics
```
$ pip3 install torch-cu1xx
```
7. dgl 1.1.0
```
$ pip3 install  dgl -f https://data.dgl.ai/wheels/cu1xx/repo.html
```
8. MPI-3.1


## 3. Prepare Datasets 
Datasets are from OGB (https://ogb.stanford.edu/), Standford-snap (https://snap.stanford.edu/), and Webgraph (https://webgraph.di.unimi.it/).
Here is an example of preparing datasets for Hyperion.

### Uk-Union Datasets
Refer to README in dataset directory for more instructions
```
$ bash prepare_datasets.sh
```

## 4. Build Hyperion from Source

```
$ bash build.sh
```

## 4. Run Hyperion
There are three steps to train a GNN model in Hyperion. In these steps, you need to change to **root** user for GPU Direct SSD Access.
### Step 1. Open msr by root for PCM
```
$ modprobe msr
```
### Step 2. Start Hyperion Server

```
$ python Hyperion_server.py --dataset_path 'dataset' --dataset_name ukunion --train_batch_size 8000 --fanout [25,10] --gpu_number 2 --epoch 2 --cache_memory 38000000 
```

### Step 3. Run Hyperion Training
After Hyperion outputs "System is ready for serving", then start training by: 
```
$ python training_backend/Hyperion_graphsage.py --class_num 2  --features_num 128 --hidden_dim 256 --hops_num 2 --gpu_number 2 --epoch 2
```


