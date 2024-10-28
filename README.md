# Hyperion is an Out-of-core GNN Training System with GPU-initiated Asynchronous Disk IO Stack
```
$ git clone https://github.com/RC4ML/Legion.git
$ git checkout Hyperion
```

## 1. Hardware 
### Hardware Recommended
Local bare-metal machine.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | SSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 768GB | PCIe 4.0x16| 80GB-PCIe-A100 | Intel P5510, Sansumg 980 pro |


## 2. Software 
We list our tested environment:
### CUDA Toolkits and NVIDIA Driver
1. Nvidia Driver Version: 515.43.04
2. CUDA 11.7 - 12.4
3. GCC/G++ 11.4.0
### OS 
4. OS: Ubuntu 22.04, Linux version 5.15.72 (customized)

5. pytorch-cu117 (to pytorch-cu124), torchmetrics
```
$ pip install torch-cuxxx
```
6. dgl 1.1.0 - 2.x
```
$ pip install dgl -f https://data.dgl.ai/wheels/cu1xx/repo.html
```
### GPU Direct Storage
We reuse the BaM (https://github.com/ZaidQureshi/bam) Kernel Module to enable GPU Direct Storage Access.
#### Disable IOMMU in Linux
$ cat /proc/cmdline | grep iommu
If either iommu=on or intel_iommu=on is found by grep, the IOMMU is enabled.
Disable it by removing iommu=on and intel_iommu=on from the CMDLINE variable in /etc/default/grub and then reconfiguring GRUB. The next time you reboot, the IOMMU will be disabled.

#### Compiling Nvidia Driver Kernel Symbols
```
$ cd /usr/src/nvidia-515.43.04/
$ sudo make
```
#### Building BaM Project
From the project root directory, do the following:
```
$ git clone https://github.com/ZaidQureshi/bam.git
$ cd bam
$ git submodule update --init --recursive
$ mkdir -p build; cd build
$ cmake ..
$ make libnvm                         # builds library
$ make benchmarks                     # builds benchmark program
$ cd build/module
$ make
```
#### Loading/Unloading the Kernel Module
Unbind the NVMe drivers according to your needs (customize unload_ssd.py):
```
$ sudo python /path/Hyperion/unload_ssd.py 
$ cd /path/BaM/build/module
$ sudo make load
```
Check whether it's successful
This should create a /dev/libnvm* device file for each controller that isn't bound to the NVMe driver.
```
$ ls /dev/
```
The module can be unloaded from the project root directory with the following:
```
$ cd build/module
$ sudo make unload
```

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
There are two steps to train a GNN model in Hyperion. In these steps, you need to change to **root** user for GPU Direct SSD Access.
### Step 1. Start Hyperion Server

```
$ python Hyperion_server.py --dataset_path 'dataset' --dataset_name ukunion --train_batch_size 8000 --fanout [25,10] --epoch 2 
```

### Step 2. Run Hyperion Training
After Hyperion outputs "System is ready for serving", then start training by: 
```
$ python training_backend/Hyperion_graphsage.py --class_num 2  --features_num 128 --hidden_dim 256 --hops_num 2 --epoch 2
```
