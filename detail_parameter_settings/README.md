# Hyperion Technical Report
## Experimental Results

Please refer to tech_report.pdf

## Code Structure
To help users understand Hyperion's implementation, I list the source code structure in this part.
```
Hyperion\
├─Hyperion_server.py 
├─sampling_server\                                  ## codes of sampling server
└training_backend\                                  ## codes of training backend

Hyperion\sampling_server\src\
├─cache\                                            ## unified cache
├─engine\                                           ## pipelining engine of sampling server 
├─storage                                           ## graph/feature storage, system storage initialization
├─include                                           ## system configurations and hashmap (https://github.com/greg7mdp/parallel-hashmap)
├─main.cu                                           ## main function, will be replaced by a python extention module in the future
└Others

Hyperion\training_backend\
├─Hyperion_graphsage.py                               ## training backend for graphsage model
├─Hyperion_gcn.py                                     ## training backend for gcn model
├─setup.py                                          ## compiling the training backend
├─ipc_service.cpp ipc_service.h ipc_cuda_kernel.cu  ## inter process communication module for training backend with sampling server
└Others
```


