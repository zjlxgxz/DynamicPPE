# DynamicPPE
Code for the KDD21 paper "Subset Node Representation Learning over Large Dynamic Graphs"

Here is the instruction of using our code and datasets:
1. to compile DynamicPPE, go to code folder and run the following command:
g++ -std=c++0x -O3 -g -Wall -o dynamic-ppe-fast dynamic_ppe_fast.cpp MurmurHash3.cpp -lpthread
2. to build enwiki20 dataset, go to code folder and use the following python script:
build_wiki_graph.py
3. academic-small is from: https://github.com/luckiezhou/DynamicTriad
4. Execute:

```
/dynamic-ppe-fast datasets/academic-small/config_t_9_d_512/ 512 0.1 0.15 hash 0 20 0
```

# Data 
Download all datasets from: https://www.dropbox.com/sh/g3i95yttpjhgm2l/AAD8pF0XtgFv0fzmTrrOO4BWa?dl=0

# Python Version 
In-progress 
