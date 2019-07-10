Some assembly required.

To run the transformer pre-processing,

1. https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators
2. At commit e97c71902ceb63b7b7e0ef821d5ab1e2ed45b014
3. With the included patch files applied
4. With the following command:
t2t_datagen --problem=translate_ende_wmt32k_packed --data_dir=/path/to/data


Notice: the patch files were generated from a diff of internal code. The github
contains externalized code; the proccess to externalize code involves string
transformerations and subsitutions which results in the patch files being
non-trivial to apply to the github version. :(


