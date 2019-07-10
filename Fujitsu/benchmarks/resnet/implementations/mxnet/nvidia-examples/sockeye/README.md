MxNet-to-ONNX exporter
==========================

What is this?
------------------

The following examples prepare the [WMT'15](http://www.statmt.org/wmt15/translation-task.html) German-English dataset for [neural machine translation (NMT)](https://en.wikipedia.org/wiki/Neural_machine_translation), and trains the [OpenNMT reference model](http://opennmt.net/Models/). 

Before you begin
----------------------------------------

**Note:** one of the preprocessing steps is documented as truncating the training set to save memory. Hence, the following lines in the `preprocess.sh` should be commented out during training until convergence:
```
head -n 200000 corpus.tc.BPE.de > corpus.tc.BPE.de.tmp && mv corpus.tc.BPE.de.tmp corpus.tc.BPE.de

head -n 200000 corpus.tc.BPE.en > corpus.tc.BPE.en.tmp && mv corpus.tc.BPE.en.tmp corpus.tc.BPE.en

```

The smaller dataset (with the above lines uncommented) can be used for performance benchmarking to save CPU RAM.

Execution
----------------------------------------

To download and preprocess the dataset, run
```
./preprocess.sh
```

To train the OpenNMT reference model, run
```
./train.sh
```

The training configuration can be changed in train.sh to adapt to your needs.
