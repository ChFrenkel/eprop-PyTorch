# Eligibility propagation (e-prop) - PyTorch-based implementation

> *Copyright (C) 2020-2022 University of Zurich*

> *The source code is free: you can redistribute it and/or modify it under the terms of the Apache v2.0 license.*

> *The software and materials distributed under this license are provided in the hope that it will be useful on an **'as is' basis, without warranties or conditions of any kind, either expressed or implied; without even the implied warranty of merchantability or fitness for a particular purpose**. See the Apache v2.0 license for more details.*

> *You should have received a copy of the Apache v2.0 license along with the source code files (see [LICENSE](LICENSE) file). If not, see <https://www.apache.org/licenses/LICENSE-2.0>.*


The provided source files contain the PyTorch-based code for training spiking recurrent neural networks (RNNs) with the eligibility propagation (e-prop) algorithm, as per the original paper:

> [G. Bellec et al., "A solution to the learning dilemma for recurrent networks of spiking neurons," *Nature communications*, vol. 11, no. 3625, 2020]

This code demonstrates e-prop on the cue accumulation task described in the e-prop paper by Bellec et al. The e-prop implementation provided here only covers the leaky integrate-and-fire (LIF) neuron model, which is presented by Bellec et al. as not able to learn the second-long time dependencies of the cue accumulation task. We solve this issue by using a second-long leakage time constant. We introduce this technique in the following paper:

> [C. Frenkel and G. Indiveri, "ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network processor enabling on-chip learning over second-long timescales," *IEEE International Solid-State Circuits Conference (ISSCC)*, 2022]

Instructions on how to use the code are available in the [main.py](main.py) source file. Parts of the code linked to dataset generation and network visualization were adapted from the original TensorFlow e-prop implemation from TU Graz, available [here](https://github.com/IGITUGraz/eligibility_propagation).

## Citation

* Upon usage of LIF neurons with increased leakage time constants to solve tasks with temporal dependencies from hundreds of ms to several seconds, please cite our ISSCC 2022 paper listed above.
* Upon usage of our PyTorch code implementing e-prop, please cite this repository (BibTeX/APA citation formats auto-converted from the CITATION.cff file are available through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/).
