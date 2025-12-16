# Dual-Qubit Hierarchical Fuzzy Neural Network: Enabling Relational Learning for Image Classification via Quantum Entanglement

This project implements various quantum-classical hybrid neural network models for image classification tasks.

## Acknowledgments

This project is inspired by and builds upon [QA-HFNN](https://github.com/wodaka/QA-HFNN) by Shengyao Wu et al. We thank the original authors for their pioneering work on quantum-assisted hierarchical fuzzy neural networks.

Original paper:
```
Wu, S., Li, R., Song, Y., Qin, S., Wen, Q., & Gao, F. (2024). 
Quantum Assisted Hierarchical Fuzzy Neural Network for Image Classification. 
IEEE Transactions on Fuzzy Systems.
```

## Models

This repository includes implementations of the following models:

- **DQ-HFNN**: Dual-Qubit Hierarchical Fuzzy Neural Network
- **FDNN**: Fuzzy Deep Neural Network
- **QA-HFNN**: Quantum Assisted Hierarchical Fuzzy Neural Network
- **DNN**: Deep Neural Network baseline

Each model contains:
- Quantum circuit layers
- Fusion layers
- Classical classification layers
- Classical neural network layers

## Requirements

### Environment Setup

We provide a pre-configured conda environment. Simply import it using:
```bash
conda env create -f environment.yml
conda activate <env_name>
```
## Dataset

Due to copyright restrictions, datasets are not included in this repository. Please download the required datasets yourself. 

Data loaders are provided in the `data/` folder for easy integration.

## Usage

To run a specific model, simply execute the corresponding run script:
```bash
# For JAFFE
python run_jaffe.py

# For Dirty-MNIST
python run_dmnist.py

# For Scene15
python run_scene15.py

# For Fashion-MNIST
python run_fashionmnist.py

# For Cifar-10
python run_cifar10.py

```

Each script is pre-configured for its corresponding dataset.

### Model Selection

All model implementations are available in the `models/` folder. To use a specific model:

1. Navigate to the relevant model file
2. Uncomment the model you want to use
3. Run the corresponding `run_*.py` script

### Parameters

For complete parameter settings and experimental configurations, please refer to our paper.

## Results

Experimental results and performance metrics are detailed in our paper.
ArXiv. https://arxiv.org/abs/2512.13274
## Citation

If you find this repository useful in your research, please consider citing:
Zhang, W., Wang, J., Ye, T., & Liao, C. (2025). Dual-Qubit Hierarchical Fuzzy Neural Network for Image Classification: Enabling Relational Learning via Quantum Entanglement. 
ArXiv. https://arxiv.org/abs/2512.13274
```

```

## Contact

If you have any questions, feel free to contact:

**WenWei Zhang**  
Email: zhangwenwei9224@qq.com  
GitHub: [Comets9224](https://github.com/Comets9224)

Pull requests are highly welcomed!

## License

   Copyright [2025] [Wenwei Zhang]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.