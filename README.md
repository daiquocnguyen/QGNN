<p align="center">
	<img src="https://github.com/daiquocnguyen/QGNN/blob/master/logo.png" width="100">
</p>

# Quaternion Graph Neural Networks<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FQGNN%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/QGNN"><a href="https://github.com/daiquocnguyen/QGNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/QGNN"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/QGNN">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/QGNN">
<a href="https://github.com/daiquocnguyen/QGNN/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/QGNN"></a>
<a href="https://github.com/daiquocnguyen/QGNN/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/QGNN"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/QGNN">

This program provides the implementation of our QGNN as described in [our paper](https://arxiv.org/pdf/2008.05089.pdf), where we propose to learn node and graph embeddings within the Quaternion space and introduce our quaternion graph neural networks (QGNN) to generalize GCNs within the Quaternion space.

<p align="center">
	<img src="https://github.com/daiquocnguyen/QGNN/blob/master/qgnn.png" width="550">
</p>

## Usage

### News
- November 02, 2020: The extended abstract of [our paper](https://arxiv.org/pdf/2008.05089.pdf) has been accepted to the NeurIPS 2020 Workshop on Differential Geometry meets Deep Learning (DiffGeo4DL).

### Requirements
- Python 	3.x
- Networkx 	2.3
- Scipy		1.3
- Tensorflow 	1.14 or
- Pytorch 	1.5.0

### Training

Regarding node classification:

	QGNN$ python train_node_cls.py --dataset cora --learning_rate 0.05 --hidden_size 16 --epochs 100 --fold 2

	QGNN$ python train_node_cls.py --dataset citeseer --learning_rate 0.05 --hidden_size 16 --epochs 100 --fold 4
	
	QGNN$ python train_node_cls.py --dataset pubmed --learning_rate 0.01 --hidden_size 64 --epochs 200 --fold 6
	
Regarding supervised graph classification:

	QGNN$ python train_graph_Sup.py --dataset IMDBBINARY --batch_size 4 --hidden_size 128 --fold_idx 2 --num_epochs 100 --num_GNN_layers 2 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_hs128_fold2_k2_1

	QGNN$ python train_graph_Sup.py --dataset DD --batch_size 4 --hidden_size 256 --fold_idx 5 --num_epochs 100 --num_GNN_layers 3 --learning_rate 0.0005 --model_name DD_bs4_hs256_fold5_k3_1

Regarding unsupervised graph classification:
	
	QGNN$ python train_graph_UnSup.py --dataset COLLAB --batch_size 4 --hidden_size 256 --num_epochs 100 --num_GNN_layers 4 --learning_rate 0.00005 --model_name COLLAB_bs4_hs256_fold0_k4_3

	QGNN$ python train_graph_UnSup.py --dataset DD --batch_size 4 --hidden_size 256 --num_epochs 100 --num_GNN_layers 2 --learning_rate 0.001 --model_name DD_bs4_hs256_fold0_k2_0

	QGNN$ python train_graph_UnSup.py --dataset IMDBBINARY --batch_size 4 --hidden_size 256 --num_epochs 100 --num_GNN_layers 2 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_hs256_fold0_k2_1

- See [Graph-Transformer](https://github.com/daiquocnguyen/Graph-Transformer) for more details about unsupervised learning. Regarding the Pytorch implementation for the unsupervised learning, you should have Cython 0.29.13 and Scikit-learn	0.21 and then change to the `log_uniform` directory to perform `make` to build `SampledSoftmax`, and then add the `log_uniform` directory to your PYTHONPATH.

## Cite  
Please cite the paper whenever QGNN is used to produce published results or incorporated into other software:

	@article{Nguyen2020QGNN,
		author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		title={Quaternion Graph Neural Networks},
		journal={arXiv preprint arXiv:2008.05089},
		year={2020}
	}

## License
As a free open-source implementation, QGNN is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

QGNN is licensed under the MIT License.
