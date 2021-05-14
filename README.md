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

This program provides the implementation of our QGNN as described in [our paper](https://arxiv.org/abs/2008.05089), where we propose to learn embeddings for nodes and graphs within the Quaternion space, hence improving the embedding quality.

<p align="center">
	<img src="https://github.com/daiquocnguyen/QGNN/blob/master/qgnn.png" width="550">
</p>

## Usage

### News
- December 2020: Update (Pytorch) QGNN layer (v2) for other downstream tasks.
- November 2020: The extended abstract of our paper has been accepted to the NeurIPS 2020 Workshop on Differential Geometry meets Deep Learning (DiffGeo4DL).
- September 2020: [A new blog](https://daiquocnguyen.github.io/blog/quaternion-graph-neural-networks) on Quaternion Graph Neural Networks.

### Requirements
- Python 	3.7
- Networkx 	2.3
- Scipy		1.3
- Tensorflow 	1.14 or
- Pytorch 	1.5.0 & CUDA 10.1

### Training

Regarding node classification:

	QGNN$ python train_node_cls.py --dataset cora --learning_rate 0.05 --hidden_size 16 --epochs 100 --fold 2

	QGNN$ python train_node_cls.py --dataset citeseer --learning_rate 0.05 --hidden_size 16 --epochs 100 --fold 4
	
	QGNN$ python train_node_cls.py --dataset pubmed --learning_rate 0.01 --hidden_size 64 --epochs 200 --fold 6
	
Regarding graph classification:

	QGNN$ python train_graph_Sup.py --dataset IMDBBINARY --batch_size 4 --hidden_size 128 --fold_idx 2 --num_epochs 100 --num_GNN_layers 2 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_hs128_fold2_k2_1

	QGNN$ python train_graph_Sup.py --dataset DD --batch_size 4 --hidden_size 256 --fold_idx 5 --num_epochs 100 --num_GNN_layers 3 --learning_rate 0.0005 --model_name DD_bs4_hs256_fold5_k3_1

## Cite  
Please cite the paper whenever QGNN is used to produce published results or incorporated into other software:

	@article{Nguyen2020QGNN,
		author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		title={Quaternion Graph Neural Networks},
		journal={NeurIPS 2020 Workshop on Differential Geometry meets Deep Learning. arXiv preprint arXiv:2008.05089},
		year={2020}
	}

## License
As a free open-source implementation, QGNN is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

QGNN is licensed under the MIT License.
