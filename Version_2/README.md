# Next Activity Prediction
Materials for Next Activity Prediction through Graph Neural Networks

## Requirements
- Python 3.10 (mandatory)
- Libraries, included in requirements.txt

For install, move to ``nextActivityPrediction`` directory, then run: <br>
``pip install -r requirements.txt``

## How it works (brief explaination to improve)
Please note that code names must be updated, this section is exclusively to illustrate the methodology of our approach.<br>
- BIG.py:<br>
<img src="FlowChart/BIG.png" width=30% height=50%><br>
- DATASET.py:<br>
<img src="FlowChart/DATASET.png" width=30% height=50%><br>
- TRAINING.py:<br>
<img src="FlowChart/TRAINING.png" width=30% height=50%><br>

## How to run
- To create instance graphs, run:<br>
``python BIG.py``

- To create state graphs for dataset enrichment with subgraphs, run:<br>
``python STATE.py``

- To preprocess instance graphs for DGCNN, run:<br>
``python DATASET.py``

- To train the DGCNN, run:<br>
``python TRAINING.py``

## Results (necessary?)
