# Env settings
```
conda create -n adalora python=3.9
conda activate adalora

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install pip
pip install -r requirements.txt
```

# Transformer setting
copy&paste `src/transformer/trainer.py` file into `anaconda3/envs/lib/python3.9/site-packages/transformers/trainer.py`

# Command
```
cd GLUE
bash scripts/adalora_train.sh
bash scripts/adalora_eval.sh
```