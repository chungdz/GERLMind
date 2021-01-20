mkdir data checkpoint result
cd data 
mkdir train valid test raw
cd ../
python -m data_prepocess.build_dict
python -m data_prepocess.find_neighbors
python -m data_prepocess.build_train --processes=10
python -m data_prepocess.build_valid --processes=10 --fsamples=valid/behaviors.small.tsv
python -m data_prepocess.build_test --processes=40
python -m data_prepocess.resplit --filenum 10 --processes 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --epoch=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --epoch=0 --filenum=40

python -m data_prepocess.find_neighbors
CUDA_VISIBLE_DEVICES=1,2,3,7 python training.py --gpus=4 --epoch=4

python -m data_prepocess.build_dict --title_len=30
python -m data_prepocess.find_neighbors --news=True
python -m data_prepocess.build_train --processes=10 --max_hist_length=50
python -m data_prepocess.resplit --filenum 10 --processes 4
python -m data_prepocess.build_valid --processes=10 --fsamples=valid/behaviors.small.tsv --max_hist_length=50
python -m data_prepocess.build_test --processes=40 --max_hist_length=50
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --epoch=10 --lr=0.0005 --batch_size=256
