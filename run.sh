mkdir data checkpoint result
cd data 
mkdir train valid test raw
python -m data_prepocess.build_dict
python -m data_prepocess.find_neighbors
python -m data_prepocess.build_train --processes=10
python build_valid.py --processes=10 --max_hist_length 30 --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40 --max_hist_length 30
