mkdir data checkpoint result
cd data 
mkdir train valid test raw
python -m data_prepocess.build_dict
python -m data_prepocess.find_neighbors
python -m data_prepocess.build_train.py --processes=10
