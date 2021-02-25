from preprocessing import preprocessing, create_bags

create_bags(10)
preprocessing('data/train.csv', 'data/train_vector', 'data/train_label')
preprocessing('data/dev.csv', 'data/dev_vector', 'data/dev_label')
