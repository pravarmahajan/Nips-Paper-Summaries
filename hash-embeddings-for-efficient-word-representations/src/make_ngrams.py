import pickle
import dataloader

dl_obj = dataloader.UniversalArticleDatasetProvider(1, valid_fraction=0.05)
dl_obj.load_data()
