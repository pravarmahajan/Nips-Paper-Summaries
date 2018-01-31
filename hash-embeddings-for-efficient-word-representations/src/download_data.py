import dataloader

choices = ["agnews", "dbpedia", "yelp_pol", "yelp_full", "yahoo",
           "amazon_full", "amazon_pol"]
for choice in choices:
    dl_obj = dataloader.UniversalArticleDatasetProvider(choice)
    dl_obj.prepare()
