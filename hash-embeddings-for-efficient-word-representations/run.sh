# Without Dictionary, Shallow Network, Hash Embedding (Sec 5.3)

#python src/main.py -ds agnews -K 10000000 -k 2 -B 1000000 -d 20 -H 0 -e 10 -b 1024 -g -emb hash

#Without Dictionary, Shallow Network, Std. Embedding (Sec 5.3)
#python src/main.py -ds agnews -K 10000000 -k 2 -B 1000000 -d 20 -H 0 -e 10 -b 1024 -g -emb std

# With Dictionary, Deep Network, Hash Embedding (Sec 5.4)
python src/main.py -ds amazon_full -K 1000000 -k 2 -B 150000 -d 200 -H 1000 1000 1000 -e 10 -b 1024 -g -dict -emb hash


# With Dictionary, Deep Network, Std. Embedding (Sec 5.4)
python src/main.py -ds amazon_full -K 1000000 -k 2 -B 150000 -d 200 -H 1000 1000 1000 -e 10 -b 1024 -g -dict -emb std
