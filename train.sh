python train.py --datapath ./data --datasets IoT23_CandC,IoT23_Attack,IoT23_DDoS,IoT23_Okiru \
--trainflag origin --labeled_ratio 0.01 --runs 5 --base_model MNP --gpu 0,1 --batch_size 128 --lr 0.005 \
--n_emb 8 --m1 0.02 --debug False --dataset2n_prototypes dataset2n_prototypes