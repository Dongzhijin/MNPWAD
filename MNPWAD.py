import torch
import numpy as np
import random
from torch.nn import functional as F
import utils
from Dataset import DataGenerator
from pdb import set_trace
import time
import json
from model import *
from utils_pretrain import *


class MNPWAD:
    def __init__(
        self,
        device="cuda",
        nbatch_per_epoch=64,
        epochs=200,
        batch_size=128,
        n_emb=8,
        lr=0.005,
        T=2,
        loss_type="smooth",
        milestones=None,
        AS_type="AnomalyScore",
        prt_step=1,
        use_es=True,
        seed=42,
        logger=None,
        base_model="MNP",
        flag="noflag",
        m1=0.02,
        lambda_kl=1,
        output_path="output",
        dataset_name="noname",
        n_prototypes=0
    ):
        self.device = device

        self.epochs = epochs
        self.nbatch_pemr_epoch = nbatch_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_emb = n_emb
        self.T = T
        self.loss_type = loss_type
        self.milestones = milestones if milestones is not None else (self.epochs,)

        self.prt_step = prt_step
        self.use_es = use_es

        self.basenet = None
        self.criterion = None
        self.data = None
        self.dim = None
        self.logger = logger

        self.param_lst = locals()
        del self.param_lst["self"]
        del self.param_lst["device"]
        del self.param_lst["prt_step"]
        del self.param_lst["logger"]
        self.logger.info(json.dumps(self.param_lst), print_text=False)  # dict转换为 str
        self.multinormalprototypes = None
        self.PretrainAE=None
        self.base_model = base_model
        self.AS_type = AS_type
        self.flag = flag
        self.m1 = m1
        self.lambda_kl = lambda_kl
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.n_prototypes=n_prototypes
        self.seed=seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

    def PretrainedAE(self, train_x, train_semi_y, val_x, val_y):
        self.dim = train_x.shape[1]
        hidden_dims = [self.dim - (self.dim - self.n_emb) // 2, self.n_emb]
        AE,multinormalprototypes = pretrain_autoencoder(
            train_x,
            train_semi_y,
            val_x,
            val_y,
            self.output_path,
            self.dataset_name,
            hidden_dims,
            self.logger,
            self.device,
            self.seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            nbatch_pemr_epoch=self.nbatch_pemr_epoch,
            learning_rate=self.lr,
            n_prototypes=self.n_prototypes
        )
        return AE,multinormalprototypes

    def fit(self, train_x, train_semi_y, val_x, val_y,pretrainAE=True):
        if pretrainAE:
            self.PretrainAE,self.multinormalprototypes = self.PretrainedAE(train_x, train_semi_y, val_x, val_y)
            self.n_prototypes=self.multinormalprototypes.shape[0]
        device = self.device
        dim = train_x.shape[1]
        self.dim = dim
        self.data = DataGenerator(
            train_x,
            train_semi_y,
            batch_size=self.batch_size,
            device=self.device,
        )
        self.basenet = eval(self.base_model)(
            input_dim=self.dim,
            hidden_dims=[self.dim - (self.dim - self.n_emb) // 2, self.n_emb],
            loss_type=self.loss_type,
            PretrainAE=self.PretrainAE,
            multinormalprototypes=self.multinormalprototypes, 
            device=self.device,
            AS_type=self.AS_type,
        ).to(self.device)

        self.criterion = eval("Loss_" + self.base_model)(
            score_loss=self.loss_type,
            device=device,
            m1=self.m1,
            lambda_kl=self.lambda_kl,
        )

        optimizer = torch.optim.Adam(
            self.basenet.parameters(), lr=self.lr, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=0.4
        )
        early_stp = utils.EarlyStopping(
            patience=15,
            model_name=self.output_path + "/checkpoints/" + self.dataset_name + "_MNP",
            verbose=False,
        )
        pre_loss_recon, pre_loss_score, pre_loss_anchor = 1, 1, 1
        self.logger.info("start training epochs...")
        for step in range(self.epochs):
            start = time.time()
            batch_triplets = self.data.load_batches(n_batches=self.nbatch_pemr_epoch)
            batch_triplets = torch.from_numpy(batch_triplets).float().to(device)
            losses = []
            losses_recon, losses_score, losses_anchor = [], [], []
            self.basenet.train()
            for batch_triplet in batch_triplets:
                pos, neg = (
                    batch_triplet[: -self.batch_size],
                    batch_triplet[-self.batch_size :],
                )
                loss, loss_recon, loss_score, loss_anchor = self.criterion(
                    self.basenet,
                    pos,
                    neg,
                    pre_loss_recon,
                    pre_loss_score,
                    pre_loss_anchor,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.data.cpu().item())
                losses_recon.append([l.data.cpu().item() for l in loss_recon])
                losses_score.append([l.data.cpu().item() for l in loss_score])
                losses_anchor.append([l.data.cpu().item() for l in loss_anchor])
            end = time.time()

            val_start = time.time()
            self.basenet.eval()

            # val #
            val_score = self.predict(val_x)

            try:
                val_auroc, val_aupr = utils.evaluate(val_y, val_score)
                if self.use_es:
                    early_metric = (1 - val_aupr) + (1 - val_auroc)
                    early_stp(early_metric, model=self.basenet)
            except ValueError:
                self.logger.info("#" * 10 + "NaN")
                val_auroc, val_aupr = -1, -1
                if self.use_es:
                    early_metric = 2*(1 - val_aupr) + (1 - val_auroc)
                    early_stp(early_metric, model=self.basenet)
                    early_stp.early_stop = True
            val_end = time.time()

            t = end - start
            val_t = val_end - val_start
            losses, losses_recon, losses_score, losses_anchor = (
                np.array(losses),
                np.array(losses_recon),
                np.array(losses_score),
                np.array(losses_anchor),
            )

            if (
                (step + 1) % self.prt_step == 0
                or step == 0
                or (self.use_es and early_stp.early_stop)
                or early_stp.current_best
            ):
                self.logger.info(
                    f"【epoch {step+1}】:"
                    f"val-auroc/pr: {val_auroc:.4f}/{val_aupr:.4f}, time: {t:.2f}s"
                )
                self.logger.info(
                    f"loss (all/recon/score/anchor): {losses.mean():.4f} / {losses_recon[:,0].mean():.4f} / {losses_score[:,0].mean():.4f} / {losses_anchor[:,0].mean():.4f}"
                )
                self.logger.info(
                    f"loss_recon (pos/neg): {losses_recon[:,1].mean():.4f} / {losses_recon[:,2].mean():.4f}"
                )
                self.logger.info(
                    f"loss_score (pos/neg): {losses_score[:,1].mean():.4f} / {losses_score[:,2].mean():.4f}"
                )
                self.logger.info(
                    f"loss_anchor (pos/neg): {losses_anchor[:,1].mean():.4f} / {losses_anchor[:,2].mean():.4f}"
                )

            if (self.use_es and early_stp.early_stop) or val_auroc== 0:
                self.basenet.load_state_dict(torch.load(early_stp.path))
                self.logger.info("early stop") if (self.use_es and early_stp.early_stop) else self.logger.info("val_auroc==0")
                break

            scheduler.step()

            pre_loss_recon = losses_recon[:, 0].mean()
            pre_loss_score = losses_score[:, 0].mean()
            pre_loss_anchor = losses_anchor[:, 0].mean()
        return

    def predict(self, x_test, output_hidden=False):
        device = self.device
        with torch.no_grad():
            self.basenet.eval()
            xx = torch.from_numpy(x_test).float().to(device)
            if output_hidden:
                _, xx_s, _, xx_h, anchors = self.basenet(xx, output_hidden)
                xx_s = xx_s.flatten()
                xx_s = xx_s.data.cpu().numpy()
                xx_h = xx_h
                xx_h = xx_h.data.cpu().numpy()
                anchors = anchors.data.cpu().numpy()
                return xx_s, xx_h, anchors
            _, xx_s, _ = self.basenet(xx)
            xx_s = xx_s.flatten()
            xx_s = xx_s.data.cpu().numpy()
        return xx_s
    
