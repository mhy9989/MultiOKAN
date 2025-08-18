from .template import Template
from models import (L_DeepONet_Model_AE_multi, L_DeepONet_Model_AE_fusion, 
                    L_DeepONet_Model_AE_Conv_multi, L_DeepONet_Model_AE_Conv_fusion,
                    L_DeepONet_Model_DON_multi, L_DeepONet_Model_DON_Conv_multi)
import numpy as np
import torch

class L_DeepONet(Template):

    def __init__(self, args, ds_config, base_criterion):
        Template.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(args)
        self.model1 = None
        self.m = args.latent_dim
        self.args = args
        self.torch_scaler = None
        X_loc = np.array([i * args.dt for i in args.data_after]).reshape(args.data_after_num, 1)
        if "DON" in self.args.model_type:
            if args.t_norm == "MinMax":
                self.x1 = torch.Tensor(2* (X_loc-X_loc.min()) / (X_loc.max() - X_loc.min()) -1).to(self.device) if args.data_after_num != 1 else torch.Tensor(X_loc).to(self.device)
            elif args.t_norm == "Standard":
                self.x1 = torch.Tensor((X_loc-X_loc.mean())/X_loc.std()).to(self.device) if args.data_after_num != 1 else torch.Tensor(X_loc).to(self.device)
            else:
                raise ValueError(f"Error type of t_norm: {args.t_norm}")

    def build_model(self, args):
        if args.model_type == "AE_multi":
            return L_DeepONet_Model_AE_multi(**args).to(self.device)
        elif args.model_type == "AE_fusion":
            return L_DeepONet_Model_AE_fusion(**args).to(self.device)
        elif args.model_type == "AE_conv_multi":
            return L_DeepONet_Model_AE_Conv_multi(**args).to(self.device)
        elif args.model_type == "AE_conv_fusion":
            return L_DeepONet_Model_AE_Conv_fusion(**args).to(self.device)
        elif args.model_type == "DON_multi":
            return L_DeepONet_Model_DON_multi(**args).to(self.device)
        elif args.model_type == "DON_conv_multi":
            return L_DeepONet_Model_DON_Conv_multi(**args).to(self.device)
    
    def predict_AE_en(self, batch_x):
        """Forward the AE en model"""
        pred_y = self.model.encoder(batch_x)
        return pred_y
    
    def predict_AE_de(self, batch_x):
        """Forward the AE de model"""
        pred_y = self.model.decoder(batch_x)
        return pred_y
    
    def predict(self, batch_x):
        """Forward the model"""
        if "DON" in self.args.model_type:
            pred_y = self.model(batch_x, self.x1)
        else:
            pred_y = self.model(batch_x)
        return pred_y

