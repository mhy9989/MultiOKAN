from models import (L_DeepOKan_Model_AE_multi, L_DeepOKan_Model_AE_fusion, 
                    L_DeepOKan_Model_AE_Conv_multi, L_DeepOKan_Model_AE_Conv_fusion,
                    L_DeepOKan_Model_DON_multi, L_DeepOKan_Model_DON_Conv_multi)
from .l_deeponet import L_DeepONet

class L_DeepOKan(L_DeepONet):

    def __init__(self, args, ds_config, base_criterion):
        L_DeepONet.__init__(self, args, ds_config, base_criterion)

    def build_model(self, args):
        if args.model_type == "AE_multi":
            return L_DeepOKan_Model_AE_multi(**args).to(self.device)
        elif args.model_type == "AE_fusion":
            return L_DeepOKan_Model_AE_fusion(**args).to(self.device)
        elif args.model_type == "AE_conv_multi":
            return L_DeepOKan_Model_AE_Conv_multi(**args).to(self.device)
        elif args.model_type == "AE_conv_fusion":
            return L_DeepOKan_Model_AE_Conv_fusion(**args).to(self.device)
        elif args.model_type == "DON_multi":
            return L_DeepOKan_Model_DON_multi(**args).to(self.device)
        elif args.model_type == "DON_conv_multi":
            return L_DeepOKan_Model_DON_Conv_multi(**args).to(self.device)
