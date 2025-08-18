import os.path as osp
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from DataDefine import get_datloader, get_lam_datloader
import torch
from deepspeed.accelerator import get_accelerator
from core import metric, Recorder
from timm.utils import AverageMeter
from utils import (plot_figure, check_dir, print_log, weights_to_cpu,
                   measure_throughput, output_namespace)


class modeltrain(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, model_data, model_path, mode = "train", infer_num = [-1]):
        """Initialize experiments (non-dist as an example)"""
        self.args = model_data[0]
        self.device = self.args.device
        self.method = model_data[1]
        self.args.method = self.args.method.lower()
        self.epoch = 0
        self.max_epochs = self.args.max_epoch
        self.steps_per_epoch = self.args.steps_per_epoch
        self.rank = self.args.rank
        self.world_size = self.args.world_size
        self.dist = self.args.dist
        self.early_stop = self.args.early_stop_epoch
        self.model_path = model_path
        self.best_loss = 100.
        self.infer_num = infer_num
        self.mode = mode
        self.inference_list = None
        self.method1 = None
        self.orgs_data=None

        self.preparation()
        if self.args.init:
            print_log(output_namespace(self.args))
            if self.args.if_display_method_info:
                self.display_method_info()


    def preparation(self):
        """Preparation of basic experiment setups"""
        if self.early_stop <= self.max_epochs // 5:
            self.early_stop = self.max_epochs * 2

        self.checkpoints_path = osp.join(self.model_path, 'checkpoints')
        # load checkpoint
        if self.args.load_from and self.mode == "train":
            if self.args.load_from == True:
                self.args.load_from = 'latest'
            self.load(name=self.args.load_from)
        # prepare data
        if self.args.init:
            self.get_data()


    def get_data(self):
        """Prepare datasets and dataloaders"""
        if self.mode == "train" or self.mode == "test":
            (self.train_loader, 
            self.vali_loader, 
            self.test_loader, 
            self.scaler,
            self.x_mesh, 
            self.y_mesh,
            self.orgs_data) = get_datloader(self.args)
            self.method.scaler = self.scaler
        else:
            (self.infer_loader,
            self.scaler, 
            self.x_mesh, 
            self.y_mesh,
            self.orgs_data) = get_datloader(self.args, "inference", self.infer_num)
            self.method.scaler = self.scaler


    def save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self.epoch + 1,
            'optimizer': self.method.optimizer.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self.dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()
            }
        if self.rank == 0:
            torch.save(checkpoint, osp.join(self.checkpoints_path, f'{name}.pth'))
        del checkpoint
    

    def save_checkpoint(self, name=''):
        """Saving models data to checkpoints"""
        checkpoint = weights_to_cpu(self.method.model.state_dict()) \
                if not self.dist else weights_to_cpu(self.method.model.module.state_dict())
        if self.rank == 0:
            torch.save(checkpoint, osp.join(self.checkpoints_path, f'{name}.pth'))
        del checkpoint


    def load(self, name=''):
        """Loading models from the checkpoint"""
        filename = osp.join(self.checkpoints_path, f'{name}.pth')
        try:
            checkpoint = torch.load(filename, map_location='cpu',weights_only=True)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self.load_from_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None and self.args.if_continue:
            self.epoch = checkpoint['epoch']
            self.method.optimizer.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])
            cur_lr = self.method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            print_log(f"Successful optimizer state_dict, Lr: {cur_lr:.5e}")
        del checkpoint


    def load_from_state_dict(self, state_dict):
        if self.dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)
        self.method.model.eval()
        print_log(f"Successful load model state_dict")


    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        if self.args.method in ['l-deeponet','l-deepokan']:
            input_dummy = torch.ones(1, 1, *self.args.input_shape).to(self.device)
            if "DON" in self.args.model_type:
                input_dummy2 = torch.ones(self.args.data_after_num,1).to(self.device)
                input_dummy = (input_dummy, input_dummy2)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(self.args.method, fps)
        else:
            fps = ''
        print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)


    def train(self):
        """Training loops of methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self.max_epochs // 10, 30), 
                            rank = self.rank, dist=self.dist, max_epochs = self.max_epochs,
                            method = self.args.method)
        num_updates = self.epoch * self.steps_per_epoch
        vali_loss = False
        early_stop = False
        epoch_time_m = AverageMeter()
        train_time_m = AverageMeter()

        for epoch in range(self.epoch, self.max_epochs):
            begin = time.time()

            if self.dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean = self.method.train_one_epoch(self.train_loader,
                                                                epoch, num_updates)
            train_time_m.update(time.time() - begin)

            self.epoch = epoch
            with torch.no_grad():
                vali_loss = self.vali()
            epoch_time_m.update(time.time() - begin)
            
            cur_lr = self.method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            
            epoch_log = 'Epoch: {0}, Steps: {1} | Lr: {2:.5e} | Train Loss: {3:.5e}'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg)
            if vali_loss:
                epoch_log += f" | Vali Loss: {vali_loss:.5e}"
            print_log(epoch_log)

            print_log(f'Epoch time: {epoch_time_m.val:.2f}s, Training time: {train_time_m.val:.2f}s | ' 
                      f'Average time: {epoch_time_m.avg:.2f}s, Average Training time: {train_time_m.avg:.2f}')

            if self.args.mem_log:
                MemAllocated = round(get_accelerator().memory_allocated() / 1024**3, 2)
                MaxMemAllocated = round(get_accelerator().max_memory_allocated() / 1024**3, 2)
                print_log(f"MemAllocated: {MemAllocated} GB, MaxMemAllocated: {MaxMemAllocated} GB")

            early_stop = recorder(loss_mean.avg, vali_loss, self.method.model, self.model_path, epoch)
            self.best_loss = recorder.val_loss_min

            # if (epoch +1)% 50 == 0 or (epoch +1) == self.max_epochs:
            #     self.save(name='latest')

            if epoch > self.early_stop and early_stop:  # early stop training
                print_log('Early stop training at f{} epoch'.format(epoch))
                break
            
            if self.args.empty_cache:
                torch.cuda.empty_cache()
            print_log("")
            
            
        self.save_checkpoint("last_checkpoint")

        if not check_dir(self.model_path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        time.sleep(1)

    def vali(self):
        """A validation loop during training"""
        results, eval_log = self.method.vali_one_epoch(self.vali_loader)
        if self.args.val_metrics:
            print_log('Val_metrics\t'+eval_log)
        return results['loss'].mean()


    def test_inference(self, min_max_delt=None, if_AE=False):
        """test of inference loop of methods"""
        best_model_path = osp.join(self.checkpoints_path, 'checkpoint.pth')
        self.load_from_state_dict(torch.load(best_model_path,weights_only=True))
        if self.mode=="inference":
            results = self.method.test_one_epoch(self.infer_loader,if_AE)
        else:
            results = self.method.test_one_epoch(self.test_loader)
        metric_list = self.method.metric_list
        channel_names = self.args.data_use

        if if_AE:
            AE_path = osp.join(self.model_path, 'saved', "AE")
            check_dir(AE_path)
            # inputs
            inputs = results["inputs"].reshape(self.args.data_num, self.args.data_after_num+1 ,
                                               self.args.features, self.args.data_width, self.args.data_height)
            np.save(osp.join(AE_path, 'AE_inputs.npy'), inputs)
            # preds
            inputs = results["preds"].reshape(self.args.data_num, self.args.data_after_num+1 ,
                                               self.args.features, self.args.latent_dim)
            np.save(osp.join(AE_path, 'AE_preds.npy'), inputs)
            return

        # Computed
        self.test_unit(results,metric_list,channel_names,"Computed",min_max_delt)

        # Original
        if self.scaler[-1]:
            results_n = self.de_norm(results, self.scaler[-1])
            self.test_unit(results_n,metric_list,channel_names,"Original",min_max_delt)
        return 
    
    def test_inference_n(self, min_max_delt=None):
        """test of inference loop of methods"""
        best_model_path = osp.join(self.checkpoints_path, 'checkpoint.pth')
        self.load_from_state_dict(torch.load(best_model_path,weights_only=True))
        if self.mode=="inference":
            results = self.method.test_one_epoch(self.infer_loader)
        else:
            results = self.method.test_one_epoch(self.test_loader)
        metric_list = self.method.metric_list
        channel_names = self.args.data_use

        preds = results["preds"]  #(num,  after, features, lam)
        inputs_org = self.orgs_data[:, :1] #(num, 1, features, H, W)
        trues_org = self.orgs_data[:, 1:] #(num, after, features, H, W)
        n, a, f, h, w = trues_org.shape

        dataloader = get_lam_datloader(self.args, preds, trues_org, self.scaler)

        resultsn = self.method1.test_one_epoch(dataloader, "de")

        preds_org_norm = resultsn["preds"] #(num * after, features, H, W)
        preds_org_norm = preds_org_norm.reshape(n, a, f, h, w) #(num, after, features, H, W)

        trues_org_norm = resultsn["labels"] #(num * after, features, H, W)
        trues_org_norm = trues_org_norm.reshape(n, a, f, h, w) #(num, after, features, H, W)

        resultsn = {
            "inputs": inputs_org,
            "preds": preds_org_norm,
            "labels": trues_org_norm
        }

        # Computed
        self.test_unit(resultsn,metric_list,channel_names,"Computed",min_max_delt)

        # Original
        if self.scaler[0]:
            resultsn = self.de_norm(resultsn, self.scaler[0])
            resultsn["inputs"] = inputs_org
            self.test_unit(resultsn,metric_list,channel_names,"Original",min_max_delt)
        return 


    def test_unit(self, results, metric_list, channel_names, mode="Computed",min_max_delt=None):
        eval_res_av, eval_log_av = metric(results['preds'], results['labels'],
                                    metrics=metric_list, channel_names=channel_names, mode = mode)
        results['metrics'] = np.array([eval_res_av['mae'], eval_res_av['mse']])
        print_log(f"Total:")
        print_log(f"{eval_log_av}\n")
        if self.rank == 0:
            folder_path = osp.join(self.model_path, 'saved', mode,"total")
            check_dir(folder_path)
            for np_data in ['metrics', 'inputs', 'labels', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])
            
            # plot input
            for i in range(self.args.features):
                input_pic = osp.join(self.model_path, "pic", mode, "input", self.args.data_use[i])
                input = results['inputs'][-1,0,i]
                check_dir(input_pic)
                min_max = [input.min(), input.max()]
                plot_figure(self.x_mesh, self.y_mesh, min_max, input, 
                            self.args.data_use[i], "input", mode, input_pic, 300)

        for t in range(self.args.data_after_num if "DON" in self.args.model_type else 1):
            eval_res_av, eval_log_av = metric(results['preds'][:, None, t], results['labels'][:,None, t],
                                        metrics=metric_list, channel_names=channel_names, mode = mode)
            print_log(f"After {t}:")
            print_log(f"{eval_log_av}\n")
            if self.rank == 0:
                self.plot_test(t, results['preds'][-1,t], results['labels'][-1,t], mode,
                            min_max_delt=min_max_delt)


    def plot_test(self, t, preds, labels, mode, 
                  dpi = 300, dir_name = "pic", 
                  min_max_base = None, min_max_delt = None):
        features = self.args.features 
        pic_folder = osp.join(self.model_path, dir_name, mode, f"after{t}")
        check_dir(pic_folder)

        for i in range(features):
            select_pic = osp.join(pic_folder, self.args.data_use[i])
            check_dir(select_pic)
            
            if min_max_base == None:
                min_max = [labels[i].min(), labels[i].max()]
            else:
                min_max = min_max_base
            plot_figure(self.x_mesh, self.y_mesh, min_max, labels[i], 
                            self.args.data_use[i], "label", mode, select_pic, dpi)
            plot_figure(self.x_mesh, self.y_mesh, min_max, preds[i], 
                            self.args.data_use[i], "pred", mode, select_pic, dpi)
            if min_max_delt == None:
                min_max = [(preds[i]-labels[i]).min(), (preds[i]-labels[i]).max()]
            else:
                min_max = min_max_delt
            plot_figure(self.x_mesh, self.y_mesh, min_max, preds[i]-labels[i],
                        self.args.data_use[i], "delt", mode, select_pic, dpi)
        return None


    def de_norm(self, results, scaler):
        results_ori = {}
        for name in results.keys():
            if name in ['inputs', 'labels', 'preds'] and scaler:
                B, T, F = results[name].shape[:3]
                results_ori[name] = np.zeros(results[name].shape)
                for b in range(B):
                    for t in range(T):
                        for f in range(F):
                            results_ori[name][b, t, f] = scaler[f].inverse_transform(results[name][b, t, f].reshape(-1,1)).reshape(*results[name].shape[3:])
            else:
                results_ori[name] = results[name]
        return results_ori
