import torch 
from torch.utils import data 
from torch import nn
from losses import eval_metrics

from dataset.dataset import SegmentationDataset

from tqdm import tqdm 
import logging

import typing
import numpy

logger = logging.getLogger(__name__)


class NetworkTrainer(object):
    """
    Implementation of the class
    for training neural network
    
    Parameters:
    -----------

    network: (nn.Module) - network container string

    optimizer: () - optimizer for training

    loss_function - loss for training 

    eval_metric - evaluation metric for validation stage

    max_epochs - maximum number of epochs 

    train_device - device, network is going to be connected to during training stage and validation stage
    (can be either cpu, mps or cuda)

    lr_scheduler - learning rate scheduling technique
    """
    def __init__(self,
        network: nn.Module,
        optimizer: nn.Module,
        loss_function: nn.Module,
        batch_size: int,
        max_epochs: int,
        train_device='cpu',
        lr_scheduler: nn.Module=None,
    ):
        self.max_epochs = max_epochs 
        self.batch_size = batch_size
        self.train_device = train_device
        self.optimizer = optimizer
        self.network = network.to(self.train_device)
        self.loss_function = loss_function
        self.eval_metric = eval_metrics.F1Score()
        self.lr_scheduler = lr_scheduler
        
    def freeze_layers(self, layers_to_freeze: typing.List[nn.Module]):
        """
        Function freezes network layers, specified in
        a given list 'layers_to_freeze'
        """
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def train(self, train_dataset: SegmentationDataset):

        self.network.train()
        
        train_data = data.DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        best_cl_loss = None
        train_loss_history = []

        for epoch in range(self.max_epochs):

            epoch_losses = []

            for imgs, masks in tqdm(train_data):
                
                # masks of probabilities [ [0.4, 0.6], [0.2, 0.8] ]
                
                predicted_masks = self.network.forward(
                imgs.to(self.train_device)).cpu()
                
                total_loss = 0
                total_losses = []
                
                # computing loss function
                for idx, pred_mask in enumerate(predicted_masks):
                    loss = self.loss_function(pred_mask, masks[idx])
                    total_loss += loss.item()
                    total_losses.append(loss)
                
                total_loss_func = sum(total_losses)
                # backward step 
                total_loss_func.backward()

                # optimizer step 
                self.optimizer.step()

                epoch_losses.append(total_loss)

            # learning scheduling step
            if self.lr_scheduler:
                self.lr_scheduler.step()

            avg_loss = numpy.mean(epoch_losses)

            if best_cl_loss is None or best_cl_loss > avg_loss:
                best_cl_loss = avg_loss
                    
            train_loss_history.append(avg_loss)
            print('%s epochs passed' % str(epoch))
            
            if epoch % 5 == 0:
                self.save_checkpoint(best_cl_loss, epoch)
            
        return best_cl_loss, train_loss_history
    
    def save_checkpoint(self, loss, epoch):
        torch.save(
            obj={
                'network_state': self.network.state_dict(),
                'optimizer_state': self.optmizer.state_dict(),
            }, f='../checkpoints/checkpoint_epoch_%s.pth' % str(epoch)
        )

    def evaluate(self, validation_dataset: SegmentationDataset):
        """
        Function evaluates network on a given
        set of validation samples
        """
        self.network.eval()
        
        loader = data.DataLoader(
            dataset=validation_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        ) 

        total_val_metric = []
        
        with torch.no_grad():

            for imgs, masks in loader:
                try:
                    predicted_masks = self.network.forward(
                        imgs.float().to(self.train_device)
                    ).cpu().to(torch.uint8)
                    
                    eval_metric = self.eval_metric(
                        predicted_masks, 
                        masks
                    )
                    total_val_metric.append(eval_metric)
                    
                except Exception as err:
                    logger.debug(err)
                    raise RuntimeError(
                    "Failed to predict mask, check logs for more info")
        return numpy.mean(total_val_metric)
    
    def predict(self, image: numpy.ndarray):
        predicted_mask = self.network.forward(image.to(self.train_device)).cpu()
        return predicted_mask
    
    def save_network(self, model_path: str, test_input: torch.tensor):
        torch.onnx.export(model=self.network, args=test_input, f=model_path)
