# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Finetuning torchvision models for the purpose of predicting
# Tinder swipes, left or right
#
# Based on https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
# Lots of things could be tweaked here:
#   - NN architecture
#   - Use a model pretrained on face tasks (instead of ImageNet)
#   - Grid search hyperparameters 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import print_function, division
from utilities import folder_assertions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import uuid
import data


class Libido:


    def __init__(self, sorted_data_dir, temp_data_dir, trained_models_dir, pretrained=True, feature_extraction=True):

        """ Assertions """
        assert sorted_data_dir is not None, "Invalid sorted data folder name"
        assert temp_data_dir is not None, "Invalid temporary data folder name"
        assert trained_models_dir is not None, "Invalid trained models folder name"

        """ Variables """

        # Folders
        self.sorted_data_dir = sorted_data_dir
        self.temp_dir = temp_data_dir
        self.temp_dir_save = os.path.join(self.temp_dir, "1/")
        self.models_dir = trained_models_dir
        self.trainable_model_path = str(uuid.uuid1()) + ".pth"

        assert folder_assertions([self.sorted_data_dir, self.temp_dir, self.temp_dir_save, self.models_dir]) == True, "Couldn't create data folders"


        # Training hyperparameters
        self.num_epochs = 25
        self.batch_size = 16
        self.pretrained = pretrained
        self.feature_extraction = feature_extraction

        self.initial_learning_rate = 0.0012
        self.learning_rate_decay = 0.0001
        self.learning_rate_decay_rate = 2

        # Dataset transformations
        # -> Augmentation and normalization for training
        # -> Normalization for testing
        
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),  # RandomResizedCrop(224), - currently assumes input images are square
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Dataset loaders
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.sorted_data_dir, x), self.data_transforms[x]) for x in ['train', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in
                       ['train', 'test']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'test']}
        
        self.class_names = self.image_datasets['train'].classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Using CUDA? {} - {}".format(torch.cuda.is_available(), self.device))

        # Initialize model
        self.model_ft = models.resnet34(pretrained=self.pretrained)
        self.set_parameter_requires_grad(self.model_ft, feature_extraction=self.feature_extraction)
        num_ftrs = self.model_ft.fc.in_features
        num_classes = len(self.class_names)
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
        self.model_ft = self.model_ft.to(self.device)


    def train_model(self, num_epochs=25):

        """ Train a model """
        """ Assumes self.sorted_data_dir containes train/left, train/right, test/left, test/right """
        """ You can obtain this format by using data.setup_entire_dataset """

        # Reassign intrinsic parameters
        self.num_epochs = num_epochs

        # Define loss criteria
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized - finetuning
        optimizer_ft = optim.SGD(self.model_ft.parameters(), lr=self.initial_learning_rate, momentum=0.9)

        # Decay LR by a factor of 0.001 every <step_size> epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=self.learning_rate_decay_rate, gamma=self.learning_rate_decay)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.model_ft = self._train_model(self.model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=self.num_epochs)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
        
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best accuracy: {:4f}'.format(best_acc))

        # Save after training
        torch.save(best_model_wts, os.path.join(self.models_dir, self.trainable_model_path))

        # Load model
        model.load_state_dict(best_model_wts)

        # Visualize best model
        self.visualize_model(model)
        plt.show()

        # Return model
        return model


    def imshow(self, inp, title=None):

        """Imshow for Tensor"""
        
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated


    def visualize_model(self, model, num_images=6):

        was_training = model.training
        model.eval()

        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            
            for i, (inputs, labels) in enumerate(self.dataloaders['test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_names[preds[j]]))
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            
            model.train(mode=was_training)

    def _dataloader_from_temp_folder(self):
        
        """
        TODO: Instead of creating this every time a new batch of images needs to be predicted,
        we could just add this dataloader initialization in the beginning (["train", "test", "infer"]), under the assumption
        thid dataloader will get updated as files in __temp__ come and go. To be tested.

        """
        
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image_dataset = datasets.ImageFolder(self.temp_dir, transform)
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return dataloader
        

    def _get_batch_predictions(self, dataloader):

        latest_model_name = self.get_latest_model()
        self.load_pretrained(latest_model_name)
        self.model_ft.eval()

        with torch.no_grad():
            
            for i, (inputs, labels) in enumerate(dataloader):
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model_ft(inputs)

                #_, preds = torch.max(outputs, 1)

                return outputs

                # TODO: Some math to round up the predictions


    def infer(self):
        
        data.preprocess_pipeline(self.temp_dir_save)
        dataloader = self._dataloader_from_temp_folder()
        
        # Mean approach
        preds = self._get_batch_predictions(dataloader)
        mean_per_class = torch.mean(preds, 1)
        mean_np = torch.mean(mean_per_class)
        mean = int(np.round(mean_np))
        
        result = self.class_names[mean]
        
        # Mode approach
        #value = stats.mode(preds.cpu().detach().numpy())[0]

        print(result)
        return result


    def set_parameter_requires_grad(self, model, feature_extraction=False):
        if feature_extraction:
            for param in model.parameters():
                param.requires_grad = False


    def show_pretrained_model(self, model_name=None):

        if not model_name:
            latest_model_name = self.get_latest_model()
            self.load_pretrained(latest_model_name) # if it fails, load latest model
        else:
            self.load_pretrained(model_name) # try to load specified model
            

        self.visualize_model(self.model_ft)
        plt.show()


    def load_pretrained(self, model_name):

        model_path = os.path.join(self.models_dir, model_name)

        if not os.path.isfile(model_path):
            raise Exception("No model named at {}".format(model_path))
        
        try:
            self.model_ft.load_state_dict(torch.load(model_path))
        except RuntimeError: # Happens if loaded model was trained on GPU and only CPU is available
            self.model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    def get_latest_model(self):

        all_models = [os.path.join(self.models_dir, m) for m in os.listdir(self.models_dir) if not os.path.isdir(m)]
        
        latest_model_path = max(all_models, key=os.path.getmtime)
        
        print(f"latest_model_path {latest_model_path}")
        
        return os.path.basename(latest_model_path)
