import torch.nn as nn
from pkg_resources import parse_version
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import numpy as np
import torch
import os

class QuantizationAwareTrainingWrapper():
    """
    Wrapper for all the trainig modalities. Activation of some modalities are based on parameter passed in the constructor
    """
    def __init__(self, student_network, num_classes, config_file, optimizer, writer, logger, teacher_network = None, annotation_converter = None,  criterion = nn.CrossEntropyLoss(),\
        current_epoch = 0, max_epochs = 100, test_rate = 10, scheduler = None, best_recall = 0.0, temperature=5, teacher_weight=0.7, quantization_framework=False, freeze_observer=None, freeze_bn=None):
        """
        Arguments:
            student_network: model to be trained using distillation loss
            num_classes: number of classes in the task
            config_file: configuration file
            optimizer: type of gradient descent
            writer: writer to TensorBoard
            logger: writer for log.txt
            teacher_network: full trained model without softmax in the end
            annotation_converter: convert from index to object name or vica versa
            criterion: loss to be used to compare output of student network and label
            current_epoch: set if it is training continuation
            max_epochs: how many epochs the training would be
            test_rate: after how many epoch would the model be tested
            scheduler: training scheduler
            best_recall: best recall of previous training
            temperature: temperature used in the knowledge distillation
            teacher_weight: how relevant the teacher loss
            quantization_framework: True if PyTorch quantization framework is used
            freeze_observer: From this epoch observers are freezed
            freeze_bn: From this epoch batch normalizations are freezed
            quantization_function: quantization function, if None, normal knowledge distillation would be used
            
        """
        self.teacher_network = teacher_network
        self.student_network = student_network
        self.num_classes = num_classes
        self.config_file = config_file
        self.optimizer = optimizer
        self.writer = writer
        self.logger = logger
        self.annotation_converter = annotation_converter
        self.criterion = criterion
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.test_rate = test_rate
        self.scheduler = scheduler
        self.best_recall = best_recall
        self.quantization_framework = quantization_framework
        self.temperature = temperature
        self.teacher_weight = teacher_weight
        self.freeze_bn = freeze_bn
        self.freeze_observer = freeze_observer

    def train(self, train_loader, test_loader):
        """
        This method represents training procedure. Based on parameter passed in the constructor, features are activated
        Arguments:
            train_loader: data loader of training dataset
            test_loader: data loader of test dataset
        """

        if torch.cuda.is_available():
            self.logger.info("CUDA FOUND")
            self.student_network.cuda()
            if self.teacher_network is not None:
                self.teacher_network.cuda()
        
        self.student_network.train()
        if self.teacher_network is not None:    
            self.teacher_network.eval()

        for epoch in range(self.max_epochs):

            if parse_version(torch.__version__) >= parse_version("0.4.0"):
                torch.set_grad_enabled(True)
                print("Pytorch version >= 0.4.0")
            else:
                print("Pytorch version < 0.4.0")

            self.student_network.train()
            if torch.cuda.is_available():
                self.student_network.cuda()
                if self.teacher_network is not None:
                    self.teacher_network.cuda()
            if self.teacher_network is not None:    
                self.teacher_network.eval()

            self.optimizer.zero_grad()

            mini_batch_losses = []

            if self.current_epoch != 0:
                self.logger.info("Continuing training from epoch: " + str(self.current_epoch))
            else:
                self.logger.info("Starting training model/pretrained model")

            if self.freeze_bn is not None:
                if epoch > self.freeze_bn:
                    self.student_network.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if self.freeze_observer is not None:
                if epoch > self.freeze_observer:
                    self.student_network.apply(torch.quantization.disable_observer)

            for i, data in enumerate(train_loader, 0):
                self.logger.info("Training on epoch " + str(self.current_epoch + 1) + ", batch " + str(i))
                
                inputs, target = data

                if torch.cuda.is_available():
                    inputs, target = Variable(inputs.cuda()), Variable(target.cuda())
                else:
                    inputs, target = Variable(inputs), Variable(target)

                if torch.cuda.is_available():
                    student_outputs = self.student_network(inputs).cuda()
                else:
                    student_outputs = self.student_network(inputs)

                softmaxFunction = nn.Softmax(dim=1)

                current_loss = self.criterion(softmaxFunction(student_outputs), target)

                if self.teacher_network is not None:
                    current_loss = (1 - self.teacher_weight) * current_loss
                    if torch.cuda.is_available():
                        with torch.no_grad():
                            teacher_outputs = self.teacher_network(inputs).cuda()
                    else:
                        with torch.no_grad():
                            teacher_outputs = self.teacher_network(inputs)
                    logSoftmaxFunction, klDiv = nn.LogSoftmax(dim=1), nn.KLDivLoss()
                    current_loss += self.teacher_weight * self.temperature**2 * klDiv(logSoftmaxFunction(student_outputs/self.temperature), softmaxFunction(teacher_outputs/self.temperature))

                mini_batch_losses.append(float(current_loss))
                current_loss.backward()


                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.writer.add_scalar("Learning rate", self.scheduler.get_lr()[0], self.current_epoch + 1)
                self.scheduler.step()
            else:
                for param_group in self.optimizer.param_groups:
                    self.writer.add_scalar("Learning rate", param_group['lr'], self.current_epoch + 1)

            current_epoch_loss = np.mean(mini_batch_losses)
            self.writer.add_scalar("loss", current_epoch_loss, self.current_epoch + 1)
            self.logger.info("Loss for epoch " + str(self.current_epoch + 1) + " is " + str(current_epoch_loss))

            if epoch % self.test_rate == 0:
                del inputs, target

                self.optimizer.zero_grad()

                student_mean_recall, student_mean_precision, student_recall_list, student_precision_list, teacher_mean_recall, teacher_mean_precision, teacher_recall_list, teacher_precision_list = self.test(test_loader)

                self.write_to_tensorboard(epoch, self.writer, self.annotation_converter, student_mean_recall, student_mean_precision, student_recall_list, student_precision_list, teacher_mean_recall, teacher_mean_precision, teacher_recall_list, teacher_precision_list)

                if student_mean_recall > self.best_recall:
                    self.best_recall = student_mean_recall
                    self.best_epoch = epoch
                    self.save_model(epoch, self.student_network, self.quantization_framework, self.optimizer, self.criterion, self.annotation_converter, self.config_file['experiment']['model_save_path'] + "/" + self.config_file['experiment']['name'] + "/exp" + str(self.config_file["experiment"]["experiment_number"]) + "/best_model.pth", self.teacher_network, self.scheduler, self.quantization_function)

            self.current_epoch += 1

        self.logger.info("Training complete for epoch " + str(epoch))
        self.logger.info("The best mean recall is " + str(self.best_recall))
        self.logger.info("The best epoch is " + str(self.best_epoch + 1))

        self.save_model(epoch, self.student_network, self.quantization_framework, self.optimizer, self.criterion, self.annotation_converter, self.config_file['experiment']['model_save_path'] + "/" + self.config_file['experiment']['name'] + "/exp" + str(self.config_file["experiment"]["experiment_number"]) + "/latest_model.pth", self.teacher_network, self.scheduler)

    def test(self, test_loader):
        """
        This method represents model testing procedure. The frequency of the test is defined in the config file.
        Arguments:
            test_loader: data loader of test dataset
        """

        if parse_version(torch.__version__) >= parse_version("0.4.0"):
            torch.set_grad_enabled(False)
            print("Testing. Pytorch version >= 0.4.0")
        else:
            print("Testing. Pytorch version < 0.4.0")

        architecture_to_evaluate = self.student_network.eval()
        if self.quantization_framework is True:
            architecture_to_evaluate = torch.quantization.convert(architecture_to_evaluate.to('cpu'), inplace=False)

        student_class_correct = [0. for i in range(self.num_classes)]
        student_class_incorrect = [0. for i in range(self.num_classes)]
        student_class_total_target = [0. for i in range(self.num_classes)]
        student_class_total_pred = [0. for i in range(self.num_classes)]
        student_available_list = []

        teacher_class_correct = [0. for i in range(self.num_classes)]
        teacher_class_incorrect = [0. for i in range(self.num_classes)]
        teacher_class_total_target = [0. for i in range(self.num_classes)]
        teacher_class_total_pred = [0. for i in range(self.num_classes)]
        teacher_available_list = []

        for data in test_loader:
            inputs, target = data

            if torch.cuda.is_available() and self.quantization_framework is False:
                inputs, target = Variable(inputs.cuda(),volatile=True), Variable(target.cuda(),volatile=True)
            else:
                inputs, target = Variable(inputs), Variable(target)

            self.optimizer.zero_grad()

            if torch.cuda.is_available() and self.quantization_framework is False:
                student_outputs = architecture_to_evaluate(inputs).cuda()
            else:
                student_outputs = architecture_to_evaluate(inputs)

            teacher_outputs = []

            teacher_pred = []

            if self.teacher_network is not None:
                if torch.cuda.is_available() and self.quantization_framework is False:
                    teacher_outputs = self.teacher_network(inputs).cuda()
                else:
                    self.teacher_network.to('cpu')
                    teacher_outputs = self.teacher_network(inputs)

            softmaxFunction = nn.Softmax(dim=1)

            _, student_pred = torch.max(softmaxFunction(student_outputs), 1)

            if self.teacher_network is not None:
                _, teacher_pred = torch.max(softmaxFunction(teacher_outputs), 1)

            student_correct = [1 if current_pred == current_target else 0 for current_pred, current_target in zip(target.data, student_pred)]
            student_incorrect = [1 if current_pred != current_target else 0 for current_pred, current_target in zip(target.data, student_pred)]

            teacher_correct = [1 if current_pred == current_target else 0 for current_pred, current_target in zip(target.data, teacher_pred)]
            teacher_incorrect = [1 if current_pred != current_target else 0 for current_pred, current_target in zip(target.data, teacher_pred)]

            for i in range(len(target.data)):
                student_current_target = int(target.data[i])
                student_current_prediction = int(student_pred[i])
                #add 1 if the target is classified right
                student_class_correct[student_current_target] += student_correct[i]
                #add 1 if the prediction is false
                student_class_incorrect[student_current_prediction] += student_incorrect[i]
                student_class_total_pred[student_current_prediction] += 1
                student_class_total_target[student_current_target] += 1
                student_available_list.append(student_current_target)

                if self.teacher_network is not None:
                    teacher_current_target = int(target.data[i])
                    teacher_current_prediction = int(teacher_pred[i])
                    #add 1 if the target is classified right
                    teacher_class_correct[teacher_current_target] += teacher_correct[i]
                    #add 1 if the prediction is false
                    teacher_class_incorrect[teacher_current_prediction] += teacher_incorrect[i]
                    teacher_class_total_pred[teacher_current_prediction] += 1
                    teacher_class_total_target[teacher_current_target] += 1
                    teacher_available_list.append(teacher_current_target)

        student_recall_list = []
        student_precision_list = []

        teacher_recall_list = []
        teacher_precision_list = []

        for i in set(student_available_list):
            student_current_recall = student_class_correct[i] / student_class_total_target[i]

            if student_class_total_pred[i] != 0:
                student_current_precision = student_class_correct[i] / student_class_total_pred[i]
            else:
                student_current_precision = float('nan')

            student_recall_list.append(student_current_recall)
            student_precision_list.append(student_current_precision)

        student_mean_recall = np.mean(student_recall_list)
        student_mean_precision = np.nanmean(student_precision_list)
        if self.teacher_network is not None:
            for i in set(teacher_available_list):
                teacher_current_recall = teacher_class_correct[i] / teacher_class_total_target[i]

                if teacher_class_total_pred[i] != 0:
                    teacher_current_precision = teacher_class_correct[i] / teacher_class_total_pred[i]
                else:
                    teacher_current_precision = float('nan')

                teacher_recall_list.append(teacher_current_recall)
                teacher_precision_list.append(teacher_current_precision)

            teacher_mean_recall = np.mean(teacher_recall_list)
            teacher_mean_precision = np.nanmean(teacher_precision_list)

            return student_mean_recall, student_mean_precision, student_recall_list, student_precision_list, teacher_mean_recall, teacher_mean_precision, teacher_recall_list, teacher_precision_list

        return student_mean_recall, student_mean_precision, student_recall_list, student_precision_list, None, None, None, None

    def write_to_tensorboard(self, epoch, writer, annotation_converter, student_mean_recall, student_mean_precision, student_recall_list, \
        student_precision_list, teacher_mean_recall = None, teacher_mean_precision = None, teacher_recall_list = None, teacher_precision_list = None):
        """
        Helper function to write test results into tensorboard instance.
        Arguments:
            epoch: the actual epoch to be written in tensorboard
            writer: tensorboard writer instance
            annotation_converter: converter to convert class index into actual class
            student_mean_recall: mean recall of student model
            student_mean_precision: mean precision of student model
            student_recall_list: list of student recall based on the class index
            student_precision_list: list of student precision based on the class index
            teacher_mean_recall: mean recall of student model, None if no teacher is used
            teacher_mean_precision: mean precision of student model, None if no teacher is used
            teacher_recall_list: list of student recall based on the class index, None if no teacher is used
            teacher_precision_list: list of student precision based on the class index, None if no teacher is used
        """
        for i in range(len(student_recall_list)):
            if annotation_converter is not None:
                self.writer.add_scalar("recall_class_" + str(i) + annotation_converter[i], student_recall_list[i], epoch + 1)
                self.writer.add_scalar("precision_class_" + str(i) + annotation_converter[i], student_precision_list[i], epoch + 1)
            else:
                self.writer.add_scalar("recall_class_" + str(i), student_recall_list[i], epoch + 1)
                self.writer.add_scalar("precision_class_" + str(i), student_precision_list[i], epoch + 1)

            self.logger.info("Recall for class " + str(i) + " in epoch " + str(epoch + 1) + " is " + str(student_recall_list[i]))
            self.logger.info("Precision for class " + str(i) + " in epoch " + str(epoch + 1) + " is " + str(student_precision_list[i]))

        self.writer.add_scalar("mean_recall_student", student_mean_recall, epoch + 1)
        self.writer.add_scalar("mean_precision_student", student_mean_precision, epoch + 1)
        self.logger.info("Mean recall student for epoch " + str(epoch+1) + " is " + str(student_mean_recall))
        self.logger.info("Mean precision student for epoch " + str(epoch+1) + " is " + str(student_mean_precision))

        if teacher_mean_recall is not None and teacher_mean_precision is not None and teacher_recall_list is not None and teacher_precision_list is not None:
            self.writer.add_scalar("mean_recall_teacher", teacher_mean_recall, epoch + 1)
            self.writer.add_scalar("mean_precision_teacher", teacher_mean_precision, epoch + 1)
            self.logger.info("Mean recall teacher for epoch " + str(epoch+1) + " is " + str(teacher_mean_recall))
            self.logger.info("Mean precision student for epoch " + str(epoch+1) + " is " + str(teacher_mean_precision))

    def save_model(self, epoch, student_network, quantization_framework, optimizer, criterion, annotation, location, teacher_network=None, scheduler=None, quantization_function=None):
        """
        Helper function to save the model in form of .pth file
        Arguments:
            epoch: the actual epoch to be saved in .pth file
            student_network: student model to be saved
            quantization_framework: whether PyTorch quantization framework is used and if used, also save quantized student model 
            optimizer: optimizer used in the training
            criterion: objective function of the training
            annotation: annotation converter to convert index into actual class and vice versa
            location: the location to store .pth file
            teacher_network: teacher network to be saved. None if teacher is not utilized
            scheduler: scheduler used in the training. None if it is not used
            quantization_function: quantization function used in the training. False if it is not used
        """
        save_dict = {}
        if quantization_framework is True:
            save_dict['float_student_model_state_dict'] = student_network.state_dict()
            quantized_student = torch.quantization.convert(student_network.to('cpu').eval(), inplace=False)
            save_dict['int_student_model_state_dict'] = quantized_student.state_dict()
            save_dict['fuse_module'] = self.config_file['train']['quantization']['fuse_module']
        else:
            save_dict['student_model_state_dict'] = student_network.state_dict()

        if teacher_network is not None:
            save_dict['teacher_model_state_dict'] = teacher_network.state_dict()

        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()

        save_dict['epoch'] = epoch
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        save_dict['loss'] = criterion
        save_dict['annotation'] = annotation

        torch.save(save_dict, location)

class BenchmarkWrapper():
    """
    Wrapper for testing model using test split of Drive&Act.
    """
    def __init__(self, network, num_classes, config_file, logger, annotation_converter, quantization_framework=False):
        """
        Arguments:
            network: model to be tested
            num_classes: number of classes in the test dataset
            config_file: configuration file of the test
            logger: logger instance
            annotation_converter: annotation converter to convert index into actual class and vice versa
            quantization_framework: True if PyTorch quantization framework is used
        """
        self.network = network
        self.num_classes = num_classes
        self.config_file = config_file
        self.logger = logger
        self.annotation_converter = annotation_converter
        self.quantization_framework = quantization_framework

    def benchmark(self, test_loader):
        """
        This method represents measurement of quality metrics defined in the thesis.
        Arguments:
            test_loader: data loader of test data
        """
        if parse_version(torch.__version__) >= parse_version("0.4.0"):
            torch.set_grad_enabled(False)
            print("Testing. Pytorch version >= 0.4.0")
        else:
            print("Testing. Pytorch version < 0.4.0")

        
        self.network.eval()

        network_class_correct = [0. for i in range(self.num_classes)]
        network_class_incorrect = [0. for i in range(self.num_classes)]
        network_class_total_target = [0. for i in range(self.num_classes)]
        network_class_total_pred = [0. for i in range(self.num_classes)]
        network_available_list = []

        for data in test_loader:
            inputs, target = data

            if torch.cuda.is_available() and self.quantization_framework is False:
                inputs, target = Variable(inputs.cuda(),volatile=True), Variable(target.cuda(),volatile=True)
                self.network.cuda()
            else:
                inputs, target = Variable(inputs), Variable(target)

            with torch.no_grad():
                if torch.cuda.is_available() and self.quantization_framework is False:
                    network_outputs = self.network(inputs).cuda()
                else:
                    network_outputs = self.network(inputs)

            softmaxFunction = nn.Softmax(dim=1)

            _, network_pred = torch.max(softmaxFunction(network_outputs), 1)

            network_correct = [1 if current_pred == current_target else 0 for current_pred, current_target in zip(target.data, network_pred)]
            network_incorrect = [1 if current_pred != current_target else 0 for current_pred, current_target in zip(target.data, network_pred)]

            for i in range(len(target.data)):
                network_current_target = int(target.data[i])
                network_current_prediction = int(network_pred[i])
                #add 1 if the target is classified right
                network_class_correct[network_current_target] += network_correct[i]
                #add 1 if the prediction is false
                network_class_incorrect[network_current_prediction] += network_incorrect[i]
                network_class_total_pred[network_current_prediction] += 1
                network_class_total_target[network_current_target] += 1
                network_available_list.append(network_current_target)

        network_recall_list = []
        network_precision_list = []

        for i in set(network_available_list):
            network_current_recall = network_class_correct[i] / network_class_total_target[i]

            if network_class_total_pred[i] != 0:
                network_current_precision = network_class_correct[i] / network_class_total_pred[i]
            else:
                network_current_precision = float('nan')

            network_recall_list.append(network_current_recall)
            network_precision_list.append(network_current_precision)

        network_mean_recall = np.mean(network_recall_list)
        network_mean_precision = np.nanmean(network_precision_list)

        self.logger.info("Mean recall: " + str(network_mean_recall))
        self.logger.info("Mean precision: " + str(network_mean_precision))

        for i in range(len(network_recall_list)):
            self.logger.info("Mean recall for class " + str(i) + " (" + str(self.annotation_converter[i]) + ")" + ": " + str(network_recall_list[i]))
            self.logger.info("Mean precision for class " + str(i) + " (" + self.annotation_converter[i] + ")" + ": " + str(network_precision_list[i]))

        softmaxFunction = nn.Softmax(dim=1)

        torch.save(self.network.state_dict(), "temp.p")
        self.logger.info('Size (MB): ' + str(os.path.getsize("temp.p")/1e6))
        os.remove('temp.p')

        inputs_batch, target_batch = next(iter(test_loader))

        inputs = inputs_batch[0]
        target = target_batch[0]

        inputs = inputs[None,:,:,:,:]


        inputs_cpu, targets_cpu = Variable(inputs), Variable(target)
        inputs_gpu, targets_gpu = Variable(inputs.cuda(),volatile=True), Variable(target.cuda(),volatile=True)

        self.network.to('cpu')

        with torch.autograd.profiler.profile() as prof:
            _ = softmaxFunction(self.network(inputs_cpu))

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        print("Total CPU time: ", str(prof.self_cpu_time_total))

        with torch.autograd.profiler.profile() as prof:
            for i in range(1000):
                _ = softmaxFunction(self.network(inputs_cpu))

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        print("Total CPU time 1000 pass: ", str(prof.self_cpu_time_total))

        if self.quantization_framework is not True:
            self.network.cuda()
            with torch.no_grad():
                torch.cuda.synchronize()
                for _ in range(20):
                    _ = self.network(inputs_gpu)
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = softmaxFunction(self.network(inputs_gpu))
                end_event.record()

                torch.cuda.synchronize()

                self.logger.info("GPU inference time (ms): " + str(start_event.elapsed_time(end_event)))






