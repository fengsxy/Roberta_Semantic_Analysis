import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
#cal f1 score
from sklearn.metrics import f1_score

class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, val_loder,test_loder,criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()
        best_f1 = 0
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)
            if n_train % 1000 == 0:
                self.logger.info(f'iteration{n_train} [train] loss: {train_loss/n_train:.4f}, acc: {n_correct/n_train:.2f}')
                if n_train % 10000 == 0:
                    val_loss, val_acc,val_f1= self._test(val_loder, criterion)
                    test_loss, test_acc,test_f1 = self._test(test_loder, criterion)
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(self.model.state_dict(), f'{self.args.dataset}_{self.args.method}_best_model.pth')
                    self.logger.info(f'iteration{n_train} [val] loss: {val_loss:.4f}, acc: {val_acc:.2f} f1: {val_f1:.2f}')
                    self.logger.info(f'iteration{n_train}[test] loss: {test_loss:.4f}, acc: {test_acc:.2f} f1: {test_f1:.2f}')
                
        return train_loss / n_train, n_correct / n_train
    

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test,f1 = 0, 0, 0,0
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                preds.extend(torch.argmax(outputs['predicts'], -1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
                n_test += targets.size(0)
        f1 = f1_score(labels, preds, average='macro')
        return test_loss / n_test, n_correct / n_test,f1

    def run(self):
        train_dataloader, val_dataloader,test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader,val_dataloader,test_dataloader, criterion, optimizer)
            val_loss, val_acc,val_f1 = self._test(val_dataloader, criterion)
            test_loss, test_acc,test_f1 = self._test(test_dataloader, criterion)
      

            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc, best_loss = val_acc, val_loss
                if epoch >= 0:
                    torch.save(self.model.state_dict(), f'{self.args.dataset}_{self.args.method}best_model.pth')
                #torch.save(self.model.state_dict(), f'{self.args.dataset}_best_model.pth')

            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}'.format(test_loss, test_acc*100, val_f1))
            self.logger.info('[val] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}'.format(val_loss, val_acc*100, test_f1))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
        self.logger.info('log saved: {}'.format(self.args.log_name))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
