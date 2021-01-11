import torch
import torch.optim as optim
import time
import random
from torch.autograd import Variable
from evaluation import *
import progressbar
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def train(params, training_docs, test_docs, data, model):
    if params['model_type'] == 'clique':
        training_data, training_labels = data.create_cliques(training_docs, params['task'], params['train_data_limit'])
        test_data, test_labels = data.create_cliques(test_docs, params['task'], params['train_data_limit'])
    elif params['model_type'] == 'sent_avg':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'sentence', params['task'], params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])
    elif params['model_type'] == 'par_seq':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])
    if USE_CUDA:
        model.cuda()
    if params['train_data_limit'] != -1:
        training_docs = training_docs[:10]
        test_docs = test_docs[:10]
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    if params['task'] == 'class' or params['task'] == 'perm' or params['task'] == 'minority':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()
    timestamp = time.time()
    best_test_acc = 0
    for epoch in range(params['num_epochs']):
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH "+str(epoch))
        total_loss = 0
        steps = int(len(training_data) / params['batch_size'])
        indices = list(range(len(training_data)))
        random.shuffle(indices)
        bar = progressbar.ProgressBar()
        model.train()
        for step in bar(range(steps)):
            batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
            sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'], params['clique_size'])
            batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'], params['clique_size'])
            model.zero_grad()
            pred_coherence = model(batch_padded, batch_lengths, original_index)
            if params['task'] == 'score_pred':
                loss = loss_fn(pred_coherence, Variable(FloatTensor(orig_batch_labels)))
            else:
                loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels)))
            mean_loss = loss / params["batch_size"]
            mean_loss.backward()
            total_loss += loss.cpu().data.numpy()
            optimizer.step()
        current_time = time.time()
        print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
        print("Train loss: " + str(total_loss[0]))
        output_name = params['model_name'] + '_epoch' + str(epoch)
        if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq':
            if params['task'] == 'minority':
                test_f05, test_precision, test_recall, test_loss = eval_docs(model, loss_fn, test_data, test_labels,
                                                                        data, params)
            elif params['task'] == 'class' or params['task'] == 'score_pred':
                test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)
            elif params['task'] == 'perm':
                test_accuracy, test_loss = eval_docs_rank(model, test_docs, data, params)
            print("Test loss: %0.3f" % test_loss)
            if params['task'] == 'score_pred':
                print("Test correlation: %0.5f" % (test_accuracy))
            elif params['task'] == 'minority':
                print("Test F0.5: %0.2f  Precision: %0.2f  Recall: %0.2f" % (test_f05, test_precision, test_recall))
            else:
                print("Test accuracy: %0.2f%%" % (test_accuracy * 100))
        elif params['model_type'] == 'clique':
            train_accuracy, train_loss = eval_cliques(model, loss_fn, training_data,
                                                                                            training_labels,
                                                                                            params['batch_size'],
                                                                                            params['clique_size'], data,
                                                                                            params['model_type'], params['task'])
            if params['task'] == 'score_pred':
                print("Train clique corr: %0.5f" % (train_accuracy))
            else:
                print("Train clique accuracy: %0.2f%%" % (train_accuracy * 100))
            test_clique_accuracy, test_loss = eval_cliques(model, loss_fn, test_data,
                                                                                            test_labels,
                                                                                            params['batch_size'],
                                                                                            params['clique_size'], data, params['model_type'], params['task'])
            print("Test loss: %0.3f" % test_loss)
            if params['task'] == 'score_pred':
                print("Test clique corr: %0.5f" % ((test_clique_accuracy)))
            else:
                print("Test clique accuracy: %0.2f%%" % ((test_clique_accuracy * 100)))
            doc_accuracy, test_precision, test_recall, test_f05 = eval_doc_cliques(model, test_docs, data, params)
            if params['task'] == 'score_pred':
                print("Test document corr: %0.5f" % (doc_accuracy))
            elif params['task'] == 'minority':
                print("Test F0.5: %0.2f  Precision: %0.2f  Recall: %0.2f" % (test_f05, test_precision, test_recall))
            else:
                print("Test document ranking accuracy: %0.2f%%" % (doc_accuracy * 100))
            test_accuracy = doc_accuracy
        if params['task'] == 'minority':
            if test_f05 > best_test_acc:
                best_test_acc = test_f05
                # save best model
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
        else:
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # save best model
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
        print()
    return best_test_acc


def test(params, test_docs, data, model):
    if params['model_type'] == 'clique':
        test_data, test_labels = data.create_cliques(test_docs, params['task'])
    elif params['model_type'] == 'sent_avg':
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])
    elif params['model_type'] == 'par_seq':
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])

    if USE_CUDA:
        model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    # output_name = params['model_name'] + '_test'
    if params['model_type'] == 'par_seq' or params['model_type'] == 'sent_avg':
        test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)
        print("Test accuracy: %0.2f%%" % (test_accuracy * 100))
    elif params['model_type'] == 'clique':
        doc_accuracy = eval_doc_cliques(model, test_docs, data, params)
        print("Test document ranking accuracy: %0.2f%%" % (doc_accuracy * 100))
