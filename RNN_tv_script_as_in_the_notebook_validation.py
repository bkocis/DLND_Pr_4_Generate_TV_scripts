
# Added regularization 


import problem_unittests as tests
from collections import Counter

import problem_unittests as tests
import torch

import helper

from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import torch.nn as nn

data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punct = {'.': '||Period||',
             ',': '||Comma||',
             '"': '||Quotation_Mark||',
             ';': '||Semicolon||',
             '!': '||Exclamation_Mark||',
             '?': '||Question_Mark||',
             '(': '||Left_Parentheses||',
             ')': '||Right_Parentheses||',
             '-': '||Dash||',
            '\n': '||Return||'
            }
    return punct




def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    #words  = set(text)
    #word_counts = Counter(words)
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}
    
    #words = utils.preprocess(text)
    #vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    #int_words = [vocab_to_int[word] for word in words]
    
    # return tuple
    return (vocab_to_int, int_to_vocab)



# Checkpoint 
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()



# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


"""
def batch_data(words, sequence_length, batch_size):
    n_batches = len(words)//batch_size
    # only full batches
    words = words[:n_batches*batch_size]
    y_len = len(words) - sequence_length
    x, y = [], []
    for idx in range(0, y_len):
        idx_end = sequence_length + idx
        x_batch = words[idx:idx_end]
        x.append(x_batch)
#         print("feature: ",x_batch)
        batch_y =  words[idx_end]
#         print("target: ", batch_y)    
        y.append(batch_y)    

    # create Tensor datasets
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    # make sure the SHUFFLE your training data
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)
    # return a dataloader
    return data_loader    
"""

from torch.utils.data.sampler import SubsetRandomSampler

# batch_data with train and validation DataLoarders
def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_batches = len(words)//batch_size
    # only full batches
    words = words[:n_batches*batch_size]
    y_len = len(words) - sequence_length
    x, y = [], []
    for idx in range(0, y_len):
        idx_end = sequence_length + idx
        x_batch = words[idx:idx_end]
        x.append(x_batch)
        batch_y =  words[idx_end]
        y.append(batch_y) 

	## split the data loadr to train and validation datasets ##

	# obtain training indices that will be used for validation
    num_workers = 0

	# percentage of training set to use as validation
    valid_size = 0.2

    num_train = len(words)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

	# define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    # create Tensor datasets
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))

	# prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)





#    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)
    # return a dataloader
    return (train_loader, valid_loader)







class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # TODO: Implement function
        # set class variables
        self.vocal_size = len(set(text))
        self.output_size = output_size
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #self.embedding_dim = embedding_dim
        
        # define model layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        #self.lstm = nn.LSTM( vocab_size, hidden_dim, n_layers, dropout = dropout, batch_first=True)
        self.lstm = nn.LSTM( embedding_dim, hidden_dim, n_layers, dropout = dropout, batch_first=True)

    #    self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function  
        batch_size = nn_input.size(0)
        #print(nn_input.size())
        
        embeds = self.embeddings(nn_input)
        
        #1 lstm layer
        lstm_output, hidden = self.lstm(embeds, hidden)

        #2 transfrom/flatten
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        
        #3 dropout
   #     out = self.dropout(lstm_output)
        #out = out.contiguous().view(-1, self.output_size)

        #4 fully connected layer 
  #      out = self.fc(out)
        out = self.fc(lstm_output)
        
        # reshape into (batch_size, seq_length, output_size)
        out = out.view( batch_size, -1, self.output_size)
        # get last batch
        out = out[:, -1]
        
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        if torch.cuda.is_available:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weigth.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden



def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    if torch.cuda.is_available:
        rnn.cuda()
        inputs = inp.cuda()
        target = target.cuda()
    
    # perform backpropagation and optimization
    hs = tuple([i.data for i in hidden])
    #1. 
    rnn.zero_grad()
    

    
    #2. calculate outputs of the network

    output, hs = rnn(inputs, hs)
    
    #3.loss function
    loss = criterion(output,target)
    
    #4. BackPropagate
    loss.backward()
    
    # 'gradient clipping'
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hs


from torch.utils.data.sampler import SubsetRandomSampler


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    valid_loss_min = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):


        train_loss =0.0
        valid_loss =0.0
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)


        rnn.train()
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

        # VALIDATION # 
        rnn.eval()
        for batch_i, (inputs, labels) in enumerate(valid_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(valid_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Validation loss Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # save the model
        if valid_loss <= valid_loss_min:
            print("Validation loss decreased, saving model...".format(valid_loss_min, valid_loss))
            valid_loss_min = valid_loss

    # returns a trained rnn
    return rnn

### HYPERPARAMETERS ####

# Data params
# Sequence Length
sequence_length = 8  # of words in a sequence
# Batch Size
batch_size = 64

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)[0]
valid_loader = batch_data(int_text, sequence_length, batch_size)[1]


# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = len(vocab_to_int)
# Embedding Dimension
embedding_dim = 300
# Hidden Dimension
hidden_dim = 300
# Number of RNN Layers
n_layers = 4

# Show stats for every n number of batches
show_every_n_batches = 1000


train_on_gpu = torch.cuda.is_available()



## TRAINING ###

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn_regularization', trained_rnn)
print('Model Trained and Saved')





# checkpoint 
_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')


# GENERATE TV SCRIPTS

import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        #current_seq = np.roll(current_seq, -1, 1)
        # needed to modigy this line here 
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences



# run the cell multiple times to get different results!
gen_length = 400 # modify the length to your preference
prime_word = 'jerry' # name for starting the script


pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
#print(generated_script)

f =  open("generated_script_1.txt","w")
f.write(generated_script)
f.close()
