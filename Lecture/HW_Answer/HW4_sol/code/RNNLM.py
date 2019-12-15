import sys
import time
import os

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import sequence_loss
from model import LanguageModel

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  ### YOUR CODE HERE
  batch_size = 64
  embed_size = 512
  hidden_size = 2048
  num_steps = 10 # RNN is unfolded into 'num_steps' time steps
  max_epochs = 50
  early_stopping = 10
  dropout = 0.9
  lr = 0.001
  ### END YOUR CODE

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible.

    Adds following nodes to the computational graph.

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout rate placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps])
    self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.num_steps])
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE
  
  def add_embedding(self):
    """Add embedding layer.

    Hint: You might find tf.nn.embedding_lookup useful.

    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs.

    Hint: Here is the dimension of the variables (embedding matrix) you will need to create:

      embedding: (len(self.vocab), embed_size) corresponding to L in HW4.

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    ### YOUR CODE HERE
    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
      embed_param = tf.get_variable('param', shape=[len(self.vocab), self.config.embed_size], trainable=True)
    embedding = tf.nn.embedding_lookup(embed_param, self.input_placeholder)
    inputs = list(tf.split(embedding, self.config.num_steps, axis=1))
    inputs = [tf.squeeze(x, axis=[1]) for x in inputs]

    ### END YOUR CODE

    return inputs

  def add_model(self, inputs):
    """Creates the RNN language model.

    Implement the equations for the RNN language model.
    Note that you may NOT use built in rnn_cell functions from tensorflow.

    Hint: Make sure to apply dropout to both the inputs and the outputs.
          How to do it for inputs has been provided.

    Hint: Use variable_scope to make it clear for each layer.
          (Names already given. You can change the given code but please keep the name.)

    Hint: Use the variable scope "RNN" to define RNN variables to enable weight sharing.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. You might find tf.zeros useful.

          Add this to self as instance variable

          self.initial_state
  
          (Don't change variable name)

    Hint: Add the last RNN output to self as instance variable

          self.final_state

          (Don't change variable name)

    Hint: To implement RNN, you need to perform an explicit for-loop over inputs.
          Read the documentation of tf.variable_scope to see how to achieve
          weight sharing.

    Hint: Here are the dimensions of the various variables you will need to
          create:
      
          RNN_H: (hidden_size, hidden_size) corresponding to H in HW4.
          RNN_I: (embed_size, hidden_size) corresponding to I in HW4.
          RNN_b: (hidden_size,) corresponding to b1 in HW4.

          (Don't change variable name)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    with tf.variable_scope('InputDropout'):
      inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

    ### YOUR CODE HERE
    time_steps = len(inputs)
    batch_size, embed_size, hidden_size = self.config.batch_size, self.config.embed_size, self.config.hidden_size
    with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE) as scope:
      self.initial_state = tf.zeros([batch_size, hidden_size])
      RNN_H = tf.get_variable('RNN_H', shape=[hidden_size, hidden_size], trainable=True)
      RNN_I = tf.get_variable('RNN_I', shape=[embed_size, hidden_size], trainable=True)
      RNN_b = tf.get_variable('RNN_b', shape=[hidden_size], trainable=True)

      h = self.initial_state
      rnn_outputs = []
      for item in inputs:
        h = tf.nn.sigmoid(tf.matmul(h, RNN_H) + tf.matmul(item, RNN_I) + RNN_b)
        rnn_outputs.append(h)
      self.final_state = h

    with tf.variable_scope('RNNDropout'):
      rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]
      
    ### END YOUR CODE

    return rnn_outputs

  def add_projection(self, rnn_outputs):
    """Adds a projection/output layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Use variable_scope to make it clear for each layer.
          (Names already given. You can change the given code but please keep the name.)

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          W: (hidden_size, len(vocab)) corresponding to U in HW4.
          b: (len(vocab),) corresponding to b2 in HW4.

          (Don't change variable name)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab))
    """
    ### YOUR CODE HERE
    with tf.variable_scope('Projection', reuse=tf.AUTO_REUSE):
      W = tf.get_variable('W', shape=[self.config.hidden_size, len(self.vocab)], trainable=True)
      b = tf.get_variable('b', shape=[len(self.vocab)], trainable=True)
      outputs = []
      for rnn_output in rnn_outputs:
        output = tf.matmul(rnn_output, W) + b
        outputs.append(output)

    ### END YOUR CODE

    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.contrib.legacy_seq2seq.sequence_loss to implement sequence loss. 

-----------------------------Info for tensorflow.contrib.legacy_seq2seq.sequence_loss----------------
def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".
Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
----------------------------------------------------------------------------------------------------

    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    logit_list = list(tf.split(output, self.config.num_steps, axis=0))
    targets = list(tf.split(self.labels_placeholder, self.config.num_steps, axis=1))
    targets = [tf.cast(tf.squeeze(x), tf.int32) for x in targets]
    weights = [tf.ones([self.config.batch_size]) for _ in range(self.config.num_steps)]
    loss = tf.contrib.legacy_seq2seq.sequence_loss(logit_list, targets, weights)
    ### END YOUR CODE

    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
    ### END YOUR CODE

    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
  
    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    output = tf.reshape(tf.concat(self.outputs, axis=1), [-1, len(self.vocab)])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)

  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model. Note that batch_size and num_steps are both 1.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)
  Hint: Dropout rate should be 1 for this work.

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in range(stop_length):
    feed = {
      model.input_placeholder: [tokens[-1:]],
      model.initial_state: state,
      model.dropout_placeholder: 1
    }
    state, y_pred = session.run([model.final_state, model.predictions[-1]], feed_dict=feed)
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM', reuse=tf.AUTO_REUSE) as scope:
    model = RNNLM_Model(config)
    # Set reuse=tf.AUTO_REUSE instructs gen_model to reuse the same variables as the model above
    gen_model = RNNLM_Model(gen_config)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    for epoch in range(config.max_epochs):
      print('Epoch {}'.format(epoch))
      start = time.time()
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print('Training perplexity: {}'.format(train_pp))
      print('Validation perplexity: {}'.format(valid_pp))
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print('Total time: {}'.format(time.time() - start))
      
    saver.restore(session, 'ptb_rnnlm.weights')
    test_pp = model.run_epoch(session, model.encoded_test)
    print('=-=' * 5)
    print('Test perplexity: {}'.format(test_pp))
    print('=-=' * 5)
    starting_text = 'in palo alto'
    while starting_text:
      print(' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0)))
      starting_text = input('> ')

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '8'
  test_RNNLM()

