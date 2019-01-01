'''
Created on Jan 1, 2019

@author: mofir
'''

def TfLSTM_RNN(_X, _weights, _biases,features_num,num_of_periods,hidden):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    #_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    #_X = tf.reshape(_X, [-1, features_num])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
 #   _X = tf.nn.relu(_X)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    #_X = tf.split(_X, num_of_periods*features_num, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.relu)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.relu)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output TODO - why static rnn?
    outputs, states = tf.nn.dynamic_rnn(lstm_cells,_X,dtype=tf.float32) #tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    #

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-num_of_periods]
    print(outputs)

    # Linear activation
    return outputs,lstm_last_output     #tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def TfConstructNetworkArch_LSTM(model_params):
    # Training

    learning_rate = 0.001
    lambda_loss_amount = 0.0015
    training_iters = 1000  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training

    num_of_periods = model_params.num_of_periods
    features_num = model_params.feature_num

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, num_of_periods*features_num,1])
    y = tf.placeholder(tf.float32, [None, num_of_periods, 1])

    hidden = 100

# Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_uniform([features_num, hidden], minval=0.01)), # Hidden layer weights
        'out': tf.Variable(tf.random_uniform([hidden, num_of_periods], minval=0.01))
        }
    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden])),
        'out': tf.Variable(tf.random_normal([num_of_periods]))
    }

    rnn_output,pred_lstm = LSTM_RNN(x, weights, biases,features_num,num_of_periods,hidden)

    stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
    print("stacked_rnn_output: ")
    print(stacked_rnn_output)

    stacked_outputs    = tf.layers.dense(stacked_rnn_output, 1)
    print("stacked_outputs: ")
    print(stacked_outputs)

    stacked_outputs = tf.shape(tf.squeeze(stacked_outputs))

    outputs = tf.reshape(stacked_outputs, [-1,num_of_periods,rnn_output])

    # Loss, optimizer and evaluation
    #l2 = lambda_loss_amount * sum(
    #tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
   #) # L2 loss prevents this overkill neural network to overfit the data
    #tf.reshape(y,[-1,num_of_periods])
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=outputs)) + l2 # Softmax loss
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
    #training_op = optimizer

    cost = tf.reduce_sum(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(cost)

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    with tf.Session() as sess:
        init.run()
        for ep in range(training_iters):
            sess.run(training_op, feed_dict={x: x_train,y: y_train})
            if ep % 100 == 0:
                mse = cost.eval(feed_dict = {x: x_train,y: y_train})
                #print(ep,"\tMSE:",mse)
        y_pred = sess.run(outputs,feed_dict={x: x_ho_data})

    return y_pred

def TfConstructNetworkArch_Simple(model_params):
    num_of_periods = model_params.num_of_periods
    features_num   = model_params.feature_num
    learning_rate  = model_params.learning_rate
    hidden         = model_params.hidden_layer_size

    ##create the tensor flow model
    inputs = 1
    output = 1
    x = tf.placeholder(tf.float32, [None,num_of_periods, features_num])
    y = tf.placeholder(tf.float32, [None,num_of_periods, output])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
    init_state = basic_cell.zero_state(num_of_periods*features_num, tf.float32)
    rnn_output, final_state = tf.nn.dynamic_rnn(basic_cell,x,dtype=tf.float32)#, initial_state=init_state)

    stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
    stacked_outputs    = tf.layers.dense(stacked_rnn_output, output)
    outputs = tf.reshape(stacked_outputs, [-1,num_of_periods,output])

    #print(outputs)
    #print(y)

    loss        = tf.reduce_sum(tf.square(outputs - y))
    optimizer   = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    return dict(
        x = x,
        y = y,
        init_state  = init_state,
        final_state = final_state,
        outputs     = outputs,
        loss        = loss,
        training_op = training_op
    )

def TfTrainNetwork(NetworkGraph, Data, num_epochs, num_steps = 200, batch_size = 32, save=False):

    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(num_epochs):
            sess.run(NetworkGraph['training_op'], feed_dict={NetworkGraph['x']: Data['x_train'],NetworkGraph['y']: Data['y_train']})
            if ep % 100 == 0:
                mse = NetworkGraph['loss'].eval(feed_dict = {NetworkGraph['x']: Data['x_train'],NetworkGraph['y']: Data['y_train']})
                logging.debug("epoch num: " + str(ep) + "MSE: " + str(mse))

        #hold-out data TODO - move to seperate hold out function
        y_pred = sess.run(NetworkGraph['outputs'],feed_dict={NetworkGraph['x']: Data['x_ho_data']})

    return y_pred
