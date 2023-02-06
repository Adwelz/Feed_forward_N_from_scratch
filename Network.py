from Layer import Layer
import numpy as np


def jacobian_soft_function(s):
    """
    :param s: outputs of the softmax
    :return: Jacobian of the softmax function
    """
    J = []
    for i in range(len(s)):
        row = []
        for k in range(len(s)):
            if k == i:
                row.append(s[k]-s[k]**2)
            else:
                row.append(-s[k]*s[i])
        J.append(row)
    return J


def l2(weights):
    """
    :param weights: all the weights of the neural network
    :return: L2 regularization
    """
    return (1/2)*np.sum(weights**2)


def l1(weights):
    """
    :param weights: all the weights of the neural network
    :return: L1 regularization
    """
    return np.sum(np.abs(weights))


def softmax(m):
    """
    :param m: a matrix or a vector
    :return: a matrix wich each columns have the softmax function applied or a vector of the softmax function applied
    """
    if m.ndim == 2:
        A = []
        for row in m:
            S = 0
            for z in row:
                S += np.exp(z)
            A.append(np.exp(row)/S)
        return np.array(A)
    if m.ndim == 1:
        S = 0
        for scalar in m:
            S += np.exp(scalar)
        return np.exp(m)/S


# verbose option
verbose = False

if verbose:
    def verboseprint(*args):
        """
        :param *args: list of arguments
        :return: the function verboseprint
        """
        for arg in args:
            print(arg,)
else:
    verboseprint = lambda *a: None


class Network:
    def __init__(self, config) -> None:
        """
        :param self: the neural network
        :param config: configuration of the neural network
        :return: the neural network with the given configuration
        """
        # set the config
        self.config = config

        # get the name of the hidden layers
        hidden_layers = []
        for config_section in config.sections():
            if config_section.split('_')[0] == 'HIDDEN':
                hidden_layers.append(config_section)

        # set the hidden layers and the output layer in the case where there is not hidden layers in the configuration
        if hidden_layers == []:
            self.hidden_layers = []

            self.output_layer = Layer(int(config['INPUT_LAYER']['input']), int(config['OUTPUT_LAYER']['size']),
                                      config['OUTPUT_LAYER']['act'], config['OUTPUT_LAYER']['wr'], lr=config['OUTPUT_LAYER']['lrate'],
                                      br=config['OUTPUT_LAYER']['br'], b_lr=config['OUTPUT_LAYER']['b_lrate'])

        # set the hidden layers and the output layer in the case where there is hidden layers in the configuration
        else:
            self.hidden_layers = []
            self.hidden_layers.append(Layer(int(config['INPUT_LAYER']['input']), int(config[hidden_layers[0]]['size']),
                                            config[hidden_layers[0]]['act'], config[hidden_layers[0]
                                                                                    ]['wr'], config[hidden_layers[0]]['lrate'],
                                            br=config[hidden_layers[0]]['br'], b_lr=config[hidden_layers[0]]['b_lrate']))
            for i in range(1, len(hidden_layers)):
                self.hidden_layers.append(Layer(int(config[hidden_layers[i-1]]['size']), int(config[hidden_layers[i]]['size']),
                                                config[hidden_layers[i]]['act'], config[hidden_layers[i]
                                                                                        ]['wr'], config[hidden_layers[i]]['lrate'],
                                                br=config[hidden_layers[i]]['br'], b_lr=config[hidden_layers[i]]['b_lrate']))
            self.output_layer = Layer(int(config[hidden_layers[-1]]['size']), int(config['OUTPUT_LAYER']['size']),
                                      config['OUTPUT_LAYER']['act'], config['OUTPUT_LAYER']['wr'], lr=config['OUTPUT_LAYER']['lrate'],
                                      br=config['OUTPUT_LAYER']['br'], b_lr=config['OUTPUT_LAYER']['b_lrate'])

    def forward_pass(self, batch_length, inputs, targets):
        """
        :param self: the neural network
        :param batch_length: the length of the batch
        :param inputs: inputs of the neural network
        :param targets: targets of the neural network
        :return: the loss of the neural network
        """
        verboseprint('FORWARD_PASS :')

        # forward pass of the hidden layers
        if self.hidden_layers != []:
            self.hidden_layers[0].inputs = inputs
            for i in range(0, len(self.hidden_layers)-1):
                verboseprint('\n[HIDDEN_LAYER_'+str(i+1)+']')
                verboseprint('inputs :', self.hidden_layers[i].inputs)
                self.hidden_layers[i].forward_pass(batch_size=batch_length)
                verboseprint('outputs :', self.hidden_layers[i].outputs)
                self.hidden_layers[i+1].inputs = self.hidden_layers[i].outputs
            verboseprint('\n[HIDDEN_LAYER_'+str(len(self.hidden_layers))+']')
            verboseprint('inputs :', self.hidden_layers[-1].inputs)
            self.hidden_layers[-1].forward_pass(batch_size=batch_length)
            verboseprint('outputs :', self.hidden_layers[-1].outputs)

        # foward pass of the output layer
            self.output_layer.inputs = self.hidden_layers[len(
                self.hidden_layers)-1].outputs
        else:
            self.output_layer.inputs = inputs
        self.output_layer.forward_pass(batch_size=batch_length)

        verboseprint('\n[OUTPUT_LAYER]')
        verboseprint('inputs :', self.output_layer.inputs)
        verboseprint('outputs :', self.output_layer.outputs)

        # apply softmax to the output of the output layer
        if self.config['OUTPUT_LAYER']['softmax'] == 'True':
            self.output_layer.softmax_outputs = softmax(
                self.output_layer.outputs)

            verboseprint('\n[SOFTMAX]')
            verboseprint('inputs :', self.output_layer.outputs)
            verboseprint('outputs :', self.output_layer.softmax_outputs)

        # compute the loss
        verboseprint('\n[LOSS]')
        verboseprint('targets :', targets)
        if self.config['OUTPUT_LAYER']['softmax'] == 'True':
            loss_list = self.output_layer.softmax_outputs-targets
        else:
            loss_list = self.output_layer.outputs-targets
        verboseprint('loss_list :', loss_list)

        # case where loss_list is a scalar
        if loss_list.ndim == 0:
            if self.config['GLOBALS']['loss'] == 'MSE':
                loss = loss_list**2
            if self.config['GLOBALS']['loss'] == 'cross_entropy':
                loss = -targets*np.log(self.output_layer.softmax_outputs)

        # case where loss_list is a vector
        if loss_list.ndim == 1:
            loss = 0
            if self.config['GLOBALS']['loss'] == 'MSE':
                loss_list_squared = loss_list**2
                loss = loss_list_squared.mean()
                verboseprint('MSE :', loss)

            if self.config['GLOBALS']['loss'] == 'cross_entropy':
                loss = - \
                    np.sum(targets*np.log(self.output_layer.softmax_outputs))
                verboseprint('cross_entropy :', loss)

        # case where loss_list is a matrix
        if loss_list.ndim == 2:
            loss_array = loss_list
            loss_list = []

            if self.config['GLOBALS']['loss'] == 'MSE':
                for i in range(len(loss_array)):
                    loss_array_squared = loss_array**2
                    loss_list.append(loss_array_squared[i].mean())
                loss = np.array(loss_list).mean()
                verboseprint('MSE :', loss)

            if self.config['GLOBALS']['loss'] == 'cross_entropy':
                for i in range(len(loss_array)):
                    if self.config['OUTPUT_LAYER']['softmax'] == 'True':
                        loss_list.append(-np.sum(targets[i]*np.log(
                            self.output_layer.softmax_outputs[i])))
                    else:
                        loss_list.append(-np.sum(targets[i]*np.log(
                            self.output_layer.outputs[i])))

                loss = np.array(loss_list).mean()
                verboseprint('cross_entropy :', loss)

        # regularised loss
        weights_of_the_whole_neural_network = []
        for i in self.hidden_layers:
            weights_of_the_whole_neural_network.append(
                np.concatenate(i.weights))
        weights_of_the_whole_neural_network.append(
            np.concatenate(self.output_layer.weights))
        weights_of_the_whole_neural_network = np.concatenate(
            weights_of_the_whole_neural_network)

        if self.config['GLOBALS']['wrt'] == 'none':
            penalty = 0

        if self.config['GLOBALS']['wrt'] == 'L2':
            penalty = l2(weights_of_the_whole_neural_network)

        if self.config['GLOBALS']['wrt'] == 'L1':
            penalty = l1(weights_of_the_whole_neural_network)

        regularised_loss = loss+float(self.config['GLOBALS']['wreg'])*penalty
        return regularised_loss

    def backward_pass(self, targets):
        """
        :param self: the neural network
        :param targets: the targets of the batch
        """
        # case where softmax is applied to the output layer
        if self.config['OUTPUT_LAYER']['softmax'] == 'True':
            outputs = self.output_layer.softmax_outputs

        # case where softmax is not applied to the output layer
        else:
            outputs = self.output_layer.outputs

        loss_list = outputs-targets

        # compute MSE
        if self.config['GLOBALS']['loss'] == 'MSE':
            if loss_list.ndim == 0:
                jacobian_l_z = 2*loss_list
            if loss_list.ndim == 1:
                jacobian_l_z = (1/len(loss_list)
                                )*2*loss_list
            if loss_list.ndim == 2:
                jacobian_l_z = (1/len(loss_list[0]))*2*loss_list

        # compute cross entropy
        if self.config['GLOBALS']['loss'] == 'cross_entropy':
            jacobian_l_z = -targets/outputs

        # compute the jacobian with the effect of the output of the output layer on the loss for each outputs of the batch
        if self.config['OUTPUT_LAYER']['softmax'] == 'True':
            jacobian_l_z_list = []
            for i in range(len(outputs)):
                jacobian_soft = jacobian_soft_function(outputs[i])
                temp_jacobian_l_z = np.dot(jacobian_l_z[i], jacobian_soft)
                jacobian_l_z_list.append(temp_jacobian_l_z)
        else:
            jacobian_l_z_list = jacobian_l_z

        # backward pass of the output layer and get the jacobians of the effect of the inputs of the output layer on the loss
        jacobian_l_y_list = self.output_layer.backward_pass(
            jacobian_l_z_list, self.config['GLOBALS']['wrt'], self.config['GLOBALS']['wreg'])

        # recursive backward pass of each hidden layer and get the jacobians of the effect of the inputs of the hidden layer on the loss
        for hl in reversed(self.hidden_layers):
            jacobian_l_y_list = hl.backward_pass(jacobian_l_y_list)
