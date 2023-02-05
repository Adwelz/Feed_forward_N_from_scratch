import numpy as np

"""
:param z: outputs of the layer
:param ft_derivative: derivation of the transfert fonction
:return: Jacobian of the effect of the inputs sum upon z
"""
def jacobian_z_sum_function(z, ft_derivative):
    L = ft_derivative(z)
    if L.ndim == 0:
        return L
    else:
        return np.diag(L)


"""
:param z: outputs of the layer
:param ft_derivative: derivation of the transfert fonction
:param w: weights of the layer
:return: Jacobian of the effect of y upon z
"""
def jacobian_z_y_function(z, ft_derivative, w):
    return np.dot(jacobian_z_sum_function(z, ft_derivative), w.T)


"""
:param y: inputs of the layer
:param z: outputs of the layer
:param ft_derivative: derivation of the transfert fonction
:return: Jacobian of the effect of the weights upon z
"""
def simplified_jacobian_z_w_function(y, z, ft_derivative):
    j_z_sum = jacobian_z_sum_function(z, ft_derivative)

    if j_z_sum.ndim == 0:
        return np.outer(y, jacobian_z_sum_function(z, ft_derivative))
    else:
        return np.outer(y, np.diag(jacobian_z_sum_function(z, ft_derivative)))


"""
:param x: input of the relu
:return: relu output
"""
def relu(x):
    return x * (x > 0)


"""
:param x: input of the derivative of relu
:return: derivative of relu output
"""
def derivative_of_relu(x):
    return 1. * (x > 0)


class Layer:

    """
    :param self: the layer
    :param inputs_size: number of inputs that the layer receive
    :param number_of_nodes: number of nodes in the layer
    :param transfer_function: the activation fonction of the layer
    :param wr: weights range
    :param lr: weights learning rate
    :param br: biais range
    :param b_lr: biais learning rate
    :param batch_size: lenght of the batch 
    :return: the layer with the given parameters
    """
    def __init__(self, inputs_size, number_of_nodes, transfer_function, wr, lr, br, b_lr) -> None:

        # set the inputs size
        self.inputs_size = inputs_size

        # set the number of nodes
        self.number_of_nodes = number_of_nodes

        # set the transfer function
        if transfer_function == 'sigmoid':
            self.transfer_function = lambda x: (1+np.exp(-x))**(-1)
            self.transfer_function_derivative = lambda x: x*(1-x)

        if transfer_function == 'tanh':
            self.transfer_function = np.tanh
            self.transfer_function_derivative = lambda x: 1-x**2

        if transfer_function == 'relu':
            self.transfer_function = relu
            self.transfer_function_derivative = derivative_of_relu

        if transfer_function == 'linear':
            self.transfer_function = lambda x: x
            self.transfer_function_derivative = lambda x: 1

        # set the weights range
        wr_str = wr.split(' ')
        l_wr = []
        for s in wr_str:
            l_wr.append(float(s))
        self.wr = l_wr

        # set the weights with a random distribution on the given range
        self.weights = np.random.rand(
            self.inputs_size, self.number_of_nodes)*(self.wr[1]-self.wr[0])-(self.wr[1]-self.wr[0])/2

        # set the biais range
        br_str = br.split(' ')
        l_br = []
        for s in br_str:
            l_br.append(float(s))
        self.br = l_br

        # set the biais with a random distribution on the given range
        self.biais = np.random.rand(
            self.number_of_nodes)*(self.br[1]-self.br[0])-(self.br[1]-self.br[0])/2

        # set the weights learning rate
        self.lr = float(lr)

        # set the biais learning rate
        self.b_lr = float(b_lr)

    """
    :param self: the layer
    """
    def forward_pass(self,batch_size):
        B=np.ones((batch_size, self.number_of_nodes))*self.biais

        # set the outputs with this formula : F(X.W+B) where F is the activation function, X the inputs, W the weights and B the bias of the layer
        self.outputs = self.transfer_function(
            np.dot(self.inputs, self.weights)+B)

    """
    :param self: the layer
    :return Jacobian of the effect of the y upon the loss
    """
    def backward_pass(self, jacobian_l_z, wrt='none', wreg='0'):

        # if jacobian_l_z is a scalar
        if jacobian_l_z.ndim == 0:
            simp_j_z_w = simplified_jacobian_z_w_function(
                self.inputs, self.outputs, self.transfer_function_derivative)

            jacobian_z_y = jacobian_z_y_function(
                self.outputs, self.transfer_function_derivative, self.weights)

            jacobian_l_y = np.dot(jacobian_l_z, jacobian_z_y)

            # jacobian_l_w given the weights regularization type
            if wrt == 'none':
                jacobian_l_w = jacobian_l_z*simp_j_z_w

            if wrt == 'L2':
                jacobian_l_w = jacobian_l_z*simp_j_z_w + float(wreg)*self.weights

            if wrt == 'L1':
                jacobian_l_w = jacobian_l_z*simp_j_z_w + float(wreg)*np.sign(self.weights)

            # update the weights
            self.weights = self.weights-self.lr*jacobian_l_w

            # update the biais
            self.biais = self.biais-self.b_lr*jacobian_l_z

            return jacobian_l_y

        # if jacobian_l_z is a vector
        if jacobian_l_z.ndim == 1:
            simp_j_z_w = simplified_jacobian_z_w_function(
                    self.inputs[0], self.outputs[0], self.transfer_function_derivative)

            jacobian_z_y = jacobian_z_y_function(
                self.outputs[0], self.transfer_function_derivative, self.weights)

            jacobian_l_y = np.dot(jacobian_l_z, jacobian_z_y)

            # jacobian_l_w given the weights regularization type
            if wrt == 'none':
                jacobian_l_w = jacobian_l_z*simp_j_z_w

            if wrt == 'L2':
                jacobian_l_w = jacobian_l_z*simp_j_z_w+float(wreg)*self.weights

            if wrt == 'L1':
                jacobian_l_w = jacobian_l_z*simp_j_z_w+float(wreg)*np.sign(self.weights)

            for i in range(1,len(self.inputs)):
                simp_j_z_w = simplified_jacobian_z_w_function(
                    self.inputs[i], self.outputs[i], self.transfer_function_derivative)

                jacobian_z_y = jacobian_z_y_function(
                    self.outputs[i], self.transfer_function_derivative, self.weights)

                jacobian_l_y += np.dot(jacobian_l_z, jacobian_z_y)

                # jacobian_l_w given the weights regularization type
                if wrt == 'none':
                    jacobian_l_w += jacobian_l_z*simp_j_z_w

                if wrt == 'L2':
                    jacobian_l_w += jacobian_l_z*simp_j_z_w+float(wreg)*self.weights

                if wrt == 'L1':
                    jacobian_l_w += jacobian_l_z*simp_j_z_w+float(wreg)*np.sign(self.weights)
            
            jacobian_l_w=jacobian_l_w/len(self.inputs)

            jacobian_l_y=jacobian_l_y/len(self.inputs)

            # update the weights
            self.weights = self.weights-self.lr*jacobian_l_w

            # update the biais
            self.biais = self.biais-self.b_lr*jacobian_l_z

            return jacobian_l_y
