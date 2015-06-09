from distributedwordreps import *

class DeepNeuralNetwork:
    def __init__(self, input_dim=0, hidden_dim=0, hidden_layer=1, output_dim=0,
                 afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):
        self.afunc = afunc
        self.d_afunc = d_afunc
        self.input = np.ones(input_dim+1)   # +1 for the bias
        self.hidden = np.array([np.ones(hidden_dim+1)
                                for i in range(hidden_layer)]) # +1 for the bias
        self.output = np.ones(output_dim)
        self.iweights = randmatrix(input_dim+1, hidden_dim)
        self.hweights = np.array([randmatrix(hidden_dim+1, hidden_dim)
                                  for i in range(hidden_layer-1)]) # +1 for bias
        self.oweights = randmatrix(hidden_dim+1, output_dim)
        self.oerr = np.zeros(output_dim)
        self.ierr = np.zeros(input_dim+1)
        self.hidden_layer = hidden_layer
        self.error = [np.zeros(hidden_dim+1) for i in range(hidden_layer)]

    def forward_propagation(self, ex):
        self.input[ : -1] = ex # ignore the bias
        self.hidden[0][:-1] = self.afunc(dot(self.input, self.iweights))
        for i in range(1, self.hidden_layer):
        	self.hidden[i][:-1] =  self.afunc(dot(self.hidden[i-1],
                                              self.hweights[i-1]))# ignore the bias
        self.output = self.afunc(dot(self.hidden[self.hidden_layer-1],
                                 self.oweights))
        return copy.deepcopy(self.output)

    def backward_propagation(self, labels, alpha=0.5):
        labels = np.array(labels)
        self.oerr = (labels-self.output) * self.d_afunc(self.output)

        tmp = self.hidden_layer-1
        self.error[tmp] = dot(self.oerr, self.oweights.T)
                          *self.d_afunc(self.hidden[tmp])
        for i in range(-self.hidden_layer+2, 1):
            self.error[-i] = dot(self.error[-i+1][:-1],
                                 self.hweights[-i].T)*self.d_afunc(self.hidden[-i])

        self.oweights += alpha * outer(self.hidden[self.hidden_layer-1], self.oerr)
        for i in range(self.hidden_layer-1):
            self.hweights[i] += alpha*outer(self.hidden[i], self.error[i+1][:-1])
        self.iweights += alpha * outer(self.input, self.error[0][:-1])
        return np.sum(0.5 * (labels-self.output)**2)

    def train(self, training_data, maxiter=5000, alpha=0.05,
              epsilon=1.5e-8, display_progress=True):
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)
            if display_progress and iteration % 10 == 0:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1

    def fit(self, ex, labels):
        self.train(zip(ex, labels), display_progress=True)

    def predict(self, ex):
        if ex.ndim == 2:
            return np.hstack((np.argmax(self.predict(x)) for x in ex))
        return self.forward_propagation(ex)

    # def hidden_representation(self, ex):
    #     self.forward_propagation(ex)
    #     return self.hidden
def iff_example():
    iff_train = [
        ([1.,1.], [1.]), # T T ==> T
        ([1.,0.], [0.]), # T F ==> F
        ([0.,1.], [0.]), # F T ==> F
        ([0.,0.], [1.])] # F F ==> T
    net = DeepNeuralNetwork(input_dim=2, hidden_dim=4, hidden_layer=4,
                            output_dim=1)
    net.train(copy.deepcopy(iff_train))
    for ex, labels in iff_train:
        prediction = net.predict(ex)
        # hidden_rep = net.hidden_representation(ex)
        print ex, labels, np.round(prediction, 2)#, np.round(hidden_rep, 2)

iff_example()
