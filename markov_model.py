import numpy as np
import pickle
import sparse

class MarkovChainModel(object):
    """Generates a prediction based on pretrained markov matrices."""
    
    def __init__(
            self, 
            langs, 
            order=1, 
            prediction_cutoff=.9, 
            base_path='./models/{}_o{}.pkl', 
            class_dict_path='./unique_bytes.npy'):
        self.langs = langs
        self.order = order
        self.prediction_cutoff=prediction_cutoff
        
        #Encode unique bytes
        unique_bytes = np.load(class_dict_path)
        byte_classes = dict()
        for i, byt in enumerate(unique_bytes):
            #correct off-by-one issue
            byt = int(byt)
            byt -= 1
            byt = str(byt)
            byte_classes[byt] = i
        #Add 'other' class in case unexpected byte encountered.
        byte_classes['other'] = i + 1
        self.class_dict = byte_classes
        
        print('loading models...')
        #order of models matches order of languages
        self.models = list()
        for lang in langs:
            path = base_path.format(lang, self.order)
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError('No model found for language: ', lang)
            self.models.append(model)
        self.models = sparse.stack(self.models, axis=0)
        print('Done!')
            
    def rolling_probs(self, byte_string, scores):
        """Gets non-normalized probability for a given byte
        string and second-order markov model."""
        class_dict = self.class_dict
        order = self.order
        prev = [-1 for i in range(order)]
        #initialize scores to nonzero values.
        scores = self.softmax(scores)
        for i, true_byte in enumerate(byte_string):
            #reference dict to get byte index
            try:
                byt = class_dict[str(true_byte)]
            except KeyError:
                byt = class_dict['other']
            if not any(np.array(prev) < 0):
                loc = tuple([slice(None)] + prev + [byt])
                arr = self.models[loc].todense()
                arr = self.softmax(arr)
                scores *= arr
                scores *= (1/scores.sum())
            for j, val in enumerate(prev):
                if j == (len(prev) - 1):
                    prev[j] = byt
                else:
                    prev[j] = prev[j+1]
            if scores.max() > self.prediction_cutoff:
                break
        return scores
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, byte_string):
        """Makes a prediction."""
        scores = np.zeros(self.models.shape[0])
        scores = self.rolling_probs(byte_string, scores)
        #take softmax of outputs
        return scores
    
    def decode(self, encoded_class):
        """Takes a number and returns a string with the corresponding class."""
        return self.langs[encoded_class]
    
    def encode(self, class_string):
        """Takes a string and converts it to its encoded value."""
        try:
            idx = self.langs.index(class_string)
        except ValueError:
            raise ValueError('Input string not found in class list.')
        return idx
    
    def predict_class(self, byte_string):
        """Makes prediction and returns class with highest probability.
        (string)
        """
        scores_vec = self.predict(byte_string)
        arg = np.argmax(scores_vec)
        return self.decode(arg)
    