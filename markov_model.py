def update_progress(progress):
    """Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    """
    barLength = 25 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
class MarkovChainModel(object):
    """Generates a prediction based on pretrained markov matrices."""
    
    def __init__(
            self, 
            langs, 
            order=1, 
            prediction_cutoff=100, 
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
        for i, true_byte in enumerate(byte_string):
            #reference dict to get byte index
            try:
                byt = class_dict[str(true_byte)]
            except KeyError:
                byt = class_dict['other']
            if not any(np.array(prev) < 0):
                loc = tuple([slice(None)] + prev + [byt])
                scores += self.models[loc].todense()
            for j, val in enumerate(prev):
                if j == (len(prev) - 1):
                    prev[j] = byt
                else:
                    prev[j] = prev[j+1]
            if i > self.prediction_cutoff:
                if (scores.max() - scores.mean()) > self.order:
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
        scores_vec = self.softmax(scores)
        return scores_vec
    
    def predict_class(self, byte_string):
        """Makes prediction and returns class with highest probability.
        (string)
        """
        scores_vec = self.predict(byte_string)
        arg = np.argmax(scores_vec)
        return self.langs[arg]