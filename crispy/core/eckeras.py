from .eclogging import load_logger
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Dense, Conv2D
from .ecf import *
from .emri import *
import pickle


logger = load_logger()

###
def get_model_memory_usage(batch_size, model):
    """
    https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    """

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0
            
    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


# Custom Callbacks to log elapsed time.

class CustomLogs(Callback):
    def on_epoch_begin(self, epoch, logs):
        self.time = time.time()
    
    def on_epoch_end(self, epoch, logs):
        hours, rem = divmod(time.time() - self.time, 3600)
        minutes, seconds = divmod(rem, 60)
        rs="{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        logs['elapsed_time'] = rs
        

### CrossValidation Logger
class CrossValidationLogger(Callback):
    def __init__(self, filename, fold_number, monitor='val_Dec_GT_Output_dice_coefficient', separator = ',', append = False):
        self.filename = filename
        self.fold_number = fold_number
        self.sep = separator
        self.append = append
        self.monitor = monitor
        
        self.set_dataframe()
        
    def set_dataframe(self):
        if os.path.isfile(self.filename):
            self.df = pd.read_csv(self.filename, index_col=0)
        else:
            self.df = pd.DataFrame(columns=[f"{self.fold_number}-fold"])
            self.df.index.name = f"epoch(monitor={self.monitor})"
        
    def on_epoch_end(self, epoch, logs):
        self.df.loc[epoch+1, f"{self.fold_number}-fold"] = logs[self.monitor]
        self.df.to_csv(self.filename)

        
### LogLogger
class LogLogger(Callback):
    """
    train_data : a list of [x_train, y_train]. e.g. [x_train, [y_train, x_train]]
    """
    
    def __init__(self, logger_object, evaluate_trainset_on_epoch_end = False, train_data = None, batch_size = 1):
        super().__init__()
        self.logger = logger_object
        self.evaluate_trainset_on_epoch_end = evaluate_trainset_on_epoch_end
        
        if evaluate_trainset_on_epoch_end:
            if train_data is None:
                raise ValueError("Train data should be given if you want to evaluate train set on end of epoch.")
            else:
                self.train_data = train_data
                self.batch_size = batch_size
                
    def on_epoch_end(self, epoch, logs):
        if self.evaluate_trainset_on_epoch_end:
            x_tr, y_tr = self.train_data
            train_eval_results = self.model.evaluate(x_tr, y_tr, verbose = 0, batch_size = self.batch_size)
        
            for i, n in enumerate(self.model.metrics_names):
                logs[n] = train_eval_results[i]
            
        self.logger.debug(f'''The result of {epoch+1} epoch: {', '.join((f"{k} = {v}" for k,v in logs.items()))}''')
        
        
### Freeze part of layers of a model.
def set_trainable_layers(model, layer_pattern, compile_parameters = {}):
    """
    model argument should be a single gpu model. If you have a multi-gpu model, pass the main model(e.g. model.layers[-3]) to model argument.
    
    You must compile the model again after running this function.
    
    Paramters
    ---------
    layer_patttern : a list of patterns of layers which you want to set trainable.
    
    Examples
    --------
    layer_pattern=['Dec_(?:VAE|GT)_128', 'Dec_(?:VAE|GT)_.*64', 'Dec_(?:GT|VAE)_.*32$', 'Dec_(?:GT|VAE)_Output', '^Dec_(?:GT|VAE)_Output']
    layer_pattern=['Dec_GT_128', 'Dec_GT_.*64', 'Dec_GT_.*32$', 'Dec_GT_Output']
    
    Returns
    -------
    rt : A dataframe have results of this function.
    
    """
    assert isinstance(layer_pattern, list), f"layer_pattern is not a list object, but {type(layer_pattern)}."
    logger.info(f"set_trainable_layers: layer name pattern = {layer_pattern}\n")
    rt = pd.DataFrame(columns = ['layer.name', 'match_result', 'layer.trainable'])
    
    for layer in model.layers:
        layer.trainable = False
        r=any(map(lambda x:re.search(x, layer.name) is not None, layer_pattern))
        if r:
            layer.trainable = True
        rt.loc[len(rt)] = [layer.name, r, layer.trainable]

    model.compile(**compile_parameters)
    logger.info(f"model was compiled.")
    if True in rt['layer.trainable'].value_counts():
        logger.info(f"{rt['layer.trainable'].value_counts()[True]} of {rt.index.size} layers are trainable.")
    else:
        logger.info(f"None of {rt.index.size} layers is trainable.")
    
#     print("You must compile the model again after running this function.")
    return rt


def load_weights_of_epoch(model, epoch_number, save_dir):
    weight_files = glob.glob(f"{save_dir}/*weight*epoch-{epoch_number:02d}*.h5")
    assert len(weight_files) == 1, "There are more than one weight files of the epoch number."
    weight_file = weight_files[0]
    model.load_weights(weight_file) ; logger.info(f"{weight_file} was loaded to the model.")
    
    
def evaluate_model(model, metric_func, trainset = None, valset = None, testset = None, epoch_number = None, batch_size = 2, save_dir = None):
    """
    Parameters
    ----------
    model : keras model object
    trainset : a trainset list which includes both x and y. (e.g. [x_train, [y_train, x_train]])
    valset : a valset list which includes both x and y. (e.g. [x_val, [y_val, x_val]])
    testset : a testset list which includes both x and y. (e.g. [x_val, [y_val, x_val]])
    
    Returns
    -------
    [trainset_prediction_output, valset_prediction_output, testset_prediction_output]
    [train_eval_results, val_eval_results, test_eval_results]
    
    """
    if epoch_number is not None:
        if save_dir is None: raise ValueError("save_dir argument should be passed when you specify an epoch_number.")
        load_weights_of_epoch(model, epoch_number, save_dir)
        model.save_weights("model_weights_temp.h5") ; logger.info("Orignial weights were backed up.")
    
    pl = [] # prediction list
    el = [] # evaluation list
    for n, d in zip(('train', 'validation', 'test'), (trainset, valset, testset)):
        logger.info(f"Evaluating the model with {n} set...")
        try:
            cpr = model.predict(d[0], batch_size = batch_size) # Current evaluation resultㄴ
            cer = [metric_func(yt, yp) for yt, yp in zip(d[1], cpr)]
            
            pl.append(cpr)
            el.append(cer)

        except Exception as e:
            pl.append([e.__class__.__name__] * 2)
            
            cer = [ -1 ] * len(d[1])
            el.append(cer)
            logger.error(f"{n} data was not processed properly. \n{e.__class__.__name__}: {e}")
        logger.debug(f"{n} evaluation results(GT, VAE):\n{cer}")
       
    if epoch_number is not None:
        model.load_weights("model_weights_temp.h5")
        logger.info("Orignial weights were restored.")
    
    return pl, el


### PruningCallbacks
class PruningCallback(Callback):
    """
    template_model argument should take model's template.
    
    Parameters
    ----------
    
    template_model : keras model
        a keras model to apply lottery hypothesis. This argument should take a template of a model(e.g. not relicated model for gpu).
    
    base_percents : dict
        Pruning thresholds of each layer type.
    
    save_dir : string
        A path to save outputs of this callback.
    
    n : int
        Use this argument when loading previous pruning results.
    
    Examples
    --------
    
    pc = PruningCallback(model, base_percents = {'Conv2D':0.2, 'Dense':0.15})
    
    model.fit(..., callbacks = [..., pc, ...], ...)
    pc.set_new_masks()
    pc.pruning_weights()
    pc.restore_initial_weights()
    
    
    # After getting a optimal result
    model.fit(..., callbacks = [..., pc, ...], ...)
    pc.set_new_masks()
    pc.pruning_weights()
    pc.restore_initial_weights()
    
    ...
            
    
    """
    
    def __init__(self, template_model, base_percents = {'Conv2D':0.2, 'Conv3D':0.2, 'Dense':0.15} , save_dir =".", layer_classes = None, masks=None, percents = None, **kwargs):
        self.model = template_model
        self.layer_classes = tuple(base_percents.keys())
        self.base_percents = base_percents
        self.layer_names_dict = self.make_layer_names_dict()# Currently, this class only supports keras.layers.Conv3D or Dense.
        self.weights = self.extract_weights(self.model)
        self.init_weights = self.model.get_weights()
        self.init_weights_filename = "lottery-ticket_initial_weights.pkl"
        
        if 'n' not in kwargs:
            self.n = 1
        
        self.save_dir = get_savepath(os.path.join(save_dir, cds() + '_pruning_callback'))
        make_folder(self.save_dir) # Make folder
        
        if percents is None: self._initialize_percents(self.model, base_percents)
        if masks is None: self._initialize_masks(self.model)
        
        # save initial weights
#         self.model.save_weights(get_savepath(os.path.join(self.save_dir, self.init_weights_filename)))
        pickle.dump(self.init_weights, open(get_savepath(os.path.join(self.save_dir, self.init_weights_filename)), "wb"))
        
        #etc
    
    def load(self, path):
        """
        load previous result of pruning callback
        
        Parameters
        ----------
        
        path : string
            a path of a file named *save_pruning_callback.pkl
        
        """
        
        self.layer_classes, self.base_percents, self.n, self.masks = pickle.load(open(path, "rb"))
        self.layer_names_dict = self.make_layer_names_dict()# Currently, this class only supports keras.layers.Conv3D or Dense.
        self.init_weights = pickle.load(open(os.path.join(os.path.dirname(path), self.init_weights_filename), "rb"))
        
        self._initialize_percents(self.model, self.base_percents)
        self.weights = self.extract_weights(self.model)
        self.save_dir = os.path.dirname(path) ; logger.info(f"save_dir was set to {self.save_dir}.")
        
    def make_layer_names_dict(self, layer_classes = [Conv2D, Conv3D, Dense]):
        rd = {}
        for i, l in enumerate(self.model.layers):
            if l.__class__.__name__ in self.layer_classes:
                rd[l.name] = {'ix':i, 'class_name':l.__class__.__name__}
        
        return rd
            
        
    def _initialize_masks(self, model):
        # Make masks
        masks = dict.fromkeys(self.layer_names_dict)

        for i, k in enumerate(masks.keys()):
            ws = model.layers[self.layer_names_dict[k]['ix']].get_weights()

            if len(ws) > 1:
                masks[k] = np.ones(ws[0].shape)
            else:
                masks[k] = np.ones(ws[0].shape)
            
        self.masks = masks
    
    
    def _initialize_percents(self, model, base_percents):
        # Make percents
        
        if not isinstance(base_percents, dict):
            base_percents = dict.fromkeys(self.layer_classes, base_percents)
            self.base_percents = base_percents
            
        percents = dict.fromkeys(self.layer_names_dict, 0)

        for i, k in enumerate(percents.keys()):
            percents[k] = ((base_percents[self.layer_names_dict[k]['class_name']] * 100) ** (1/self.n)) / 100 # percent value
        
        self.percents = percents
    
    
    def extract_weights(self, model):
        # Make masks
        wd = dict.fromkeys(self.layer_names_dict) # weights dictionary

        for i, k in enumerate(wd.keys()):
            cw = model.layers[self.layer_names_dict[k]['ix']].get_weights()
            if len(cw) > 1:
                wd[k] = cw[0] # bias is not included in the whole process.
            else:
                wd[k] = cw
        
        return wd
        
        
    def set_new_masks(self, mode='local'):
        """
        Make new masks for the layers of the model.
        
        Parameters
        ----------
        mode : string
            Choose the mode by which the new masks will be made
                
        
        """
        # Check arguments is valid.
        assert mode in ['global', 'local'], "Mode argument must take 'global' or 'local'."
        
        percents, masks, model = self.percents, self.masks, self.model
        
        final_weights = self.extract_weights(model)
        
        logger.info(f"base percents : {self.base_percents}")
        def set_global_cutoff(masks, final_weights, percents, layer_class):
            flatten_weights = []
            for k, v in final_weights.items():
                flatten_weights.append(v[masks[k] == 1].flatten().tolist())
            flatten_weights = list(itertools.chain(*flatten_weights))

            sorted_weights = np.sort(np.abs(flatten_weights))
            cutoff_index = np.round(self.base_percents[layer_class] * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index]
            
            return cutoff
            logger.info(f"Global cutoff: {cutoff}")
        
        if mode == 'global':
            cutoffs = dict.fromkeys(self.layer_classes)
            
            # extract weights per class.
            for i, k in enumerate(cutoffs):
                wpc = {kk:v for kk, v in final_weights.items() if self.layer_names_dict[kk]['class_name'] == k} # weight per class 
                mpc = {kk:v for kk, v in masks.items() if self.layer_names_dict[kk]['class_name'] == k}
                
                cutoffs[k] = set_global_cutoff(mpc, wpc, percents, k)
                
                logger.info(f"Global cutoffs: {cutoffs.items()}")
            
        def prune_by_percent_once(percent, mask, final_weight, cutoff):
            # Put the weights that aren't masked out in sorted order.
#             print(final_weight)
            sorted_weights = np.sort(np.abs(final_weight[mask == 1]))
#             print(sorted_weights, sorted_weights.shape)

            # Determine the cutoff for weights to be pruned.
            if mode == 'local':
                cutoff_index = np.round(percent * sorted_weights.size).astype(int)
                cutoff = sorted_weights[cutoff_index]
                logger.info(f"cutoff index: {cutoff_index}")
            
            logger.info(f"cutoff: {cutoff}")
#             print("percentile :", np.percentile(sorted_weights, 20))
            
            numerator = np.where(np.abs(final_weight[mask == 1]) <= cutoff)[0].size
            denominator = np.where(np.abs(final_weight[mask == 1]) > cutoff)[0].size
            logger.info(f"{numerator} " + 
            f"out of {denominator+numerator}({round(numerator/(denominator+numerator)*100, 1)}%) parameters will be pruned.")
            
            # Prune all weights below the cutoff.
            return np.where(np.abs(final_weight) <= cutoff, np.zeros(mask.shape), mask)

        new_masks = {}
        
        for k, percent in percents.items():
            logger.info(k)
            if mode == 'global':
                logger.info(f"Class name: {self.layer_names_dict[k]['class_name']}, Cutoff : {cutoffs[self.layer_names_dict[k]['class_name']]}")
                new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k], cutoffs[self.layer_names_dict[k]['class_name']])
            
            elif mode == 'local':
                new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k], cutoff = None)

        self.old_masks = deepcopy(self.masks)
        self.masks = new_masks
        
        logger.info(f">> self.masks was newly set.")
        
        
    def pruning_weights(self, model=None, masks=None):
        """
        Apply current masks to the model and save current information(weights-masks-percents) to a pickle file.
        """
        
        if model is None: model = self.model
        if masks is None: masks = self.masks
        
        # Backup og_weights and old masks.
        savepath = get_savepath(os.path.join(self.save_dir, f"{self.n}-th_save_pruning_callback.pkl"))
        with open(savepath, "wb") as f:
            pickle.dump([self.layer_classes, self.base_percents, self.n, self.masks], f, protocol=4)
        
        print(f"{self.n}-th information was saved to {savepath}.")
        
        self._apply_masks(model, masks)
        
        
        self.weights = self.extract_weights(self.model)
        
        self.n += 1
        self._initialize_percents(self.model, self.base_percents)
        
        logger.info(f">> N was set to {self.n}.")
        logger.info(f">> Base percents were reduced to {self.percents}.")
        logger.info(f">> Masks were applied to the model weights.")
        logger.info(f">> self.weights was newly set.")
        
    def _apply_masks(self, model=None, masks=None):
        if model is None: model = self.model
        if masks is None: masks = self.masks
        
        for l in model.layers:
            new_weights = []
            if l.name in masks.keys():
                weights = l.get_weights()
#                 print(l.name, masks[l.name].shape, weights[0].shape)
                if len(weights) > 1:
                    new_weights.append(weights[0] * masks[l.name])
                    new_weights.append(weights[1])
                
                l.set_weights(new_weights)
                
    def restore_initial_weights(self):
        """
        Restore initial values of remaining weights.
        """
        self.model.set_weights(self.init_weights)
        self._apply_masks()
        logger.info(">> Initial weights were restored.")
        
    def on_train_begin(self, logs):
        # Apply mask to the weights of model.
        self._apply_masks()
        
    def on_batch_end(self, batch, logs):
        # Apply mask to the weights of model.
        self._apply_masks()
    
    def on_train_end(self, logs):
        # Save current pruningcallback instance to pkl file.
        pass
    

###
def set_weight_decay(model, alpha):
    """
    Adds L2 regularization to the layers of models.
    
    This function can be applied to Conv2D, Dense or DepthwiseConv2D
    
    Parameters
    ----------
    
    model : a keras model object.
    
    alpha : float
        lambda value.
        
    """
    for layer in model.layers:
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.depthwise_kernel))
        elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.bias))

            
###            
def csv_remove_duplicates(csv_path):
    """
    Drops duplicated rows based on epoch column in csv log.
    
    Parameters
    ----------
    csv_path : string
        csv file path.
        
    """
    
    
    csv_p = pd.read_csv(csv_path)
    csv_p.drop_duplicates(['epoch'], keep = 'last').to_csv(csv_path, index = False)
    
    
    
### PruningCallbacks
class EarlyStopping_c1(keras.callbacks.EarlyStopping):
    """
    Derived class of keras.callbacks.EarlyStopping but has an attribute named best_epoch.
    
    The best epoch in training was assigned to it.
    """
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)
                    
                    
### 
def save_model(model, json_path):
    """
    Save a model's config to a json file.
    
    Parameters
    ----------
    model : a keras model object
    
    json_path : a json file path
    
    """
    model_json = model.to_json()
    with open(json_path, "w") as json_file : 
        json_file.write(model_json)
        
        
###
def load_model(json_path, weight_h5 = None):
    """
    Load a model with a json config file and load weights with a h5 file.
    
    Parameters
    ----------
    json_path : a json file path containing a model config.
    
    weight_h5 : a h5 file path containing model weights.
    
    """
    with open(json_path) as f:
        json_config = f.read()
        
    model = model_from_json(json_config)
    
    if weight_h5:
        model.load_weights(weight_h5)
    
    return model    