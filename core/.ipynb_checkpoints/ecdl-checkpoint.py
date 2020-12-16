import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

### Preactivation Resnet50 block's F (Residual function)
class PreactRes50_BlockF(keras.layers.Layer):
    def __init__(self, filters, kernel_size = (3, 3), data_format = 'channels_last', kernel_initializer = "he_normal", first_conv2d_strides = (1, 1), name = '', **kwargs):
        """
        initdocstring
        """
        super().__init__(**kwargs)
        
        self.hidden = [
                        keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_1_{name}"),
                        keras.layers.ReLU(name = f"Relu_1_{name}"),
                        keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = first_conv2d_strides, padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"Conv2D_1_{name}"),
            
                        keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_2_{name}"),
                        keras.layers.ReLU(name = f"Relu_2_{name}"),
                        keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=(1, 1), padding = 'same',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"Conv2D_2_{name}"),
            
                        keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_3_{name}"),
                        keras.layers.ReLU(name = f"Relu_3_{name}"),
                        keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1), strides=(1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"Conv2D_3_{name}"),
                        
                    ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
            
        return Z
    

### DecoderConvs in https://arxiv.org/abs/1810.13230
class DecoderConv(keras.layers.Layer):
    def __init__(self, cardinality, filters, data_format = 'channels_last', kernel_initializer = "he_normal", name = '', **kwargs):
        """
        initdocstring
        """
        super().__init__(**kwargs)
        
        filter_per_path = int(filters / cardinality)
        
        self.hidden = [
                        keras.layers.Conv2D(filters = filter_per_path * 2, kernel_size = (5, 5), strides = (1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"Conv2D_1_{name}"),
            
                        keras.layers.Conv2D(filters = filter_per_path * 2, kernel_size = (3, 3), strides=(1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"Conv2D_2_{name}"),
            
                        keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides=(1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"Conv2D_3_{name}"),
                    ]
        
    def call(self, inputs):
        Z = inputs
        
        for layer in self.hidden:
            Z = layer(Z)
        
        return Z
    
    
### DecoderBlock (Unused) in https://arxiv.org/abs/1810.13230
class DecoderBlock(keras.layers.Layer):
    def __init__(self, cardinality, filters, data_format = 'channels_last', kernel_initializer = "he_normal", name = "", first_conv2d_strides = (1, 1), **kwargs):
        """
        initdocstring
        """
        super().__init__(name = name, **kwargs)
        
        # filters, cardinality check
        assert filters % cardinality == 0, "`filters / cardinality` is not an int. filters = {}, cardinality = {}".format(filters, cardinality)
        
        self.hidden = []
        for c in range(cardinality):
            self.hidden.append(
                                DecoderConv(cardinality, filters, data_format = 'channels_last', kernel_initializer = "he_normal", name = f'{self.name}_C_{c}', **kwargs)
                            )
        
        self.conv_reduce_channel = keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{self.name}_reduce_channel_conv")
        self.add1 = keras.layers.Add(name = f"{self.name}_aggregate_tranformation")
        self.add2 = keras.layers.Add(name = f"{self.name}_Res_add")
        
        
    def build(self, batch_input_shape):
        self.shortcut_resize = Resize2D(target_size = [batch_input_shape[1] - 6, batch_input_shape[2] - 6], name = f"{self.name}_shortcut_resize")
        
        super().build(batch_input_shape)
        
    def call(self, inputs):
        sc = inputs
        
        rl = []
        for dec_convs in self.hidden:
            rl.append(dec_convs(sc))
        
        Z = self.add1(rl)
        
        sc = self.conv_reduce_channel(sc)
        sc = self.shortcut_resize(sc)
        
        Z = self.add2([sc, Z])
        
        return Z
    
    
### Resize2D layer
class Resize2D(keras.layers.Layer):
    def __init__(self, target_size, method = 'nearest', name = '', **kwargs):
        """
        target_size : a sequence of height, width
        """
        super().__init__(**kwargs)
        
        self.name_c = name
        self.method = method
        self.target_size_tensor = tf.constant([target_size[0], target_size[1]])
        
    def call(self, inputs):
        Z = inputs
        return tf.image.resize(Z, self.target_size_tensor, method = self.method, name = self.name_c)
    

### CentralCrop2D layer
class CentralCrop2D(keras.layers.Layer):
    def __init__(self, target_size, name = '', **kwargs):
        """
        target_size : a sequence of height, width
        """
        super().__init__(**kwargs)
        
        self.th, self.tw = target_size
        
    def call(self, inputs):
        Z = inputs
        
        ih = tf.shape(Z)[1]
        iw = tf.shape(Z)[2]
        
        oh = tf.cast(
                    tf.cast(tf.round(ih / 2), tf.float32) - tf.round(self.th / 2), tf.int32)
        ow = tf.cast(
                    tf.cast(tf.round(iw / 2), tf.float32) - tf.round(self.tw / 2), tf.int32)
        
        return tf.image.crop_to_bounding_box(Z, oh, ow, self.th, self.tw)
    
    
    
### ZeroPaddingDepth
class ZeroPaddingDepth(keras.layers.Layer):
    """
    Padding Zeros along depth.
    
    Currently, only channels_last format is supported.
    
    """
    def __init__(self, target_depth, name = "", **kwargs):
        super().__init__(**kwargs)
        
        self.target_depth = target_depth
        self.concat = keras.layers.Concatenate(name = f"concat_{name}")
        self.name_c = name
        
    def call(self, inputs):
        Z = inputs
        inp_shape = tf.shape(Z)
        target_shape = tf.convert_to_tensor([inp_shape[0], inp_shape[1], inp_shape[2], self.target_depth - inp_shape[-1]])
        
        zp = tf.zeros(shape = target_shape, dtype = inputs.dtype, name = f"zeropadding_{self.name_c}")
        
        Z = self.concat([Z, zp])
        
        return Z
    

### Preact Resnet50 Segmentation model in https://arxiv.org/abs/1810.13230
class PreactResnetModified(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        ### layers definition 
        ## conv1
        self.conv1 = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides=(1, 1), padding = "valid",
                                            data_format = 'channels_last', kernel_initializer = "he_normal", name = "conv1")
        
        ## conv2_x
        self.conv2_1_F = PreactRes50_BlockF(filters = 64, name = "conv2_1_F")
        self.conv2_1_zp = ZeroPaddingDepth(target_depth = 256, name = "conv2_1_depth-zeropadding")
        
        self.conv2_1_add = keras.layers.Add(name = "conv2_1_add")
        
        self.conv2_2_F = PreactRes50_BlockF(filters = 64, name = "conv2_2_F")
        self.conv2_2_add = keras.layers.Add(name = "conv2_2_add")
        
        self.conv2_3_F = PreactRes50_BlockF(filters = 64, name = "conv2_3_F")
        self.conv2_3_add = keras.layers.Add(name = "conv2_3_add")
        
        self.conv2_crop = CentralCrop2D((60, 60), name = "conv2_crop")
        
        
        ## conv3_x
        self.conv3_1_F = PreactRes50_BlockF(filters = 128, first_conv2d_strides = (2, 2), name = "conv3_1_F")
        self.conv3_1_downsample = keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv3_1_downsample")
        self.conv3_1_zp = ZeroPaddingDepth(target_depth = 512, name = "conv3_1_depth-zeropadding")
        self.conv3_1_add = keras.layers.Add(name = "conv3_1_add")
        
        self.conv3_2_F = PreactRes50_BlockF(filters = 128, name = "conv3_2_F")
        self.conv3_2_add = keras.layers.Add(name = "conv3_2_add")
        
        self.conv3_3_F = PreactRes50_BlockF(filters = 128, name = "conv3_3_F")
        self.conv3_3_add = keras.layers.Add(name = "conv3_3_add")
        
        self.conv3_4_F = PreactRes50_BlockF(filters = 128, name = "conv3_4_F")
        self.conv3_4_add = keras.layers.Add(name = "conv3_4_add")
        
        self.conv3_crop = CentralCrop2D((36, 36), name = "conv3_crop")
        
        
        ## conv4_x
        self.conv4_1_F = PreactRes50_BlockF(filters = 256, first_conv2d_strides = (2, 2), name = "conv4_1_F")
        self.conv4_1_downsample = keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv4_1_downsample")
        self.conv4_1_zp = ZeroPaddingDepth(target_depth = 1024, name = "conv4_1_depth-zeropadding")
        self.conv4_1_add = keras.layers.Add(name = "conv4_1_add")
        
        self.conv4_2_F = PreactRes50_BlockF(filters = 256, name = "conv4_2_F")
        self.conv4_2_add = keras.layers.Add(name = "conv4_2_add")
        
        self.conv4_3_F = PreactRes50_BlockF(filters = 256, name = "conv4_3_F")
        self.conv4_3_add = keras.layers.Add(name = "conv4_3_add")
        
        self.conv4_4_F = PreactRes50_BlockF(filters = 256, name = "conv4_4_F")
        self.conv4_4_add = keras.layers.Add(name = "conv4_4_add")
        
        self.conv4_5_F = PreactRes50_BlockF(filters = 256, name = "conv4_5_F")
        self.conv4_5_add = keras.layers.Add(name = "conv4_5_add")
        
        self.conv4_6_F = PreactRes50_BlockF(filters = 256, name = "conv4_6_F")
        self.conv4_6_add = keras.layers.Add(name = "conv4_6_add")
        
        self.conv4_crop = CentralCrop2D((24, 24), name = "conv4_crop")
        
        
        ## conv5_x
        self.conv5_1_F = PreactRes50_BlockF(filters = 512, first_conv2d_strides = (2, 2), name = "conv5_1_F")
        self.conv5_1_downsample = keras.layers.Conv2D(filters = 1024, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv5_1_downsample")
        self.conv5_1_zp = ZeroPaddingDepth(target_depth = 2048, name = "conv5_1_depth-zeropadding")
        self.conv5_1_add = keras.layers.Add(name = "conv5_1_add")
        
        self.conv5_2_F = PreactRes50_BlockF(filters = 512, name = "conv5_2_F")
        self.conv5_2_add = keras.layers.Add(name = "conv5_2_add")
        
        self.conv5_3_F = PreactRes50_BlockF(filters = 512, name = "conv5_3_F")
        self.conv5_3_add = keras.layers.Add(name = "conv5_3_add")
        
        self.post_bn = keras.layers.BatchNormalization(axis = -1, name = f"Post_BatchNormalization")
        self.post_relu = keras.layers.ReLU(name = f"Post_Relu")
        
        ## decoder 4
        self.dec4_transpose = keras.layers.Conv2DTranspose(filters = 1024, kernel_size = (2, 2), strides = (2, 2))
        
        
        ## decoder 3
        self.dec3_add = keras.layers.Add(name = "dec3_add")
        self.dec3_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec3")
        self.dec3_rl = keras.layers.ReLU(name = f"Relu_dec3")
        self.dec3 = DecoderBlock(cardinality = 256, filters = 512, name = 'dec3')
        self.dec3_rs = Resize2D((36, 36), name = "dec3_rs")
        
        
        ## decoder 2
        self.dec2_add = keras.layers.Add(name = "dec2_add")
        self.dec2_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec2")
        self.dec2_rl = keras.layers.ReLU(name = f"Relu_dec2")
        self.dec2 = DecoderBlock(cardinality = 128, filters = 256, name = 'dec2')
        self.dec2_rs = Resize2D((60, 60), name = "dec2_rs")
        
        
        ## decoder 1
        self.dec1_add = keras.layers.Add(name = "dec1_add")
        self.dec1_bn1 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_1_dec1")
        self.dec1_rl1 = keras.layers.ReLU(name = f"Relu_1_dec1")
        self.dec1 = DecoderBlock(cardinality = 64, filters = 128, name = 'dec1')
        self.dec1_bn2 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_2_dec1")
        self.dec1_rl2 = keras.layers.ReLU(name = f"Relu_2_dec1")       
  
        
        ### output sigmoid
        self.out_conv = keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                            activation = 'sigmoid', kernel_initializer = "he_normal", name = f"out_conv", dtype = 'float32')
#         self.out_act = keras.layers.Activation('linear', dtype='float32', name = "linear_to_float32") # For using mixed precision. The output dtype should be float32, not float16.
        
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        
    def call(self, inputs):
        ### ouput node connection
        Z = inputs
            
            
        ## conv1    
        Zo = self.conv1(Z)
        
        
        ## conv2
        Z = self.conv2_1_F(Zo)
        Zs = self.conv2_1_zp(Zo)
        Zo = self.conv2_1_add([Z, Zs])
        
        Z = self.conv2_2_F(Zo)
        Zo = self.conv2_2_add([Zo, Z])
        
        Z = self.conv2_3_F(Zo)
        Zo = self.conv2_3_add([Zo, Z])
        
        Z2 = self.conv2_crop(Zo)
        
        
        ## conv3
        Z = self.conv3_1_F(Zo)
        Zs = self.conv3_1_downsample(Zo)
        Zs = self.conv3_1_zp(Zs)
        Zo = self.conv3_1_add([Z, Zs])
        
        Z = self.conv3_2_F(Zo)
        Zo = self.conv3_2_add([Zo, Z])
        
        Z = self.conv3_3_F(Zo)
        Zo = self.conv3_3_add([Zo, Z])
        
        Z = self.conv3_4_F(Zo)
        Zo = self.conv3_4_add([Zo, Z])
        
        Z3 = self.conv3_crop(Zo)
        
        
        ## conv4
        Z = self.conv4_1_F(Zo)
        Zs = self.conv4_1_downsample(Zo)
        Zs = self.conv4_1_zp(Zs)
        Zo = self.conv4_1_add([Z, Zs])
        
        Z = self.conv4_2_F(Zo)
        Zo = self.conv4_2_add([Zo, Z])
        
        Z = self.conv4_3_F(Zo)
        Zo = self.conv4_3_add([Zo, Z])
        
        Z = self.conv4_4_F(Zo)
        Zo = self.conv4_4_add([Zo, Z])
        
        Z = self.conv4_5_F(Zo)
        Zo = self.conv4_5_add([Zo, Z])
        
        Z = self.conv4_6_F(Zo)
        Zo = self.conv4_6_add([Zo, Z])
        
        Z4 = self.conv4_crop(Zo)
        
        
        ## conv 5
        Z = self.conv5_1_F(Zo)
        Zs = self.conv5_1_downsample(Zo)
        Zs = self.conv5_1_zp(Zs)
        Zo = self.conv5_1_add([Z, Zs])
        
        Z = self.conv5_2_F(Zo)
        Zo = self.conv5_2_add([Zo, Z])
        
        Z = self.conv5_3_F(Zo)
        Z = self.conv5_3_add([Zo, Z])
        
        Z = self.post_bn(Z)
        Zo = self.post_relu(Z)
        
        ## decoder 4(Orange color box in the article)
        Zo = self.dec4_transpose(Zo)
        
        ## decoder 3
        Z = self.dec3_add([Zo, Z4])
        Z = self.dec3_bn(Z)
        Z = self.dec3_rl(Z)
        Z = self.dec3(Z)
        Zo = self.dec3_rs(Z)
        
        ## decoder 2
        Z = self.dec2_add([Zo, Z3])
        Z = self.dec2_bn(Z)
        Z = self.dec2_rl(Z)
        Z = self.dec2(Z)
        Zo = self.dec2_rs(Z)
        
        
        ## decoder 1
        Z = self.dec1_add([Zo, Z2])
        Z = self.dec1_bn1(Z)
        Z = self.dec1_rl1(Z)
        Z = self.dec1(Z)
        Z = self.dec1_bn2(Z)
        Zo = self.dec1_rl2(Z)
        
        
        ## out
        dec_out = self.out_conv(Zo)
#         dec_out = self.out_act(Z)
#         print(Zo.dtype, self.out_conv.kernel.dtype, dec_out.dtype)
#         tf.print(dec_out)
        
        return dec_out
    
### Preactivation Resnet50 block's F (Residual function)
class PreactRes50_BlockF2(keras.layers.Layer):
    def __init__(self, filters, kernel_size = (3, 3), data_format = 'channels_last', kernel_initializer = "he_normal", first_conv2d_strides = (1, 1), name = '', **kwargs):
        """
        initdocstring
        """
        super().__init__(**kwargs)
        
        self.hidden = [
                        keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = first_conv2d_strides, padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{name}_1_conv"),
                        keras.layers.BatchNormalization(axis = -1, name = f"{name}_1_bn"),
                        keras.layers.ReLU(name = f"{name}_1_relu"),
            
                        keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=(1, 1), padding = 'same',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{name}_2_conv"),
                        keras.layers.BatchNormalization(axis = -1, name = f"{name}_2_bn"),
                        keras.layers.ReLU(name = f"{name}_2_relu"),
            
                        keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1), strides=(1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{name}_3_conv"),
                        
                    ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
            
        return Z
    
### Preact Resnet50 Segmentation model in https://arxiv.org/abs/1810.13230
class PreactResnetModified2(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        ### layers definition 
        ## conv1
        self.conv1 = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides=(1, 1), padding = "valid",
                                            data_format = 'channels_last', kernel_initializer = "he_normal", name = "conv1")
        
        ## conv2_x
        self.conv2_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block1_preact_bn")
        self.conv2_1_preact_relu = keras.layers.ReLU(name = f"conv2_block1_preact_relu")
        self.conv2_1_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block1")
        self.conv2_1_zp = ZeroPaddingDepth(target_depth = 256, name = "conv2_block1_depth-zeropadding")
        self.conv2_1_add = keras.layers.Add(name = "conv2_block1_out")
        
        self.conv2_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block2_preact_bn")
        self.conv2_2_preact_relu = keras.layers.ReLU(name = f"conv2_block2_preact_relu")
        self.conv2_2_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block2")
        self.conv2_2_add = keras.layers.Add(name = "conv2_block2_out")
        
        self.conv2_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block3_preact_bn")
        self.conv2_3_preact_relu = keras.layers.ReLU(name = f"conv2_block3_preact_relu")
        self.conv2_3_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block3")
        self.conv2_3_add = keras.layers.Add(name = "conv2_block3_out")
        
        
        ## conv3_x
        self.conv3_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block1_preact_bn")
        self.conv3_1_preact_relu = keras.layers.ReLU(name = f"conv3_block1_preact_relu")
        self.conv2_crop = CentralCrop2D((60, 60), name = "conv2_crop")
        
        self.conv3_1_F = PreactRes50_BlockF2(filters = 128, first_conv2d_strides = (2, 2), name = "conv3_block1")
        self.conv3_1_downsample = keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv3_block1_downsample")
        self.conv3_1_zp = ZeroPaddingDepth(target_depth = 512, name = "conv3_block1_depth-zeropadding")
        self.conv3_1_add = keras.layers.Add(name = "conv3_block1_out")
        
        
        self.conv3_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block2_preact_bn")
        self.conv3_2_preact_relu = keras.layers.ReLU(name = f"conv3_block2_preact_relu")

        self.conv3_2_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block2")
        self.conv3_2_add = keras.layers.Add(name = "conv3_block2_out")
        
        self.conv3_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block3_preact_bn")
        self.conv3_3_preact_relu = keras.layers.ReLU(name = f"conv3_block3_preact_relu")
        self.conv3_3_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block3")
        self.conv3_3_add = keras.layers.Add(name = "conv3_block3_out")
        
        
        self.conv3_4_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block4_preact_bn")
        self.conv3_4_preact_relu = keras.layers.ReLU(name = f"conv3_block4_preact_relu")
        self.conv3_4_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block4")
        self.conv3_4_add = keras.layers.Add(name = "conv3_block4_out")
        
        
        ## conv4_x
        self.conv4_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block1_preact_bn")
        self.conv4_1_preact_relu = keras.layers.ReLU(name = f"conv4_block1_preact_relu")
        self.conv3_crop = CentralCrop2D((36, 36), name = "conv3_crop")
        
        self.conv4_1_F = PreactRes50_BlockF2(filters = 256, first_conv2d_strides = (2, 2), name = "conv4_block1")
        self.conv4_1_downsample = keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv4_block1_downsample")
        self.conv4_1_zp = ZeroPaddingDepth(target_depth = 1024, name = "conv4_block1_depth-zeropadding")
        self.conv4_1_add = keras.layers.Add(name = "conv4_block1_out")
        
        
        self.conv4_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block2_preact_bn")
        self.conv4_2_preact_relu = keras.layers.ReLU(name = f"conv4_block2_preact_relu")
        self.conv4_2_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block2")
        self.conv4_2_add = keras.layers.Add(name = "conv4_block2_out")
        
        
        self.conv4_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block3_preact_bn")
        self.conv4_3_preact_relu = keras.layers.ReLU(name = f"conv4_block3_preact_relu")
        self.conv4_3_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block3")
        self.conv4_3_add = keras.layers.Add(name = "conv4_block3_out")
        
        
        self.conv4_4_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block4_preact_bn")
        self.conv4_4_preact_relu = keras.layers.ReLU(name = f"conv4_block4_preact_relu")
        self.conv4_4_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block4")
        self.conv4_4_add = keras.layers.Add(name = "conv4_block4_out")
        
        
        self.conv4_5_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block5_preact_bn")
        self.conv4_5_preact_relu = keras.layers.ReLU(name = f"conv4_block5_preact_relu")
        self.conv4_5_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block5")
        self.conv4_5_add = keras.layers.Add(name = "conv4_block5_out")
        
        
        self.conv4_6_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block6_preact_bn")
        self.conv4_6_preact_relu = keras.layers.ReLU(name = f"conv4_block6_preact_relu")
        self.conv4_6_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block6")
        self.conv4_6_add = keras.layers.Add(name = "conv4_block6_out")
        
        
        ## conv5_x
        self.conv5_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block1_preact_bn")
        self.conv5_1_preact_relu = keras.layers.ReLU(name = f"conv5_block1_preact_relu")
        self.conv4_crop = CentralCrop2D((24, 24), name = "conv4_crop")
        
        self.conv5_1_F = PreactRes50_BlockF2(filters = 512, first_conv2d_strides = (2, 2), name = "conv5_block1F")
        self.conv5_1_downsample = keras.layers.Conv2D(filters = 1024, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv5_block1_downsample")
        self.conv5_1_zp = ZeroPaddingDepth(target_depth = 2048, name = "conv5_block1_depth-zeropadding")
        self.conv5_1_add = keras.layers.Add(name = "conv5_block1_out")
        
        
        self.conv5_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block2_preact_bn")
        self.conv5_2_preact_relu = keras.layers.ReLU(name = f"conv5_block2_preact_relu")
        self.conv5_2_F = PreactRes50_BlockF2(filters = 512, name = "conv5_block2")
        self.conv5_2_add = keras.layers.Add(name = "conv5_block2_out")
        
        
        self.conv5_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block3_preact_bn")
        self.conv5_3_preact_relu = keras.layers.ReLU(name = f"conv5_block3_preact_relu")
        self.conv5_3_F = PreactRes50_BlockF2(filters = 512, name = "conv5_block3")
        self.conv5_3_add = keras.layers.Add(name = "conv5_block3_out")
        
        
        self.post_bn = keras.layers.BatchNormalization(axis = -1, name = f"Post_BatchNormalization")
        self.post_relu = keras.layers.ReLU(name = f"Post_Relu")
        
        
        ## decoder 4
        self.dec4_transpose = keras.layers.Conv2DTranspose(filters = 1024, kernel_size = (2, 2), strides = (2, 2))
        
        
        ## decoder 3
        self.dec3_add = keras.layers.Add(name = "dec3_add")
        self.dec3_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec3")
        self.dec3_rl = keras.layers.ReLU(name = f"Relu_dec3")
        self.dec3 = DecoderBlock(cardinality = 256, filters = 512, name = 'dec3')
        self.dec3_rs = Resize2D((36, 36), name = "dec3_rs")
        
        
        ## decoder 2
        self.dec2_add = keras.layers.Add(name = "dec2_add")
        self.dec2_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec2")
        self.dec2_rl = keras.layers.ReLU(name = f"Relu_dec2")
        self.dec2 = DecoderBlock(cardinality = 128, filters = 256, name = 'dec2')
        self.dec2_rs = Resize2D((60, 60), name = "dec2_rs")
        
        
        ## decoder 1
        self.dec1_add = keras.layers.Add(name = "dec1_add")
        self.dec1_bn1 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_1_dec1")
        self.dec1_rl1 = keras.layers.ReLU(name = f"Relu_1_dec1")
        self.dec1 = DecoderBlock(cardinality = 64, filters = 128, name = 'dec1')
        self.dec1_bn2 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_2_dec1")
        self.dec1_rl2 = keras.layers.ReLU(name = f"Relu_2_dec1")       
  
        
        ### output sigmoid
        self.out_conv = keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                            activation = 'sigmoid', kernel_initializer = "he_normal", name = f"out_conv", dtype = 'float32')
#         self.out_act = keras.layers.Activation('linear', dtype='float32', name = "linear_to_float32") # For using mixed precision. The output dtype should be float32, not float16.
        
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        
    def call(self, inputs):
        ### ouput node connection
        Z = inputs
            
            
        ## conv1    
        Zo = self.conv1(Z)
        
        
        ## conv2
        Z = self.conv2_1_preact_bn(Zo)
        Z = self.conv2_1_preact_relu(Z)
        Z = self.conv2_1_F(Z)
        Zs = self.conv2_1_zp(Zo)
        Zo = self.conv2_1_add([Z, Zs])
        
        Z = self.conv2_2_preact_bn(Zo)
        Z = self.conv2_2_preact_relu(Z)
        Z = self.conv2_2_F(Z)
        Zo = self.conv2_2_add([Zo, Z])
        
        Z = self.conv2_3_preact_bn(Zo)
        Z = self.conv2_3_preact_relu(Z)
        Z = self.conv2_3_F(Z)
        Zo = self.conv2_3_add([Zo, Z])
        
        
        ## conv3
        Z = self.conv3_1_preact_bn(Zo)
        Z = self.conv3_1_preact_relu(Z)
        Z2 = self.conv2_crop(Z)
        
        Z = self.conv3_1_F(Z)
        Zs = self.conv3_1_downsample(Zo)
        Zs = self.conv3_1_zp(Zs)
        Zo = self.conv3_1_add([Z, Zs])
        
        Z = self.conv3_2_preact_bn(Zo)
        Z = self.conv3_2_preact_relu(Z)
        Z = self.conv3_2_F(Z)
        Zo = self.conv3_2_add([Zo, Z])
        
        Z = self.conv3_3_preact_bn(Zo)
        Z = self.conv3_3_preact_relu(Z)
        Z = self.conv3_3_F(Z)
        Zo = self.conv3_3_add([Zo, Z])
        
        Z = self.conv3_4_preact_bn(Zo)
        Z = self.conv3_4_preact_relu(Z)
        Z = self.conv3_4_F(Z)
        Zo = self.conv3_4_add([Zo, Z])
        
        
        ## conv4
        Z = self.conv4_1_preact_bn(Zo)
        Z = self.conv4_1_preact_relu(Z)
        Z3 = self.conv3_crop(Z)
        Z = self.conv4_1_F(Z)
        
        Zs = self.conv4_1_downsample(Zo)
        Zs = self.conv4_1_zp(Zs)
        Zo = self.conv4_1_add([Z, Zs])
        
        Z = self.conv4_2_preact_bn(Zo)
        Z = self.conv4_2_preact_relu(Z)
        Z = self.conv4_2_F(Z)
        Zo = self.conv4_2_add([Zo, Z])
        
        Z = self.conv4_3_preact_bn(Zo)
        Z = self.conv4_3_preact_relu(Z)
        Z = self.conv4_3_F(Z)
        Zo = self.conv4_3_add([Zo, Z])
        
        Z = self.conv4_4_preact_bn(Zo)
        Z = self.conv4_4_preact_relu(Z)
        Z = self.conv4_4_F(Z)
        Zo = self.conv4_4_add([Zo, Z])
        
        Z = self.conv4_5_preact_bn(Zo)
        Z = self.conv4_5_preact_relu(Z)
        Z = self.conv4_5_F(Z)
        Zo = self.conv4_5_add([Zo, Z])
        
        Z = self.conv4_6_preact_bn(Zo)
        Z = self.conv4_6_preact_relu(Z)
        Z = self.conv4_6_F(Z)
        Zo = self.conv4_6_add([Zo, Z])
        
        
        ## conv 5
        Z = self.conv5_1_preact_bn(Zo)
        Z = self.conv5_1_preact_relu(Z)
        Z4 = self.conv4_crop(Z)
        
        Z = self.conv5_1_F(Z)
        Zs = self.conv5_1_downsample(Zo)
        Zs = self.conv5_1_zp(Zs)
        Zo = self.conv5_1_add([Z, Zs])
        
        Z = self.conv5_2_preact_bn(Zo)
        Z = self.conv5_2_preact_relu(Z)
        Z = self.conv5_2_F(Z)
        Zo = self.conv5_2_add([Zo, Z])
        
        Z = self.conv5_3_preact_bn(Zo)
        Z = self.conv5_3_preact_relu(Z)
        Z = self.conv5_3_F(Z)
        Z = self.conv5_3_add([Zo, Z])
        
        Z = self.post_bn(Z)
        Zo = self.post_relu(Z)
        
#         return Zo
        
        
        ## decoder 4(Orange color box in the article)
        Zo = self.dec4_transpose(Zo)
        
        ## decoder 3
        Z = self.dec3_add([Zo, Z4])
        Z = self.dec3_bn(Z)
        Z = self.dec3_rl(Z)
        Z = self.dec3(Z)
        Zo = self.dec3_rs(Z)
        
        ## decoder 2
        Z = self.dec2_add([Zo, Z3])
        Z = self.dec2_bn(Z)
        Z = self.dec2_rl(Z)
        Z = self.dec2(Z)
        Zo = self.dec2_rs(Z)
        
        
        ## decoder 1
        Z = self.dec1_add([Zo, Z2])
        Z = self.dec1_bn1(Z)
        Z = self.dec1_rl1(Z)
        Z = self.dec1(Z)
        Z = self.dec1_bn2(Z)
        Zo = self.dec1_rl2(Z)
        
        
        ## out
        dec_out = self.out_conv(Zo)
#         dec_out = self.out_act(Z)
#         print(Zo.dtype, self.out_conv.kernel.dtype, dec_out.dtype)
#         tf.print(dec_out)
        
        return dec_out
    
    
### DecoderBlock (Unused) in https://arxiv.org/abs/1810.13230
class DecoderBlock_GC(keras.layers.Layer):
    def __init__(self, cardinality, filters, data_format = 'channels_last', kernel_initializer = "he_normal", name = "", first_conv2d_strides = (1, 1), **kwargs):
        """
        initdocstring
        """
        super().__init__(name = name, **kwargs)
        
        # filters, cardinality check
        assert filters % cardinality == 0, "`filters / cardinality` is not an int. filters = {}, cardinality = {}".format(filters, cardinality)
        
        
        self.conv_reduce = keras.layers.Conv2D(filters = filters * 2, kernel_size = (5, 5), strides=(1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{self.name}_reduce_conv")
        self.conv_group = keras.layers.Conv2D(filters = filters * 2, kernel_size = (3, 3), strides=(1, 1), padding = 'valid', 
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{self.name}_group_conv",
                                             groups = cardinality)
        self.conv_restore = keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides=(1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{self.name}_restore_conv")
        
        
        self.hidden = [self.conv_reduce, self.conv_group, self.conv_restore]
        
        self.conv_reduce_channel = keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{self.name}_reduce_channel_conv")
        self.add = keras.layers.Add(name = f"{self.name}_out")
        
        
    def build(self, batch_input_shape):
        self.shortcut_resize = Resize2D(target_size = [batch_input_shape[1] - 6, batch_input_shape[2] - 6], name = f"{self.name}_shortcut_resize")
        
        super().build(batch_input_shape)
        
    def call(self, inputs):
        sc = inputs
        
        Z = self.conv_reduce(sc)
        Z = self.conv_group(Z)
        Z = self.conv_restore(Z)
        
        sc = self.conv_reduce_channel(sc)
        sc = self.shortcut_resize(sc)
        
        Z = self.add([sc, Z])
        
        return Z
    
    
### Preact Resnet50 Segmentation model in https://arxiv.org/abs/1810.13230
class PreactResnetModified3(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        ### layers definition 
        ## conv1
        self.conv1 = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides=(1, 1), padding = "valid",
                                            data_format = 'channels_last', kernel_initializer = "he_normal", name = "conv1")
        
        ## conv2_x
        self.conv2_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block1_preact_bn")
        self.conv2_1_preact_relu = keras.layers.ReLU(name = f"conv2_block1_preact_relu")
        self.conv2_1_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block1")
        self.conv2_1_zp = ZeroPaddingDepth(target_depth = 256, name = "conv2_block1_depth-zeropadding")
        self.conv2_1_add = keras.layers.Add(name = "conv2_block1_out")
        
        self.conv2_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block2_preact_bn")
        self.conv2_2_preact_relu = keras.layers.ReLU(name = f"conv2_block2_preact_relu")
        self.conv2_2_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block2")
        self.conv2_2_add = keras.layers.Add(name = "conv2_block2_out")
        
        self.conv2_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block3_preact_bn")
        self.conv2_3_preact_relu = keras.layers.ReLU(name = f"conv2_block3_preact_relu")
        self.conv2_3_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block3")
        self.conv2_3_add = keras.layers.Add(name = "conv2_block3_out")
        
        
        ## conv3_x
        self.conv3_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block1_preact_bn")
        self.conv3_1_preact_relu = keras.layers.ReLU(name = f"conv3_block1_preact_relu")
        self.conv2_crop = CentralCrop2D((60, 60), name = "conv2_crop")
        
        self.conv3_1_F = PreactRes50_BlockF2(filters = 128, first_conv2d_strides = (2, 2), name = "conv3_block1")
        self.conv3_1_downsample = keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv3_block1_downsample")
        self.conv3_1_zp = ZeroPaddingDepth(target_depth = 512, name = "conv3_block1_depth-zeropadding")
        self.conv3_1_add = keras.layers.Add(name = "conv3_block1_out")
        
        
        self.conv3_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block2_preact_bn")
        self.conv3_2_preact_relu = keras.layers.ReLU(name = f"conv3_block2_preact_relu")

        self.conv3_2_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block2")
        self.conv3_2_add = keras.layers.Add(name = "conv3_block2_out")
        
        self.conv3_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block3_preact_bn")
        self.conv3_3_preact_relu = keras.layers.ReLU(name = f"conv3_block3_preact_relu")
        self.conv3_3_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block3")
        self.conv3_3_add = keras.layers.Add(name = "conv3_block3_out")
        
        
        self.conv3_4_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block4_preact_bn")
        self.conv3_4_preact_relu = keras.layers.ReLU(name = f"conv3_block4_preact_relu")
        self.conv3_4_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block4")
        self.conv3_4_add = keras.layers.Add(name = "conv3_block4_out")
        
        
        ## conv4_x
        self.conv4_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block1_preact_bn")
        self.conv4_1_preact_relu = keras.layers.ReLU(name = f"conv4_block1_preact_relu")
        self.conv3_crop = CentralCrop2D((36, 36), name = "conv3_crop")
        
        self.conv4_1_F = PreactRes50_BlockF2(filters = 256, first_conv2d_strides = (2, 2), name = "conv4_block1")
        self.conv4_1_downsample = keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv4_block1_downsample")
        self.conv4_1_zp = ZeroPaddingDepth(target_depth = 1024, name = "conv4_block1_depth-zeropadding")
        self.conv4_1_add = keras.layers.Add(name = "conv4_block1_out")
        
        
        self.conv4_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block2_preact_bn")
        self.conv4_2_preact_relu = keras.layers.ReLU(name = f"conv4_block2_preact_relu")
        self.conv4_2_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block2")
        self.conv4_2_add = keras.layers.Add(name = "conv4_block2_out")
        
        
        self.conv4_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block3_preact_bn")
        self.conv4_3_preact_relu = keras.layers.ReLU(name = f"conv4_block3_preact_relu")
        self.conv4_3_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block3")
        self.conv4_3_add = keras.layers.Add(name = "conv4_block3_out")
        
        
        self.conv4_4_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block4_preact_bn")
        self.conv4_4_preact_relu = keras.layers.ReLU(name = f"conv4_block4_preact_relu")
        self.conv4_4_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block4")
        self.conv4_4_add = keras.layers.Add(name = "conv4_block4_out")
        
        
        self.conv4_5_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block5_preact_bn")
        self.conv4_5_preact_relu = keras.layers.ReLU(name = f"conv4_block5_preact_relu")
        self.conv4_5_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block5")
        self.conv4_5_add = keras.layers.Add(name = "conv4_block5_out")
        
        
        self.conv4_6_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block6_preact_bn")
        self.conv4_6_preact_relu = keras.layers.ReLU(name = f"conv4_block6_preact_relu")
        self.conv4_6_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block6")
        self.conv4_6_add = keras.layers.Add(name = "conv4_block6_out")
        
        
        ## conv5_x
        self.conv5_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block1_preact_bn")
        self.conv5_1_preact_relu = keras.layers.ReLU(name = f"conv5_block1_preact_relu")
        self.conv4_crop = CentralCrop2D((24, 24), name = "conv4_crop")
        
        self.conv5_1_F = PreactRes50_BlockF2(filters = 512, first_conv2d_strides = (2, 2), name = "conv5_block1F")
        self.conv5_1_downsample = keras.layers.Conv2D(filters = 1024, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv5_block1_downsample")
        self.conv5_1_zp = ZeroPaddingDepth(target_depth = 2048, name = "conv5_block1_depth-zeropadding")
        self.conv5_1_add = keras.layers.Add(name = "conv5_block1_out")
        
        
        self.conv5_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block2_preact_bn")
        self.conv5_2_preact_relu = keras.layers.ReLU(name = f"conv5_block2_preact_relu")
        self.conv5_2_F = PreactRes50_BlockF2(filters = 512, name = "conv5_block2")
        self.conv5_2_add = keras.layers.Add(name = "conv5_block2_out")
        
        
        self.conv5_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block3_preact_bn")
        self.conv5_3_preact_relu = keras.layers.ReLU(name = f"conv5_block3_preact_relu")
        self.conv5_3_F = PreactRes50_BlockF2(filters = 512, name = "conv5_block3")
        self.conv5_3_add = keras.layers.Add(name = "conv5_block3_out")
        
        
        self.post_bn = keras.layers.BatchNormalization(axis = -1, name = f"Post_BatchNormalization")
        self.post_relu = keras.layers.ReLU(name = f"Post_Relu")
        
        
        ## decoder 4
        self.dec4_transpose = keras.layers.Conv2DTranspose(filters = 1024, kernel_size = (2, 2), strides = (2, 2))
        
        
        ## decoder 3
        self.dec3_add = keras.layers.Add(name = "dec3_add")
        self.dec3_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec3")
        self.dec3_rl = keras.layers.ReLU(name = f"Relu_dec3")
        self.dec3 = DecoderBlock_GC(cardinality = 256, filters = 512, name = 'dec3')
        self.dec3_rs = Resize2D((36, 36), name = "dec3_rs")
        
        
        ## decoder 2
        self.dec2_add = keras.layers.Add(name = "dec2_add")
        self.dec2_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec2")
        self.dec2_rl = keras.layers.ReLU(name = f"Relu_dec2")
        self.dec2 = DecoderBlock_GC(cardinality = 128, filters = 256, name = 'dec2')
        self.dec2_rs = Resize2D((60, 60), name = "dec2_rs")
        
        
        ## decoder 1
        self.dec1_add = keras.layers.Add(name = "dec1_add")
        self.dec1_bn1 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_1_dec1")
        self.dec1_rl1 = keras.layers.ReLU(name = f"Relu_1_dec1")
        self.dec1 = DecoderBlock_GC(cardinality = 64, filters = 128, name = 'dec1')
        self.dec1_bn2 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_2_dec1")
        self.dec1_rl2 = keras.layers.ReLU(name = f"Relu_2_dec1")       
  
        
        ### output sigmoid
        self.out_conv = keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                            activation = 'sigmoid', kernel_initializer = "he_normal", name = f"out_conv", dtype = 'float32')
#         self.out_act = keras.layers.Activation('linear', dtype='float32', name = "linear_to_float32") # For using mixed precision. The output dtype should be float32, not float16.
        
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        
    def call(self, inputs):
        ### ouput node connection
        Z = inputs
            
            
        ## conv1    
        Zo = self.conv1(Z)
        
        
        ## conv2
        Z = self.conv2_1_preact_bn(Zo)
        Z = self.conv2_1_preact_relu(Z)
        Z = self.conv2_1_F(Z)
        Zs = self.conv2_1_zp(Zo)
        Zo = self.conv2_1_add([Z, Zs])
        
        Z = self.conv2_2_preact_bn(Zo)
        Z = self.conv2_2_preact_relu(Z)
        Z = self.conv2_2_F(Z)
        Zo = self.conv2_2_add([Zo, Z])
        
        Z = self.conv2_3_preact_bn(Zo)
        Z = self.conv2_3_preact_relu(Z)
        Z = self.conv2_3_F(Z)
        Zo = self.conv2_3_add([Zo, Z])
        
        
        ## conv3
        Z = self.conv3_1_preact_bn(Zo)
        Z = self.conv3_1_preact_relu(Z)
        Z2 = self.conv2_crop(Z)
        
        Z = self.conv3_1_F(Z)
        Zs = self.conv3_1_downsample(Zo)
        Zs = self.conv3_1_zp(Zs)
        Zo = self.conv3_1_add([Z, Zs])
        
        Z = self.conv3_2_preact_bn(Zo)
        Z = self.conv3_2_preact_relu(Z)
        Z = self.conv3_2_F(Z)
        Zo = self.conv3_2_add([Zo, Z])
        
        Z = self.conv3_3_preact_bn(Zo)
        Z = self.conv3_3_preact_relu(Z)
        Z = self.conv3_3_F(Z)
        Zo = self.conv3_3_add([Zo, Z])
        
        Z = self.conv3_4_preact_bn(Zo)
        Z = self.conv3_4_preact_relu(Z)
        Z = self.conv3_4_F(Z)
        Zo = self.conv3_4_add([Zo, Z])
        
        
        ## conv4
        Z = self.conv4_1_preact_bn(Zo)
        Z = self.conv4_1_preact_relu(Z)
        Z3 = self.conv3_crop(Z)
        Z = self.conv4_1_F(Z)
        
        Zs = self.conv4_1_downsample(Zo)
        Zs = self.conv4_1_zp(Zs)
        Zo = self.conv4_1_add([Z, Zs])
        
        Z = self.conv4_2_preact_bn(Zo)
        Z = self.conv4_2_preact_relu(Z)
        Z = self.conv4_2_F(Z)
        Zo = self.conv4_2_add([Zo, Z])
        
        Z = self.conv4_3_preact_bn(Zo)
        Z = self.conv4_3_preact_relu(Z)
        Z = self.conv4_3_F(Z)
        Zo = self.conv4_3_add([Zo, Z])
        
        Z = self.conv4_4_preact_bn(Zo)
        Z = self.conv4_4_preact_relu(Z)
        Z = self.conv4_4_F(Z)
        Zo = self.conv4_4_add([Zo, Z])
        
        Z = self.conv4_5_preact_bn(Zo)
        Z = self.conv4_5_preact_relu(Z)
        Z = self.conv4_5_F(Z)
        Zo = self.conv4_5_add([Zo, Z])
        
        Z = self.conv4_6_preact_bn(Zo)
        Z = self.conv4_6_preact_relu(Z)
        Z = self.conv4_6_F(Z)
        Zo = self.conv4_6_add([Zo, Z])
        
        
        ## conv 5
        Z = self.conv5_1_preact_bn(Zo)
        Z = self.conv5_1_preact_relu(Z)
        Z4 = self.conv4_crop(Z)
        
        Z = self.conv5_1_F(Z)
        Zs = self.conv5_1_downsample(Zo)
        Zs = self.conv5_1_zp(Zs)
        Zo = self.conv5_1_add([Z, Zs])
        
        Z = self.conv5_2_preact_bn(Zo)
        Z = self.conv5_2_preact_relu(Z)
        Z = self.conv5_2_F(Z)
        Zo = self.conv5_2_add([Zo, Z])
        
        Z = self.conv5_3_preact_bn(Zo)
        Z = self.conv5_3_preact_relu(Z)
        Z = self.conv5_3_F(Z)
        Z = self.conv5_3_add([Zo, Z])
        
        Z = self.post_bn(Z)
        Zo = self.post_relu(Z)
        
#         return Zo
        
        
        ## decoder 4(Orange color box in the article)
        Zo = self.dec4_transpose(Zo)
        
        ## decoder 3
        Z = self.dec3_add([Zo, Z4])
        Z = self.dec3_bn(Z)
        Z = self.dec3_rl(Z)
        Z = self.dec3(Z)
        Zo = self.dec3_rs(Z)
        
        ## decoder 2
        Z = self.dec2_add([Zo, Z3])
        Z = self.dec2_bn(Z)
        Z = self.dec2_rl(Z)
        Z = self.dec2(Z)
        Zo = self.dec2_rs(Z)
        
        
        ## decoder 1
        Z = self.dec1_add([Zo, Z2])
        Z = self.dec1_bn1(Z)
        Z = self.dec1_rl1(Z)
        Z = self.dec1(Z)
        Z = self.dec1_bn2(Z)
        Zo = self.dec1_rl2(Z)
        
        
        ## out
        dec_out = self.out_conv(Zo)
#         dec_out = self.out_act(Z)
#         print(Zo.dtype, self.out_conv.kernel.dtype, dec_out.dtype)
#         tf.print(dec_out)
        
        return dec_out
    
    
### Preact Resnet50 Segmentation model in https://arxiv.org/abs/1810.13230
class PreactResnetModified4(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        ### layers definition 
        ## conv1
        self.conv1 = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides=(1, 1), padding = "valid",
                                            data_format = 'channels_last', kernel_initializer = "he_normal", name = "conv1")
        
        ## conv2_x
        self.conv2_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block1_preact_bn")
        self.conv2_1_preact_relu = keras.layers.ReLU(name = f"conv2_block1_preact_relu")
        self.conv2_1_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block1")
        self.conv2_1_zp = ZeroPaddingDepth(target_depth = 256, name = "conv2_block1_depth-zeropadding")
        self.conv2_1_add = keras.layers.Add(name = "conv2_block1_out")
        
        self.conv2_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block2_preact_bn")
        self.conv2_2_preact_relu = keras.layers.ReLU(name = f"conv2_block2_preact_relu")
        self.conv2_2_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block2")
        self.conv2_2_add = keras.layers.Add(name = "conv2_block2_out")
        
        self.conv2_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv2_block3_preact_bn")
        self.conv2_3_preact_relu = keras.layers.ReLU(name = f"conv2_block3_preact_relu")
        self.conv2_3_F = PreactRes50_BlockF2(filters = 64, name = "conv2_block3")
        self.conv2_3_add = keras.layers.Add(name = "conv2_block3_out")
        
        
        ## conv3_x
        self.conv3_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block1_preact_bn")
        self.conv3_1_preact_relu = keras.layers.ReLU(name = f"conv3_block1_preact_relu")
        self.conv2_crop = CentralCrop2D((60, 60), name = "conv2_crop")
        
        self.conv3_1_F = PreactRes50_BlockF2(filters = 128, first_conv2d_strides = (2, 2), name = "conv3_block1")
        self.conv3_1_downsample = keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv3_block1_downsample")
        self.conv3_1_zp = ZeroPaddingDepth(target_depth = 512, name = "conv3_block1_depth-zeropadding")
        self.conv3_1_add = keras.layers.Add(name = "conv3_block1_out")
        
        
        self.conv3_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block2_preact_bn")
        self.conv3_2_preact_relu = keras.layers.ReLU(name = f"conv3_block2_preact_relu")

        self.conv3_2_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block2")
        self.conv3_2_add = keras.layers.Add(name = "conv3_block2_out")
        
        self.conv3_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block3_preact_bn")
        self.conv3_3_preact_relu = keras.layers.ReLU(name = f"conv3_block3_preact_relu")
        self.conv3_3_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block3")
        self.conv3_3_add = keras.layers.Add(name = "conv3_block3_out")
        
        
        self.conv3_4_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv3_block4_preact_bn")
        self.conv3_4_preact_relu = keras.layers.ReLU(name = f"conv3_block4_preact_relu")
        self.conv3_4_F = PreactRes50_BlockF2(filters = 128, name = "conv3_block4")
        self.conv3_4_add = keras.layers.Add(name = "conv3_block4_out")
        
        
        ## conv4_x
        self.conv4_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block1_preact_bn")
        self.conv4_1_preact_relu = keras.layers.ReLU(name = f"conv4_block1_preact_relu")
        self.conv3_crop = CentralCrop2D((36, 36), name = "conv3_crop")
        
        self.conv4_1_F = PreactRes50_BlockF2(filters = 256, first_conv2d_strides = (2, 2), name = "conv4_block1")
        self.conv4_1_downsample = keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv4_block1_downsample")
        self.conv4_1_zp = ZeroPaddingDepth(target_depth = 1024, name = "conv4_block1_depth-zeropadding")
        self.conv4_1_add = keras.layers.Add(name = "conv4_block1_out")
        
        
        self.conv4_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block2_preact_bn")
        self.conv4_2_preact_relu = keras.layers.ReLU(name = f"conv4_block2_preact_relu")
        self.conv4_2_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block2")
        self.conv4_2_add = keras.layers.Add(name = "conv4_block2_out")
        
        
        self.conv4_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block3_preact_bn")
        self.conv4_3_preact_relu = keras.layers.ReLU(name = f"conv4_block3_preact_relu")
        self.conv4_3_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block3")
        self.conv4_3_add = keras.layers.Add(name = "conv4_block3_out")
        
        
        self.conv4_4_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block4_preact_bn")
        self.conv4_4_preact_relu = keras.layers.ReLU(name = f"conv4_block4_preact_relu")
        self.conv4_4_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block4")
        self.conv4_4_add = keras.layers.Add(name = "conv4_block4_out")
        
        
        self.conv4_5_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block5_preact_bn")
        self.conv4_5_preact_relu = keras.layers.ReLU(name = f"conv4_block5_preact_relu")
        self.conv4_5_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block5")
        self.conv4_5_add = keras.layers.Add(name = "conv4_block5_out")
        
        
        self.conv4_6_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv4_block6_preact_bn")
        self.conv4_6_preact_relu = keras.layers.ReLU(name = f"conv4_block6_preact_relu")
        self.conv4_6_F = PreactRes50_BlockF2(filters = 256, name = "conv4_block6")
        self.conv4_6_add = keras.layers.Add(name = "conv4_block6_out")
        
        
        ## conv5_x
        self.conv5_1_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block1_preact_bn")
        self.conv5_1_preact_relu = keras.layers.ReLU(name = f"conv5_block1_preact_relu")
        self.conv4_crop = CentralCrop2D((24, 24), name = "conv4_crop")
        
        self.conv5_1_F = PreactRes50_BlockF2(filters = 512, first_conv2d_strides = (2, 2), name = "conv5_block1F")
        self.conv5_1_downsample = keras.layers.Conv2D(filters = 1024, kernel_size = (1, 1), strides=(2, 2), padding = "valid",
                                            kernel_initializer = "he_normal", name = "conv5_block1_downsample")
        self.conv5_1_zp = ZeroPaddingDepth(target_depth = 2048, name = "conv5_block1_depth-zeropadding")
        self.conv5_1_add = keras.layers.Add(name = "conv5_block1_out")
        
        
        self.conv5_2_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block2_preact_bn")
        self.conv5_2_preact_relu = keras.layers.ReLU(name = f"conv5_block2_preact_relu")
        self.conv5_2_F = PreactRes50_BlockF2(filters = 512, name = "conv5_block2")
        self.conv5_2_add = keras.layers.Add(name = "conv5_block2_out")
        
        
        self.conv5_3_preact_bn = keras.layers.BatchNormalization(axis = -1, name = f"conv5_block3_preact_bn")
        self.conv5_3_preact_relu = keras.layers.ReLU(name = f"conv5_block3_preact_relu")
        self.conv5_3_F = PreactRes50_BlockF2(filters = 512, name = "conv5_block3")
        self.conv5_3_add = keras.layers.Add(name = "conv5_block3_out")
        
        
        self.post_bn = keras.layers.BatchNormalization(axis = -1, name = f"Post_BatchNormalization")
        self.post_relu = keras.layers.ReLU(name = f"Post_Relu")
        
        
        ## decoder 4
        self.dec4_transpose = keras.layers.Conv2DTranspose(filters = 1024, kernel_size = (2, 2), strides = (2, 2))
        
        
        ## decoder 3
        self.dec3_add = keras.layers.Add(name = "dec3_add")
        self.dec3_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec3")
        self.dec3_rl = keras.layers.ReLU(name = f"Relu_dec3")
        self.dec3 = DecoderBlock_GC(cardinality = 256, filters = 512, name = 'dec3')
        self.dec3_rs = Resize2D((36, 36), name = "dec3_rs")
        
        
        ## decoder 2
        self.dec2_add = keras.layers.Add(name = "dec2_add")
        self.dec2_bn = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_dec2")
        self.dec2_rl = keras.layers.ReLU(name = f"Relu_dec2")
        self.dec2 = DecoderBlock_GC(cardinality = 128, filters = 256, name = 'dec2')
        self.dec2_rs = Resize2D((60, 60), name = "dec2_rs")
        
        
        ## decoder 1
        self.dec1_add = keras.layers.Add(name = "dec1_add")
        self.dec1_bn1 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_1_dec1")
        self.dec1_rl1 = keras.layers.ReLU(name = f"Relu_1_dec1")
        self.dec1 = DecoderBlock_GC(cardinality = 64, filters = 128, name = 'dec1')
        self.dec1_bn2 = keras.layers.BatchNormalization(axis = -1, name = f"BatchNormalization_2_dec1")
        self.dec1_rl2 = keras.layers.ReLU(name = f"Relu_2_dec1")       
  
        
        ### output sigmoid
        self.out_conv = keras.layers.Conv2D(filters = 2, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                            activation = 'softmax', kernel_initializer = "he_normal", name = f"out_conv", dtype = 'float32')
#         self.out_act = keras.layers.Activation('linear', dtype='float32', name = "linear_to_float32") # For using mixed precision. The output dtype should be float32, not float16.
        
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
        
    def call(self, inputs):
        ### ouput node connection
        Z = inputs
            
            
        ## conv1    
        Zo = self.conv1(Z)
        
        
        ## conv2
        Z = self.conv2_1_preact_bn(Zo)
        Z = self.conv2_1_preact_relu(Z)
        Z = self.conv2_1_F(Z)
        Zs = self.conv2_1_zp(Zo)
        Zo = self.conv2_1_add([Z, Zs])
        
        Z = self.conv2_2_preact_bn(Zo)
        Z = self.conv2_2_preact_relu(Z)
        Z = self.conv2_2_F(Z)
        Zo = self.conv2_2_add([Zo, Z])
        
        Z = self.conv2_3_preact_bn(Zo)
        Z = self.conv2_3_preact_relu(Z)
        Z = self.conv2_3_F(Z)
        Zo = self.conv2_3_add([Zo, Z])
        
        
        ## conv3
        Z = self.conv3_1_preact_bn(Zo)
        Z = self.conv3_1_preact_relu(Z)
        Z2 = self.conv2_crop(Z)
        
        Z = self.conv3_1_F(Z)
        Zs = self.conv3_1_downsample(Zo)
        Zs = self.conv3_1_zp(Zs)
        Zo = self.conv3_1_add([Z, Zs])
        
        Z = self.conv3_2_preact_bn(Zo)
        Z = self.conv3_2_preact_relu(Z)
        Z = self.conv3_2_F(Z)
        Zo = self.conv3_2_add([Zo, Z])
        
        Z = self.conv3_3_preact_bn(Zo)
        Z = self.conv3_3_preact_relu(Z)
        Z = self.conv3_3_F(Z)
        Zo = self.conv3_3_add([Zo, Z])
        
        Z = self.conv3_4_preact_bn(Zo)
        Z = self.conv3_4_preact_relu(Z)
        Z = self.conv3_4_F(Z)
        Zo = self.conv3_4_add([Zo, Z])
        
        
        ## conv4
        Z = self.conv4_1_preact_bn(Zo)
        Z = self.conv4_1_preact_relu(Z)
        Z3 = self.conv3_crop(Z)
        Z = self.conv4_1_F(Z)
        
        Zs = self.conv4_1_downsample(Zo)
        Zs = self.conv4_1_zp(Zs)
        Zo = self.conv4_1_add([Z, Zs])
        
        Z = self.conv4_2_preact_bn(Zo)
        Z = self.conv4_2_preact_relu(Z)
        Z = self.conv4_2_F(Z)
        Zo = self.conv4_2_add([Zo, Z])
        
        Z = self.conv4_3_preact_bn(Zo)
        Z = self.conv4_3_preact_relu(Z)
        Z = self.conv4_3_F(Z)
        Zo = self.conv4_3_add([Zo, Z])
        
        Z = self.conv4_4_preact_bn(Zo)
        Z = self.conv4_4_preact_relu(Z)
        Z = self.conv4_4_F(Z)
        Zo = self.conv4_4_add([Zo, Z])
        
        Z = self.conv4_5_preact_bn(Zo)
        Z = self.conv4_5_preact_relu(Z)
        Z = self.conv4_5_F(Z)
        Zo = self.conv4_5_add([Zo, Z])
        
        Z = self.conv4_6_preact_bn(Zo)
        Z = self.conv4_6_preact_relu(Z)
        Z = self.conv4_6_F(Z)
        Zo = self.conv4_6_add([Zo, Z])
        
        
        ## conv 5
        Z = self.conv5_1_preact_bn(Zo)
        Z = self.conv5_1_preact_relu(Z)
        Z4 = self.conv4_crop(Z)
        
        Z = self.conv5_1_F(Z)
        Zs = self.conv5_1_downsample(Zo)
        Zs = self.conv5_1_zp(Zs)
        Zo = self.conv5_1_add([Z, Zs])
        
        Z = self.conv5_2_preact_bn(Zo)
        Z = self.conv5_2_preact_relu(Z)
        Z = self.conv5_2_F(Z)
        Zo = self.conv5_2_add([Zo, Z])
        
        Z = self.conv5_3_preact_bn(Zo)
        Z = self.conv5_3_preact_relu(Z)
        Z = self.conv5_3_F(Z)
        Z = self.conv5_3_add([Zo, Z])
        
        Z = self.post_bn(Z)
        Zo = self.post_relu(Z)
        
#         return Zo
        
        
        ## decoder 4(Orange color box in the article)
        Zo = self.dec4_transpose(Zo)
        
        ## decoder 3
        Z = self.dec3_add([Zo, Z4])
        Z = self.dec3_bn(Z)
        Z = self.dec3_rl(Z)
        Z = self.dec3(Z)
        Zo = self.dec3_rs(Z)
        
        ## decoder 2
        Z = self.dec2_add([Zo, Z3])
        Z = self.dec2_bn(Z)
        Z = self.dec2_rl(Z)
        Z = self.dec2(Z)
        Zo = self.dec2_rs(Z)
        
        
        ## decoder 1
        Z = self.dec1_add([Zo, Z2])
        Z = self.dec1_bn1(Z)
        Z = self.dec1_rl1(Z)
        Z = self.dec1(Z)
        Z = self.dec1_bn2(Z)
        Zo = self.dec1_rl2(Z)
        
        
        ## out
        dec_out = self.out_conv(Zo)
#         dec_out = self.out_act(Z)
#         print(Zo.dtype, self.out_conv.kernel.dtype, dec_out.dtype)
#         tf.print(dec_out)
        
        return dec_out
    
    
### DiceLoss
class DiceLoss(keras.losses.Loss):
    def __init__(self, calc_axis = [-3, -2], epsilon = 1e-8, **kwargs):
        """
        calc_axis : axis when calculating intersection and union. 
                    e.g. when using 2d images and the data format is 'channels_last', axis could be (-3, -2)
        """
        super().__init__(**kwargs)
        
        self.calc_axis = calc_axis
        self.epsilon = epsilon
        
    
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis = self.calc_axis)
        dn = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis = self.calc_axis) + self.epsilon
        return 1 - tf.reduce_mean(2 * intersection / dn, axis = -1)
    
    
### dice metric
def dice_coef_metric(epsilon = 1e-8, calc_axis = [-3,-2]):

    def dice_coefficient(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis = calc_axis)
        dn = K.sum(K.square(y_true) + K.square(y_pred), axis = calc_axis) + epsilon
        return K.mean(2 * intersection / dn, axis = -1)
    
    return dice_coefficient
    
    
    
### VAELossLayer
class LossVAE(keras.layers.Layer):
    def __init__(self, weight_L2, weight_KL, n, **kwargs):
        super().__init__(**kwargs)
        
        self.weight_L2 = weight_L2
        self.weight_KL = weight_KL
        self.n = n
        
    def call(self, inputs):
        x, out_VAE, z_mean, z_var = inputs
        loss_L2 = tf.reduce_mean(tf.square(x - out_VAE), axis=(1, 2, 3, 4)) # original axis value is (1,2,3,4).
        loss_KL = (1 / self.n) * tf.reduce_sum(
            tf.exp(z_var) + tf.square(z_mean) - 1. - z_var,
            axis=-1
        )
        
        VAE_loss = tf.reduce_mean(tf.add(self.weight_L2 * loss_L2, self.weight_KL * loss_KL, name = "add_L2_KL"), name = "mean_VAELoss")
        self.add_loss(VAE_loss)
        
        return     
    

### 3D autoencoder regularization model's green block
class green_block(keras.layers.Layer):
    def __init__(self, filters, regularizer, data_format='channels_first', name=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            Conv3D(filters=filters, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = regularizer, data_format=data_format, name=f'Res_{name}' if name else None),
            GroupNormalization(groups = 8, axis = 1 if data_format == 'channels_first' else 0, name = f'GroupNorm_1_{name}' if name else None),
            Activation('relu', name=f'Relu_1_{name}' if name else None),
            Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = regularizer, data_format=data_format, name=f'Conv3D_1_{name}' if name else None),
            GroupNormalization(groups = 8, axis = 1 if data_format == 'channels_first' else 0, name = f'GroupNorm_2_{name}' if name else None),
            Activation('relu', name=f'Relu_2_{name}' if name else None),
            Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = regularizer, data_format=data_format, name=f'Conv3D_2_{name}' if name else None),
            Add(name=f'Out_{name}' if name else None)
        ]
    
    def call(self, inputs):
        Z = inputs
        inp_res = self.hidden[0](Z)
        Z = self.hidden[1](Z)
        
        for layer in self.hidden[2:7]:
            Z = layer(Z)
            
        Z = self.hidden[7]([Z, inp_res])
        
        return Z
    
    
    
### 3D autoencoder regularization model
class conv3d_autoenc_reg(keras.Model):
    def __init__(self, input_shape=(4, 160, 192, 128), output_channels=3, l2_reg_weight = 1e-5, weight_L2=0.1, weight_KL=0.1, 
                 dice_e=1e-8, test_mode = True, n_gpu = 1, GL_weight = 1, VL_weight = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.c, self.H, self.W, self.D = input_shape
        self.n = self.c * self.H * self.W * self.D
        assert len(input_shape) == 4, "Input shape must be a 4-tuple"
        if test_mode is not True: assert (self.c % 4) == 0, "The no. of channels must be divisible by 4"
        assert (self.H % 16) == 0 and (self.W % 16) == 0 and (self.D % 16) == 0, "All the input dimensions must be divisible by 16"
        self.l2_regularizer = l2(l2_reg_weight) if l2_reg_weight is not None else None
        
        self.input_shape_p = input_shape
        self.output_channels = output_channels
        self.l2_reg_weight = l2_reg_weight
        self.weight_L2 = weight_L2
        self.weight_KL = weight_KL
        self.dice_e = dice_e
        self.GL_weight = GL_weight
        self.VL_weight = VL_weight
        
        self.LossVAE = LossVAE(weight_L2, weight_KL, self.n)
        
        ## The Initial Block
        self.Input_x1 = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        kernel_regularizer = self.l2_regularizer,
        data_format='channels_first',
        name='Input_x1')
        
        ## Dropout (0.2)
        self.spatial_dropout = SpatialDropout3D(0.2, data_format='channels_first')
        
        ## Green Block x1 (output filters = 32)
        self.x1 = green_block(32, regularizer = self.l2_regularizer, name='x1')
        self.Enc_DownSample_32 = Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=2,
            padding='same',
            kernel_regularizer = self.l2_regularizer,
            data_format='channels_first',
            name='Enc_DownSample_32')
        
        ## Green Block x2 (output filters = 64)
        self.Enc_64_1 = green_block(64, regularizer = self.l2_regularizer, name='Enc_64_1')
        self.x2 = green_block(64, regularizer = self.l2_regularizer, name='x2')
        self.Enc_DownSample_64 = Conv3D(
                            filters=64,
                            kernel_size=(3, 3, 3),
                            strides=2,
                            padding='same',
                            kernel_regularizer = self.l2_regularizer,
                            data_format='channels_first',
                            name='Enc_DownSample_64')
        
        ## Green Blocks x2 (output filters = 128)
        self.Enc_128_1 = green_block(128, regularizer = self.l2_regularizer, name='Enc_128_1')
        self.x3 = green_block(128, regularizer = self.l2_regularizer, name='x3')
        self.Enc_DownSample_128 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_regularizer = self.l2_regularizer, 
                                         data_format='channels_first', name='Enc_DownSample_128')
        
        ## Green Blocks x4 (output filters = 256)
        self.Enc_256_1 = green_block(256, regularizer = self.l2_regularizer, name='Enc_256_1')
        self.Enc_256_2 = green_block(256, regularizer = self.l2_regularizer, name='Enc_256_2')
        self.Enc_256_3 = green_block(256, regularizer = self.l2_regularizer, name='Enc_256_3')
        self.x4 = green_block(256, regularizer = self.l2_regularizer, name='x4')
        
        # -------------------------------------------------------------------------
        # Decoder
        # -------------------------------------------------------------------------

        ## GT (Groud Truth) Part
        # -------------------------------------------------------------------------
        
        ### Green Block x1 (output filters=128)
        self.Dec_GT_ReduceDepth_128 = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', name='Dec_GT_ReduceDepth_128')
        self.Dec_GT_UpSample_128 = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_128') 
        self.Input_Dec_GT_128 = Add(name='Input_Dec_GT_128')
        self.Dec_GT_128 = green_block(128, regularizer = self.l2_regularizer, name='Dec_GT_128')
        
        ### Green Block x1 (output filters=64)
        self.Dec_GT_ReduceDepth_64 = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', name='Dec_GT_ReduceDepth_64')
        self.Dec_GT_UpSample_64 = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_64')
        self.Input_Dec_GT_64 = Add(name='Input_Dec_GT_64')
        self.Dec_GT_64 = green_block(64, regularizer = self.l2_regularizer, name='Dec_GT_64')
        
        ### Green Block x1 (output filters=32)
        self.Dec_GT_ReduceDepth_32 = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', 
                                       name='Dec_GT_ReduceDepth_32')
        self.Dec_GT_UpSample_32 = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_32')
        self.Input_Dec_GT_32 = Add(name='Input_Dec_GT_32')
        self.Dec_GT_32 = green_block(32, regularizer = self.l2_regularizer, name='Dec_GT_32')
        
        ### Blue Block x1 (output filters=32)
        self.Input_Dec_GT_Output = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = self.l2_regularizer, 
                                     data_format='channels_first', name='Input_Dec_GT_Output')
        
        ### Output Block
        self.Dec_GT_Output = Conv3D(filters=self.output_channels, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, 
                                data_format='channels_first', activation='sigmoid', name='Dec_GT_Output')
        
        ## VAE (Variational Auto Encoder) Part
        # -------------------------------------------------------------------------

        ### VD Block (Reducing dimensionality of the data)
        self.Dec_VAE_VD_GN = GroupNormalization(groups=8, axis=1, name='Dec_VAE_VD_GN')
        self.Dec_VAE_VD_relu = Activation('relu', name='Dec_VAE_VD_relu')
        self.Dec_VAE_VD_Conv3D = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_regularizer = self.l2_regularizer, 
                                   data_format='channels_first', name='Dec_VAE_VD_Conv3D')
        
        # Not mentioned in the paper, but the author used a Flattening layer here.
        self.Dec_VAE_VD_Flatten = Flatten(name='Dec_VAE_VD_Flatten')
        self.Dec_VAE_VD_Dense = Dense(256, name='Dec_VAE_VD_Dense')

        ### VDraw Block (Sampling)
        self.Dec_VAE_VDraw_Mean = Dense(128, name='Dec_VAE_VDraw_Mean')
        self.Dec_VAE_VDraw_Var = Dense(128, name='Dec_VAE_VDraw_Var')
#         self.Dec_VAE_VDraw_Sampling = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')
        self.Dec_VAE_VDraw_Sampling = sampling()

        ### VU Block (Upsizing back to a depth of 256)
        c1 = 1
        self.VU_Dense1 = Dense((c1) * (self.H//16) * (self.W//16) * (self.D//16))
        self.VU_relu = Activation('relu')
        self.VU_reshape = Reshape(((c1), (self.H//16), (self.W//16), (self.D//16)))
        self.Dec_VAE_ReduceDepth_256 = Conv3D(filters=256, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first',
                                            name='Dec_VAE_ReduceDepth_256')
        self.Dec_VAE_UpSample_256 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_256')

        ### Green Block x1 (output filters=128)
        self.Dec_VAE_ReduceDepth_128 = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', 
                                         name='Dec_VAE_ReduceDepth_128')
        self.Dec_VAE_UpSample_128 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_128')
        self.Dec_VAE_128 = green_block(128, regularizer = self.l2_regularizer, name='Dec_VAE_128')

        ### Green Block x1 (output filters=64)
        self.Dec_VAE_ReduceDepth_64 = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first',
                                        name='Dec_VAE_ReduceDepth_64')
        self.Dec_VAE_UpSample_64 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_64')
        self.Dec_VAE_64 = green_block(64, regularizer = self.l2_regularizer, name='Dec_VAE_64')

        ### Green Block x1 (output filters=32)
        self.Dec_VAE_ReduceDepth_32 = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first',
                                        name='Dec_VAE_ReduceDepth_32')
        self.Dec_VAE_UpSample_32 = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_32')
        self.Dec_VAE_32 = green_block(32, regularizer = self.l2_regularizer, name='Dec_VAE_32')

        ### Blue Block x1 (output filters=32)
        self.Input_Dec_VAE_Output = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_regularizer = self.l2_regularizer, 
                                      data_format='channels_first', name='Input_Dec_VAE_Output')

        ### Output Block
        self.Dec_VAE_Output = Conv3D(filters=self.c, kernel_size=(1, 1, 1), strides=1, kernel_regularizer = self.l2_regularizer, data_format='channels_first', 
                                     name='Dec_VAE_Output')
        
#     def build(self, batch_input_shape):
#         n_inputs = batch_input_shape[-1]
        
#         ### super build
#         super().build(batch_input_shape)
        
    def call(self, inputs, training=None):
        Z = inputs
        x = self.Input_x1(Z)
        
        ## Dropout (0.2)
        x = self.spatial_dropout(x)

        ## Green Block x1 (output filters = 32)
        x1 = self.x1(x)
        x = self.Enc_DownSample_32(x1)

        ## Green Block x2 (output filters = 64)
        x = self.Enc_64_1(x)
        x2 = self.x2(x)
        x = self.Enc_DownSample_64(x2)

        ## Green Blocks x2 (output filters = 128)
        x = self.Enc_128_1(x)
        x3 = self.x3(x)
        x = self.Enc_DownSample_128(x3)

        ## Green Blocks x4 (output filters = 256)
        x = self.Enc_256_1(x)
        x = self.Enc_256_2(x)
        x = self.Enc_256_3(x)
        x4 = self.x4(x)

        # -------------------------------------------------------------------------
        # Decoder
        # -------------------------------------------------------------------------

        ## GT (Groud Truth) Part
        # -------------------------------------------------------------------------

        ### Green Block x1 (output filters=128)
        x = self.Dec_GT_ReduceDepth_128(x4)
        x = self.Dec_GT_UpSample_128(x)
        x = self.Input_Dec_GT_128([x, x3])
        x = self.Dec_GT_128(x)

        ### Green Block x1 (output filters=64)
        x = self.Dec_GT_ReduceDepth_64(x)
        x = self.Dec_GT_UpSample_64(x)
        x = self.Input_Dec_GT_64([x, x2])
        x = self.Dec_GT_64(x)

        ### Green Block x1 (output filters=32)
        x = self.Dec_GT_ReduceDepth_32(x)
        x = self.Dec_GT_UpSample_32(x)
        x = self.Input_Dec_GT_32([x, x1])
        x = self.Dec_GT_32(x)

        ### Blue Block x1 (output filters=32)
        x = self.Input_Dec_GT_Output(x)

        ### Output Block
        out_GT = self.Dec_GT_Output(x)

        ## VAE (Variational Auto Encoder) Part
        # -------------------------------------------------------------------------

        ### VD Block (Reducing dimensionality of the data)
        x = self.Dec_VAE_VD_GN(x4)
        x = self.Dec_VAE_VD_relu(x)
        x = self.Dec_VAE_VD_Conv3D(x)

        # Not mentioned in the paper, but the author used a Flattening layer here.
        x = self.Dec_VAE_VD_Flatten(x)
        x = self.Dec_VAE_VD_Dense(x)

        ### VDraw Block (Sampling)
        z_mean = self.Dec_VAE_VDraw_Mean(x)
        z_var = self.Dec_VAE_VDraw_Var(x)
        x = self.Dec_VAE_VDraw_Sampling([z_mean, z_var])

        ### VU Block (Upsizing back to a depth of 256)
        x = self.VU_Dense1(x)
        x = self.VU_relu(x)
        x = self.VU_reshape(x)
        x = self.Dec_VAE_ReduceDepth_256(x)
        x = self.Dec_VAE_UpSample_256(x)

        ### Green Block x1 (output filters=128)
        x = self.Dec_VAE_ReduceDepth_128(x)
        x = self.Dec_VAE_UpSample_128(x)
        x = self.Dec_VAE_128(x)

        ### Green Block x1 (output filters=64)
        x = self.Dec_VAE_ReduceDepth_64(x)
        x = self.Dec_VAE_UpSample_64(x)
        x = self.Dec_VAE_64(x)

        ### Green Block x1 (output filters=32)
        x = self.Dec_VAE_ReduceDepth_32(x)
        x = self.Dec_VAE_UpSample_32(x)
        x = self.Dec_VAE_32(x)

        ### Blue Block x1 (output filters=32)
        x = self.Input_Dec_VAE_Output(x)

        ### Output Block
        out_VAE = self.Dec_VAE_Output(x) 
        
        self.LossVAE([Z, out_VAE, z_mean, z_var])
        
        return out_GT
    
    
###
def set_weight_decay(model, alpha):
    """
    This function can be applied to Conv2D, Dense or DepthwiseConv2D
    """
    for layer in model.layers:
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.depthwise_kernel))
        elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.bias))
            
            
### LogLogger
class LogLogger(keras.callbacks.Callback):
    """
    Logging the train job to a logger object.
    
    train_data, batch_size argument will be used in evaluate trainset on each end of epochs. If you don't want to evaluate it, ignore the arguments.
    
    train_data : a list of [x_train, y_train]. e.g. [x_train, [y_train, x_train]]
    """
    
    def __init__(self, logger_object, evaluate_trainset_on_epoch_end = False, train_data = None, batch_size = 1, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger_object
        self.evaluate_trainset_on_epoch_end = evaluate_trainset_on_epoch_end
        
        if evaluate_trainset_on_epoch_end:
            if train_data is None:
                raise ValueError("Train data should be given if you want to evaluate train set on end of epoch.")
            else:
                self.train_data = train_data
                self.batch_size = batch_size
    
    def on_train_batch_end(self, batch, logs=None):
        self.logger.info("For batch {}: loss = {:7.4f}, metric = {:7.4f}.".format(batch, logs["loss"], logs["dice_coefficient"]))
    
    def on_epoch_end(self, epoch, logs):
        if self.evaluate_trainset_on_epoch_end:
            x_tr, y_tr = self.train_data
            train_eval_results = self.model.evaluate(x_tr, y_tr, verbose = 0, batch_size = self.batch_size)
        
            for i, n in enumerate(self.model.metrics_names):
                logs[n] = train_eval_results[i]
            
        self.logger.debug(f'''The result of {epoch+1} epoch: {', '.join((f"{k} = {v}" for k,v in logs.items()))}''')