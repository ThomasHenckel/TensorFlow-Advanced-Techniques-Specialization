# Custom Models, Layers, and Loss Functions with TensorFlow
 
## Functional API
[Keras Functional API Documentation](https://keras.io/guides/functional_api/) 
In Keras you can use the sequential or the Functional syntax to build models, while the sequential only allows you to build linear models, the Functional syntax allow for much more advanced models with multiple inputs and outputs, and also looping architectures.
 
The code below show an example of a model with two outputs, one for the wine_quality and one for the wine_type.
Full example are found here: [C1W1_Assignment.ipynb](./C1W1_Assignment.ipynb)
```python
def model()
   inputs = tf.keras.layers.Input(shape=(11,))
   # connect a Dense layer with 128 neurons and a relu activation
   x = tf.keras.layers.Dense(128, activation='relu', name="first_base_dense")(inputs)
  
   # connect another Dense layer with 128 neurons and a relu activation
   x = tf.keras.layers.Dense(128, activation='relu', name="second_base_dense")(x)
 
   # connect the output Dense layer for regression
   wine_quality = Dense(units='1', name='wine_quality')(x)
 
   # connect the output Dense layer for classification. this will use a sigmoid activation.
   wine_type = Dense(units='1', activation=tf.keras.activations.sigmoid, name='wine_type')(x)
 
   # define the model using the input and output layers
   model = Model(inputs=inputs, outputs=[wine_quality,wine_type])
```
 
### Siamese network
Another great example of using the functional api is the siamese network that constructs a model that takes images as input and gives a score telling how similar the images are. special for the network is that both sides of the network are in fact the same network with the same weight. 
The Example also uses a lamda layers and custom loss function, something that is explained in more detail later.
[Siamese network example](./C1_W1_Lab_3_siamese-network.ipynb)
 
## Custom loss function
Keras have several [build loss functions](https://keras.io/api/losses/) but sometimes they might not fit the learning problem you want to solve, like in the Siamese network.
for these scenarios you can build and use your own custom loss function.
Custom loss functions can be build in a function or a class.
when building it as a class the input parameters are always y_true and y_pred and you cant pass any other parameters. 
I prefer to alway build loss functions as classes as it allows parsing other variables to the loss function, making it configurable to different scenarios, or to allow tuning of the loss functions hyperparameters.
 
```python
def model()
# wrapper function that accepts the hyperparameter
def my_huber_loss(threshold):
    # function that accepts the ground truth and predictions
   def my_huber_loss(y_true, y_pred):
       error = y_true - y_pred
       is_small_error = tf.abs(error) <= threshold
       small_error_loss = tf.square(error) / 2
       big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
      
       return tf.where(is_small_error, small_error_loss, big_error_loss)
 
   # return the inner function tuned by the hyperparameter
   return my_huber_loss
 
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=my_huber_loss(threshold=1.2))
```
Remember to use the [tensorflow math operations](https://www.tensorflow.org/api_docs/python/tf/math) inside the loss function.
It is possible to use normal math operations syntax on operations like multiplication and addition as they are overloaded. So when using these on tensorflow tensors the tf operations are used.
 
[C1_W2_Lab_2_huber-object-loss.ipynb](./C1_W2_Lab_2_huber-object-loss.ipynb) Full example of the my_huber_loss
[C1W2_Assignment.ipynb](./C1W2_Assignment.ipynb) implements [RMSE](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError) as a custom loss function
 
## Lambda Layers & Custom Layers
 
lambda layers can be used when the [build-in keras layers](https://keras.io/api/layers/) don't have the functionality you need, and the functionality you want in your layer is fairly simple. For more complex functionality you want to creat a custom layer, which will be described a bit later.
### Lambda Layers
The following is a simple lamda layer, that is taking the absolute values just the same way as [relu activation](https://keras.io/api/layers/activations/#relu-function).
```python
tf.keras.layers.Lambda(lambda x: tf.abs(x))
```
Replacing relu activation with a lamda layer is of course not so useful or exiting, but it shows the use of lamda layers. 
[C1_W3_Lab_1_lambda-layer.ipynb](./C1_W3_Lab_1_lambda-layer.ipynb) example of lamda layer in use
 
### Custom Layers
Custom layers can extend the keras base layer, unlike lamda layers custom layers have a state, and and can be trainable.
Below is an example of a custom dense layer, that has the states w(weight) and b(bias)
When extending keras base layer you need to implement the init, build and call function.
```python
class SimpleDense(Layer):
 
   # add an activation parameter
   def __init__(self, units=32, activation=None):
       super(SimpleDense, self).__init__()
       self.units = units
      
       # define the activation to get from the built-in activation layers in Keras
       self.activation = tf.keras.activations.get(activation)
 
 
   def build(self, input_shape):
       w_init = tf.random_normal_initializer()
       self.w = tf.Variable(name="kernel",
           initial_value=w_init(shape=(input_shape[-1], self.units),
                                dtype='float32'),
           trainable=True)
       b_init = tf.zeros_initializer()
       self.b = tf.Variable(name="bias",
           initial_value=b_init(shape=(self.units,), dtype='float32'),
           trainable=True)
       super().build(input_shape)
 
 
   def call(self, inputs):
      
       # pass the computation to the activation layer
       return self.activation(tf.matmul(inputs, self.w) + self.b)
```
[C1_W3_Lab_3_custom-layer-activation.ipynb](./C1_W3_Lab_3_custom-layer-activation.ipynb) example of custom layers in use
[C1W3_Assignment.ipynb](./C1W3_Assignment.ipynb) example of a custom quadratic layer applied to fashion MNIST
 
## Custom Models
An alternative to creating your model in a function, it is also possible to extend the [Model class](https://keras.io/api/models/model/).
Extending the model class should give you additional control and the possibility to change the way data flows through the network.
[XLNet](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/xlnet.py) is an example that uses model subclassing. but you can easely write modular and configurable network without it.
 
An example of [ResNet](https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py) from tensorflow repo that does not subclass tf.keras.Model
 
And an example of [ResNet](./C1_W4_Lab_2_resnet-example.ipynb) from the course using subclassing, this example is less code mainly because it does not implement the full ResNet model
 
When extending the model class the init and call function should be implemented
```python
class MyModel(tf.keras.Model):
 
 def __init__(self):
   super(MyModel, self).__init__()
   self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
   self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
 
 def call(self, inputs):
   x = self.dense1(inputs)
   return self.dense2(x)
```
Finally a [VGG network](./C1W4_Assignment.ipynb) example using subclassing
 
## Built-in and Custom Callbacks
 
### Built-in callbacks
There are a number of [built-in](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) callbacks. A callback is specified in the training loop, and is called before and after each epoch.
My favorites are EarlyStopping and TensorBoard. The first lets you stop the training when a number of epochs have not shown any improvements, and it can save the best model. Tensorboard saves a log file that can be opened by tensorboard, and is a great way to investigate how the training progressed, and also a good way to document each of your experiments
 
```python
model = build_model(dense_units=256)
model.compile(
   optimizer='sgd',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy'])
 
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
 
early_stopping_callback = EarlyStopping(patience=3,
                                       min_delta=0.05,
                                       baseline=0.8,
                                       mode='min',
                                       monitor='val_loss',
                                       restore_best_weights=True,
                                       verbose=1)
 
model.fit(train_batches,
         epochs=50,
         validation_data=validation_batches,
         verbose=2,
         callbacks=[early_stopping_callback, tensorboard_callback]
         )
```
[C1_W5_Lab_1_exploring-callbacks.ipynb](./C1_W5_Lab_1_exploring-callbacks.ipynb) shows how to use many of the build in callbacks
 
### Custom callbacks
Custom callbacks are a great way to implement logic that has to run before or after an epoch, this might be to log something, or to determine if training has to stop, or maybe to adjust hyperparameters while training.
 
The custom callback below shows how to stop training if the loss and validation loss is starting to diverge
```python
class DetectOverfittingCallback(tf.keras.callbacks.Callback):
   def __init__(self, threshold=0.7):
       super(DetectOverfittingCallback, self).__init__()
       self.threshold = threshold
 
   def on_epoch_end(self, epoch, logs=None):
       ratio = logs["val_loss"] / logs["loss"]
       print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, ratio))
 
       if ratio > self.threshold:
           print("Stopping training...")
           self.model.stop_training = True
 
model = get_model()
_ = model.fit(x_train, y_train,
             validation_data=(x_test, y_test),
             batch_size=64,
             epochs=3,
             verbose=0,
             callbacks=[DetectOverfittingCallback()])
```
 
More example are found [here](./C1_W5_Lab_2_custom-callbacks.ipynb)

