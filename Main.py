import tensorflow
from PIL.ImImagePlugin import number
from keras.src.saving import load_model
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


# Loading dataset
dataset = keras.datasets.mnist
((number_train, ident_train),(number_test, ident_test)) = dataset.load_data()
'''
# Exploring Data / shape = 28,28. length = 60.000. total = 10
length = len(number_test)
shape = number_train.shape
min = ident_train.min()
max = ident_train.max()
total_ident = 10
print(length, shape, min, max)
'''
# Visualizing data
classes = ['0','1','2','3','4','5','6','7','8','9']
'''
plt.imshow(number_train[9])
plt.title(ident_train[9])
plt.show()

for image in range(10):
    plt.subplot(2, 5, image+1)
    plt.imshow(number_train[image])
    plt.title(classes[ident_train[image]])
plt.show()

# Normalizing Color
plt.imshow(number_train[0])
plt.colorbar()
plt.show()
'''
number_train = number_train / float(255)

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(256, activation = tensorflow.nn.relu),
    keras.layers.Dense(256, activation = tensorflow.nn.softmax)
])

# Compiler
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics  = ['accuracy']
)

graph = model.fit(number_train, ident_train, epochs=5, validation_split = 0.2)

# Saving model
model.save('model.h5')
model_save = load_model('model.h5')
'''
# Accuracy graph
plt.plot(graph.history['accuracy'])
plt.plot(graph.history['val_accuracy'])
plt.xlabel('Acurácia')
plt.ylabel('Épocas')
plt.legend(['Treino', 'Avaliação'])
plt.show()

# Loss graph
plt.plot(graph.history['loss'])
plt.plot(graph.history['val_loss'])
plt.xlabel('Perdas')
plt.ylabel('Épocas')
plt.legend(['Treino', 'Avaliação'])
plt.show()

# Seeing test results
for i in range(10):
    test = model.predict(number_test)
    print('Resultado do teste: ', np.argmax(test[i]))
    print('Resultado do teste: ', ident_test[i])
'''

# evaluating model
loss_test, acc_test = model.evaluate(number_test, ident_test)
print('Losses during the test: ', loss_test)
print('Accuracy: ', acc_test)

