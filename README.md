# CMPT 414 FinalAssignment: Implement a CNN using Numpy by Aleksansor Minderov and Derek Roy

This model was trained and built on the MNIST Dataset. MNIST Data can be found at the link here: https://drive.google.com/file/d/1SrpQX4a302pzH50R_NF1mmcUSQeoyHHM/view?usp=sharing

This code was written in Python 3 and should support 3.6 or 3.7



The following libraries are required for this code: 
- numpy
- sklearn
- scipy 
- Pillow
- kivy

Running the command: `pip3 install numpy sklearn scipy Pillow kivy` or `pip install numpy sklearn scipy Pillow kivy` (depending on how your system is configured) will install the libraries.



Library Versions used on testing:
- scikit-learn==0.22.2.post1
- scipy==1.4.1
- numpy==1.18.1
- Pillow==6.1.0
- Kivy==1.11.1



This CNN has Model load, Model save, Train, Test, and Predict functionality, however for the sake of our executable and python script we will only support prediction with our pretrained model. 

To use the project either run the executable or the "(prediction python script)": 

## Execution
### Use CNN directly
First, get training and testing images:
```python
from data import get_data

X_train, X_test, Y_train, Y_test = get_data()
```

Then, initialize CNN, passing one of the images such that the CNN knows what dimensions to expect
```python
from cnn import CNN

nn = CNN(X_train[0])
```

Train the CNN, configuring number of epochs and number of images to train on:
```python
nn.train(X_train, Y_train, epochs=10, images_limit=100)
# Inside .train(), it uses .feed_forward() and .back_prop() methods
```

Get the prediction on a single image like so:
```python
nn.feed_forward(X_test[0])
```

Save the model:
```python
nn.save_model('file_name')
```

Load a model:
```python
nn.load_model('file_name')
```
### Drawing by Hand
To test the CNN on your own hand-drawn images, launch this script:
```bash
python3 draw.py
```

It will display a window where you can draw using your mouse, and will display the result in the console.

Video demo: https://youtu.be/olITdJCMPgc
