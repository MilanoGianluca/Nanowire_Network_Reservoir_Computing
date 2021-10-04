****** MNIST *******

The whole process to simulate Reservoir Computing algorithm on the MNIST dataset is by running sequentially:

-RC_Mnist_28x28_network_stimulation.py
-RC_Mnist_28x28_hist_generator.py
-RC_Mnist_28x28_readout.py

Since the first two steps are time consuming, we provide the outcome data (raw_data) of the first two steps to directly perform the last step. Note that hist_train has been splitted into six parts for file dimensions issues.

In 'RC_Mnist_28x28_readout.py' there is the possibility to perform a new training or import our trained data (training_data_backup).
The new training option may produce slightly different output with respect to the paper results due to the intrinsic randomicity of the process.


****** MACKEY GLASS ******

Run 'mackey_glass.py' to obtain the results of mackey glass equation forecasting task.
