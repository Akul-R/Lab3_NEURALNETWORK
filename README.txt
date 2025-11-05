Functions in the ANN Class
  ANN(sizes, func, bits) - Constructor method, takes 3 arguments:
    - sizes:
      - a vector of integers to show the size of each layer.
      - for example, sizes = { 4, 8, 16 } tells the program that there are 4 neurons in the input layer, 8 neurons in the hidden layer and 16 neurons in the output layer
      - the first number (sizes[0]) must be the same as bits and the last number must equal 2^n where n is bits (if bits is 4, then layer[0] = 4 and layer.back() = 2^4 = 16)
      - can be as big as you want (assuming your numbers are correct) but bear in mind, more layers mean longer training time and a chance your model will overfit the data 
        and not learn properly (the simpler the network, the better).
    
    - func:
      - a string to choose activation function.
      - current support for sigmoid (pass "sigmoid" as argument) and ReLU (pass "relu" as argument).
      - invalid arguments will throw an error
      - for binary classification, ReLU is often better as it trains model faster (~5k epochs to train 4 bit model using ReLU whereas sigmoid took around 15k-20k to train same model).
                                                                                                                
    - bits:
      - an integer to select what bit number will be inputted.
      - current maximum is 64 bits
      - ensure your layers are scaled properly and you select an appropriate training rate and and max epochs
      - selecting larger values will lead to much longer training times (both due to the increased number of epochs required and the more complex matrix multiplication).
                                                                                                                 
  training_start(max_epoch, out_epoch, training_rate) - method to start training model, takes 3 arguments:
    - max_epochs:
      - an integer to set the maximum number of training cycles (epochs)

    - out_epoch:
      - an integer to set how often the program will print the results of training (setting it to 500 will make it print out every 500 epochs, setting to 1 will print out on every epoch)

    - training_rate:                                                                                                             
      - a double to select how quick the model learns. you should aim to keep this number fairly low (0.01 worked to train 4 bit model, 0.001 worked to train 8 bit model. scale according to number of bits)

  testing(print_all_confidence) - method to test model using user inputs (train model first before running), takes 1 argument:
    - print_all_confidence:
      - a boolean to select whether the program prints the confidence scores for each number. setting to true makes it print everything out (not recommended if you are using a large number of bits).
                                                                                                                                               
  save(weight_file_name, bias_file_name) - method to save the weights and biases of each layer after training model. takes in 2 arguments for file name but will create weight.wts and bias.bss if no names given.
  
  load(weight_file_name, bias_file_name) - method to load the weights and biases from a .wts or .bss file. takes in 2 arguments for the file names of each

  accuracy(print_all_wrong) - method to measure how accurate the model is. takes in 1 boolean argurment, if true, will print out all numbers the model guessed incorrectly (not recommended for large bit sizes)
