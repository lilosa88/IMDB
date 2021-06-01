# IMDB
# Objective

- This project was carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the third course in this specialization. 
  
- From TensorFlow Data Services (TFTS) library we used the IMDB reviews dataset. This dataset was authored by [Andrew Mass et al at Stanford](http://ai.stanford.edu/~amaas/data/sentiment/). 


# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, re, nltk, json, keras, tensorflow

# Data description

- This is a dataset for binary sentiment classification. It contanins 50,000 movie reviews which are classified as positive of negative. From which a set of 25,000 highly polar movie reviews are for training, and 25,000 for testing. Raw text and already processed bag of words formats are provided. 

# Preprocessing

- Test and train split: This dataset has about 50,000 records. So, we train on 25,000 and validate on the rest. 

-For both train and test data we apply the following steps:

  - We carried out the pre-processing with the following hyperparameters:
    - vocab_size = 10000 
    - embedding_dim = 16 
    - max_length = 120 
    - trunc_type='post'
    - oov_tok = "<OOV>"
  
  - We convert the two train and test sets into arrays
  
  - First, we apply tokenizer which is an encoder ofer by Tensorflow and keras. This works by generating a dictionary of word encodings and creating vectors out    of the sentences. The hyper-parameter vocab_size is given as the number of words. So by setting this hyperparameter, what the tokenizer will do is take the      top number of words given in vocab_size and just encode those. On the other hand, in many cases, it's a good idea to instead of just ignoring unseen words, to    put a special value when an unseen word is encountered. You can do this with a property on the tokenizer. This property is oov token and is set in the            tokenizer constructor. I've specified that I want the token OOV for outer vocabulary to be used for words that aren't in the word index.
  
  - Second, we apply the fit_on_texts method of the tokenizer that actually encodes the data following the hyper-parameter given previosuly. 
  
  - Third, we apply the word_index method. The tokenizer provides a word index property which returns a dictionary containing key value pairs, where the key is     the word, and the value is the token for that word. An important thing to highlight is that tokenizer method strips punctuation out and convert all in           lowercase.
  
  - Fourth, we turn the sentences into lists of values based on these tokens.To do so, we apply the method texts_to_sequences.
  
  - Fifth, we manipulate these lists to make every sentence the same length, otherwise, it may be hard to train a neural network with them. To do so, we apply      the method pad_sequences that use padding. First, in order to use the padding functions you'll have to import pad sequences from                                  tensorflow.carastoppreprocessing.sequence. Then once the tokenizer has created the sequences, these sequences can be passed to pad sequences in order to have    them padded. The list of sentences then is padded out into a matrix where each row in the matrix has the same length. This is achieved by putting the            appropriate number of zeros before the sentence. If we prefer the zeros being on the right side then we set the parameter padding equals post. Normally, the      matrix width has the same size as the longest sentence. However, this can be override that with the maxlen parameter. If I have sentences longer than the        maxlength, then I'll lose information. If padding is pre (what it is by default), I will lose information from the beginning of the sentence. In order to        override this so that we will lose it from the end instead, we can do so with the truncating parameter.
  
 
# Neural Network models
  
  - First model:
  
    - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
    - One Embedding layer:  This is the process that help us to go from just a string of numbers representing words to actually get text sentiment. This process       is called embedding, with the idea being that words and associated words are clustered as vectors in a multi-dimensional space. Basically pick a vector in       a higher-dimensional space, and words that are found together are given similar vectors. Then over time, words can begin to cluster together. The meaning         of the words can come from the labeling of the dataset. As a result we have the vectors for each word with their associated sentiment.The results of the         embedding will be a 2D array with the length of the sentence and the embedding dimension for example 16 as its size. So we need to flatten it out in much         the same way as we needed to flatten out our images. 
    - One Flatten layer
    - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer       consisted in 6 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=10

    - We obtained Accuracy 1.0 for the train data and Accuracy 0.8254 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-06-01%2007-00-52.png" width="320" height="460">
      </p>  
  
  - Second model:
  
    - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
    - One Embedding layer:  This is the process that help us to go from just a string of numbers representing words to actually get text sentiment. This process       is called embedding, with the idea being that words and associated words are clustered as vectors in a multi-dimensional space. Basically pick a vector in       a higher-dimensional space, and words that are found together are given similar vectors. Then over time, words can begin to cluster together. The meaning         of the words can come from the labeling of the dataset. As a result we have the vectors for each word with their associated sentiment.The results of the         embedding will be a 2D array with the length of the sentence and the embedding dimension for example 16 as its size. So we need to flatten it out in much         the same way as we needed to flatten out our images. 
    - One  Global Average Pooling 1D layer: Often in natural language processing, a different layer type than a flatten is used, and this is a global average           pooling 1D. The reason for this is the size of the output vector being fed into the dense layers. This layer averages across the vector to flatten it out.       As a result the process should be a little faster.
    - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer       consisted in 6 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=10

    - We obtained Accuracy 0.9509 for the train data and Accuracy 0.8316 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-06-01%2007-01-15.png" width="320" height="460">
      </p>
  
    - Third model:
  
    - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
      - One Embedding layer
      - One Bidirectional layer with GRU(32) 
      - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer       consisted in 6 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=50

    - We obtained Accuracy 0.9832 for the train data and Accuracy 0.8195 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-06-01%2007-01-39.png" width="320" height="460">
      </p>  
  
  - Fourth model:
  
    - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
      - One Embedding layer
      - One Bidirectional layer with LSTM(32) 
      - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense            layer consisted in 6 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=50

    - We obtained Accuracy 1.0 for the train data and Accuracy 0.8135 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-06-01%2007-01-58.png" width="320" height="460">
      </p>  
  
  - Fifth model:
  
    - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
      - One Embedding layer
      - One Conv1D layer: You specify the number of convolutions that you want to learn, their size, and their activation function. Now words will be grouped             into the size of the filter in this case 5. And convolutions will learned that can map the word classification to the desired output. As the size of the         input is 120 words, and a filter that is 5 words long will shave off 2 words from the front and back, leaving us with 116. The 128 filters that we               specified will show up here as part of the convolutional layer.
      - on GlobalAveragePooling1D
      - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense             layer consisted in 6 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=50

    - We obtained Accuracy 1.0 for the train data and Accuracy 0.7966 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-06-01%2007-02-18.png" width="320" height="460">
      </p>  
  
  
    - Sixth model:
      
      - In the previous models we use TFDS to load the reviews from the IMDb dataset and perform classification on them. However, often with prepackaged datasets         like these, some data scientists have done the work. For this model we use the version of the IMDb dataset that has been pre-tokenized. This tokenization         is done on sub words.
  
      - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
          - One Embedding layer 
          - One  Global Average Pooling 1D layer
          - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense             layer consisted in 6 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=10

    - We obtained Accuracy 0.9411 for the train data and Accuracy 0.8681 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-05-31%2017-23-43.png" width="320" height="460">
      </p>  

    - Seventh model:
      
      - In the previous models we use TFDS to load the reviews from the IMDb dataset and perform classification on them. However, often with prepackaged datasets         like these, some data scientists have done the work. For this model we use the version of the IMDb dataset that has been pre-tokenized. This tokenization         is done on sub words.
  
      - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
          - One Embedding layer 
          - One Bidirectional layer: The parameter passed in is the number of outputs that I desire from that layer, in this case it's 64. With                               tf.keras.layers.Bidirectional, it will make my cell state go in both directions. LSTM
          - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense             layer consisted in 64 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

    - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

    - The number of epochs=10

    - We obtained Accuracy 0.8928 for the train data and Accuracy 0.7867 for the validation data.
  
      <p align="center">
        <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-05-31%2017-23-43.png" width="320" height="460">
      </p>  
  
   - Eigth model:
      
      - In the previous models we use TFDS to load the reviews from the IMDb dataset and perform classification on them. However, often with prepackaged datasets         like these, some data scientists have done the work. For this model we use the version of the IMDb dataset that has been pre-tokenized. This tokenization         is done on sub words.
  
      - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the         following:
          - One Embedding layer 
          - Two Bidirectional layer: The parameter passed in is the number of outputs that I desire from that layer, in this case it's LSTM(64) for the first one             and LSTM(32) for the second one. But when you feed an LSTM into another one, you do have to put the return_sequences equal true parameter into the               first one. 
          - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense             layer consisted in 64 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

      - We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

      - The number of epochs=10

      - We obtained Accuracy 0.9774 for the train data and Accuracy 0.8595 for the validation data.
  
        <p align="center">
         <img src="https://github.com/lilosa88/IMDB/blob/main/Images/Screenshot%20from%202021-05-31%2017-23-43.png" width="320" height="460">
        </p> 
  
  



