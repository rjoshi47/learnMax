<h2>Emojifier.py</h2>
<h3>Summary</h3>
A sequence model that takes 5 character text input and predicts an emoji.

<h3>Model</h3>

| Layer (type)         | Output Shape            | Param #  |
| ------------- |:-------------:| -----:|
| input_2 (InputLayer)      | (None, 10)   | 0 |
| embedding_2 (Embedding)      | (None, 10, 50)         |   20000050 |
| lstm_3 (LSTM)  | (None, 10, 128)       |    91648 |
| dropout_3 (Dropout)       | (None, 10, 128)    | 0 |
| lstm_4 (LSTM)     | (None, 128)        |   131584 |
| dropout_4 (Dropout) | (None, 128)        |    0 |
| dense_2 (Dense)        | (None, 5)  | 645 |
| activation_2 (Activation)       | (None, 5)        |   0 |


<h3>Some predictions by model</h3>

| Input        | Prediction           | 
| ------------- |:-------------:|
| want to eat food      | 🍴 | 
| She loves me       | ❤️      |   
| I got new job | 😄      |  
| I lost my phone | 😞|  
