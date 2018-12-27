<h1>Emojifier.py</h1>
<h3>Summary</h3>:
A sequence model that takes 5 character text input and predicts an emoji.

<h3>Model</h3>:

| Layer (type)         | Output Shape            | Param #  |
| ------------- |:-------------:| -----:|
| input_2 (InputLayer)      | right-aligned | $1600 |
| embedding_2 (Embedding)      | centered      |   $12 |
| lstm_3 (LSTM)  | are neat      |    $1 |
| dropout_3 (Dropout)       | right-aligned | $1600 |
| lstm_4 (LSTM)     | centered      |   $12 |
| dropout_4 (Dropout) | are neat      |    $1 |
| dense_2 (Dense)        | right-aligned | $1600 |
| activation_2 (Activation)       | centered      |   $12 |


<h2>Some predictions by model</h2>
Input               Output

want to eat food    🍴

She loves me        ❤️

I got new job       😄

I lost my phone     😞
