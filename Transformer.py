import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from preprocessing import clean_text

# Custom layer for multi-head self-attention
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        # Dimensionality of the embedding and number of attention heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Dimension of each attention head
        self.head_dim = embed_dim // num_heads

        # Ensure embedding dimension is divisible by the number of heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        # Linear transformations for query, key, value, and output
        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)

    # Helper function to split heads of the input tensor
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # Forward pass of the multi-head self-attention layer
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Linear transformations for query, key, and value
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Splitting into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention mechanism
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        # Rearrange and concatenate attention heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))

        # Linear transformation for the output
        output = self.dense(concat_attention)

        return output, attention_weights

    # Scaled dot-product attention mechanism
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask to prevent attending to padded positions
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # Softmax activation to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # Weighted sum using attention weights
        output = tf.matmul(attention_weights, v)

        return output, attention_weights


# Custom layer for a Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention layer
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        # Feedforward neural network
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        # Layer normalization for both attention and feedforward output
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout layers for regularization
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # Forward pass of the Transformer block
    def call(self, inputs, training=True):
        # Multi-head self-attention layer
        attn_output, _ = self.att(inputs, inputs, inputs, mask=None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feedforward neural network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual connection and layer normalization
        return self.layernorm2(out1 + ffn_output)


# Custom layer for token and position embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # Embedding layer for tokens
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # Embedding layer for positions
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    # Forward pass of the token and position embedding layer
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        # Create positions tensor
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        # Token embedding and addition with position embeddings
        x = self.token_emb(x)
        return x + positions


# Function to create the entire transformer model
def create_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, num_classes):
    inputs = layers.Input(shape=(maxlen,))
    # Token and position embedding layer
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Stack multiple Transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Global average pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling1D()(x)
    # Dropout layer for regularization
    x = layers.Dropout(0.1)(x)
    # Dense layer with ReLU activation for additional non-linearity
    x = layers.Dense(20, activation="relu")(x)
    # Additional dropout layer
    x = layers.Dropout(0.1)(x)
    # Output layer with softmax activation for classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



df = pd.read_excel("train.xlsx")

df['review_description'] = df['review_description'].apply(lambda x: clean_text(x))

# Tokenize and pad text data
max_len = 100 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['review_description'])
sequences = tokenizer.texts_to_sequences(df['review_description'])
X = pad_sequences(sequences, maxlen=max_len)

y = df['rating'].values + 1  # Add 1 to each label to make them 0, 1, 2

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
embed_dim = 64  
num_heads = 4 
ff_dim = 64  
num_blocks = 4  
num_classes = 3

# Build and compile the model
model = create_transformer_model(max_len, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, num_classes)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

batch_size = 64 
epochs = 10

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks= [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')])

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
model.save("transformer_model.keras")

test_df = pd.read_csv('test _no_label.csv')

# Tokenize and pad the text data in the new DataFrame
new_sequences = tokenizer.texts_to_sequences(test_df['review_description'])
new_X = pad_sequences(new_sequences, maxlen=max_len)

# Make predictions using the trained model
predictions = model.predict(new_X)

# Convert the predicted probabilities to class labels
predicted_labels = predictions.argmax(axis=1) - 1  # Subtract 1 to convert back to -1, 0, 1

# Add the predicted labels to the new DataFrame
test_df['rating'] = predicted_labels
test_df = test_df[['ID', 'rating']]

test_df.to_csv('Results.csv', index = False)
