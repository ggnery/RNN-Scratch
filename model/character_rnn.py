from .rnn import RNN
import numpy as np

class CharacterRNN:
    def __init__(self, hidden_dim=32, learning_rate=0.1):
        """
        Character-level RNN for learning text patterns
        
        Args:
            hidden_dim: Size of hidden layer
            learning_rate: Learning rate for optimization
        """
        # Define character set
        self.chars = 'abcdefghijklmnopqrstuvwxyz '
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Initialize RNN
        self.rnn = RNN(
            input_size=self.vocab_size,
            hidden_size=hidden_dim,
            output_size=self.vocab_size,
            learning_rate=learning_rate
        )
        
    def char_to_onehot(self, char):
        """Convert character to one-hot vector"""
        vec = np.zeros((self.vocab_size, 1))
        if char in self.char_to_idx:
            vec[self.char_to_idx[char]] = 1
        return vec
    
    def text_to_sequence(self, text):
        """Convert text to sequence of one-hot vectors"""
        return [self.char_to_onehot(ch) for ch in text.lower()]
    
    def train_on_text(self, text, epochs=1000, print_every=100):
        """
        Train RNN on a text pattern
        
        Args:
            text: Training text
            epochs: Number of training epochs
            print_every: Print loss every N epochs
        """
        # Prepare training data
        text = text.lower()
        losses = []
        
        print(f"Training on pattern: '{text}'")
        print("="*50)
        
        for epoch in range(epochs):
            # Create input and target sequences
            # Input: all characters except last
            # Target: all characters except first (shifted by 1)
            x_sequence = self.text_to_sequence(text[:-1])
            y_sequence = self.text_to_sequence(text[1:])
            
            # Train step
            loss = self.rnn.train_step(x_sequence, y_sequence)
            losses.append(loss)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d}, Loss: {loss:.4f}")
                
                # Generate sample
                generated = self.generate_text(text[0], len(text))
                print(f"Generated: '{generated}'")
                print("-"*50)
        
        return losses
    
    def predict_next_char(self, current_char, hidden_state=None):
        """
        Predict next character given current character
        
        Args:
            current_char: Current character
            hidden_state: Previous hidden state (optional)
        
        Returns:
            next_char: Predicted next character
            new_hidden_state: Updated hidden state
        """
        # Convert to one-hot
        x = [self.char_to_onehot(current_char.lower())]
        
        # Forward pass
        y_hat_sequence, cache = self.rnn.forward(x, hidden_state)
        
        # Get prediction
        probs = y_hat_sequence[0].flatten()
        
        # Sample from distribution
        next_idx = np.random.choice(self.vocab_size, p=probs)
        next_char = self.idx_to_char[next_idx]
        
        # Get final hidden state
        new_hidden_state = cache['a_sequence'][-1]
        
        return next_char, new_hidden_state
    
    def generate_text(self, seed_char, length):
        """
        Generate text starting from a seed character
        
        Args:
            seed_char: Starting character
            length: Length of text to generate
        
        Returns:
            generated_text: Generated string
        """
        generated = seed_char
        current_char = seed_char
        hidden_state = None
        
        for _ in range(length - 1):
            next_char, hidden_state = self.predict_next_char(current_char, hidden_state)
            generated += next_char
            current_char = next_char
            
        return generated

