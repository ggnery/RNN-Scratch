from model import CharacterRNN
import matplotlib.pyplot as plt

def main():
    # Create character RNN
    char_rnn = CharacterRNN(hidden_dim=16, learning_rate=0.1)
    
    # Example 1: Learn a simple repeating pattern
    print("\n=== EXAMPLE 1: Learning 'hello world' ===")
    pattern1 = "hello world"
    losses1 = char_rnn.train_on_text(pattern1 * 15, epochs=1000, print_every=100)
    
    # Example 2: Learn a different pattern
    print("\n\n=== EXAMPLE 2: Learning 'abcdefg' ===")
    char_rnn2 = CharacterRNN(hidden_dim=16, learning_rate=0.1)
    pattern2 = "abcdefg"
    losses2 = char_rnn2.train_on_text(pattern2 * 15, epochs=500, print_every=100)
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses1)
    plt.title("'hello world' Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(losses2)
    plt.title("'abcdefg' Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    
    plt.tight_layout()
    plt.show()
    
    # Test generation with different starting characters
    print("\n\n=== GENERATION TESTS ===")
    print("\nGenerating from 'h' with model trained on 'hello world':")
    for i in range(5):
        generated = char_rnn.generate_text('h', 15)
        print(f"  {i+1}: '{generated}'")
    
    print("\nGenerating from 'a' with model trained on 'abcdefg':")
    for i in range(5):
        generated = char_rnn2.generate_text('a', 10)
        print(f"  {i+1}: '{generated}'")
    

if __name__ == "__main__":
    main()