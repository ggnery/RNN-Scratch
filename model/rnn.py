from typing import List, Tuple, Dict
import numpy as np

class RNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
        """
        Initialize the RNN model.

        Args:
            input_size: The size of the input features.
            hidden_size: The size of the hidden state.
            output_size: The size of the output.
            learning_rate: The learning rate for the model.
        """
        
        super().__init__()
        self.input_size = input_size # x input size (n)
        self.hidden_size = hidden_size # h hidden size (m)
        self.output_size = output_size # y output size (c)
        self.lr =learning_rate
        
        # weight matrices
        self.W_xh = np.random.randn(self.hidden_size, self.input_size) # Matrix that is multiplied by x to get h
        self.W_ah = np.random.randn(self.hidden_size, self.hidden_size) # Matrix that is multiplied by a to get h
        self.W_ao = np.random.randn(self.output_size, self.hidden_size) # Matrix that is multiplied by a to get o
        
        # bias 
        self.b_h = np.random.random((self.hidden_size, 1)) # Hidden bias 
        self.b_o = np.random.random((self.output_size, 1)) # Output bias
        
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
    def forward(self, x_sequence: List[np.ndarray], a_0: np.ndarray = None) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        """
        Forward propagation through time
        
        Args:
            x_sequence: List of input vectors, each of shape (input_size, 1)
            a_0: Initial hidden state, shape (hidden_size, 1)
        
        Returns:
            y_hat_sequence: List of predictions
            cache: Dictionary containing intermediate values for backprop
        """
        T = len(x_sequence) # number of time steps
        
        h_sequence = [] # hidden state sequence
        a_sequence = [] # activation sequence
        o_sequence = [] # output sequence
        y_hat_sequence = [] # predicted output sequence
        
        if a_0 is None:
            a_prev = np.zeros((self.hidden_size, 1))
        else:
            a_prev = a_0
        
        # Forward propagation through time
        for t in range(T):
            # Equation (a): h^<t> = W_xh · x^<t> + W_ah · a^<t-1> + b_h
            h_t = self.W_xh @ x_sequence[t] + self.W_ah @ a_prev + self.b_h
            
            # Equation (b): a^<t> = tanh(h^<t>)
            a_t = np.tanh(h_t)
            
            # Equation (c): o^<t> = W_ao · a^<t> + b_o
            o_t = self.W_ao @ a_t + self.b_o
            
            # Equation (d): ŷ^<t> = softmax(o^<t>)
            y_hat_t =  self.softmax(o_t)
            
            # Store values
            h_sequence.append(h_t)
            a_sequence.append(a_t)
            o_sequence.append(o_t)
            y_hat_sequence.append(y_hat_t)            
            
            a_prev = a_t # update activation for next time step
        
        # cache for backward pass
        cache = {
            'x_sequence': x_sequence,
            'h_sequence': h_sequence,
            'a_sequence': a_sequence,
            'o_sequence': o_sequence,
            'y_hat_sequence': y_hat_sequence
        }
        
        return y_hat_sequence, cache
    
    def backward(self, y_sequence: List[np.ndarray], dL_do_sequence: List[np.ndarray], cache: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:        
        """
        Backward propagation through time
        
        Args:
            y_sequence: List of true labels (one-hot encoded)
            dL_do_sequence: List of gradients of loss with respect to output
            cache: Dictionary from forward pass
        
        Returns:
            gradients: Dictionary containing gradients for all parameters
        """
        T = len(y_sequence)
        x_sequence = cache['x_sequence'] # input sequence
        a_sequence = cache['a_sequence'] # activation sequence
        
        # Initialize gradients
        dL_dW_xh = np.zeros_like(self.W_xh)
        dL_dW_ah = np.zeros_like(self.W_ah)
        dL_dW_ao = np.zeros_like(self.W_ao)
        dL_db_h = np.zeros_like(self.b_h)
        dL_db_o = np.zeros_like(self.b_o)
        
        # Initialize gradients for last time step as 0
        dL_da_next = np.zeros((self.hidden_size, 1))
        
        # Backward pass through time
        for t in reversed(range(T)):
            # ∂L^<t>/∂o^<t>
            dL_do_t = dL_do_sequence[t] 
            
            # Equation (c): ∂L^<t>/∂a^<t> = W_ao^T · ∂L^<t>/∂o^<t> + W_ah^T · ∂L^<t>/∂h^<t+1>
            dL_da_t = self.W_ao.T @ dL_do_t + self.W_ah.T @ dL_da_next
            
            # Equation (d): ∂L^<t>/∂h^<t> = ∂L^<t>/∂a^<t> · (1 - (a^<t>)²)
            # Here is a element-wise multiplication as ∂L^<t>/∂a^<t> and a^<t> are mx1 vectors
            dL_dh_t = dL_da_t * (1 - a_sequence[t]**2) 
            
            # Acumulate gradients
            # Equation (e): ∂L/∂W_ao = Σ(∂L^<t>/∂o^<t> · (a^<t>)^T)
            dL_dW_ao += dL_do_t @ a_sequence[t].T
            
            # Equation (f): ∂L/∂b_o = Σ(∂L^<t>/∂o^<t>)
            dL_db_o += dL_do_t
            
            # Equation (g): ∂L/∂W_ah = Σ(∂L^<t>/∂h^<t> · (a^<t-1>)^T)
            if t > 0:
                dL_dW_ah += dL_dh_t @ a_sequence[t-1].T
            else:
                # For t=0, use zero initial state 
                dL_dW_ah += dL_dh_t @ np.zeros((self.hidden_size, 1)).T
                
            # Equation (h): ∂L/∂W_xh = Σ(∂L^<t>/∂h^<t> · (x^<t>)^T)
            dL_dW_xh += dL_dh_t @ x_sequence[t].T
            
            # Equation (i): ∂L/∂b_h = Σ(∂L^<t>/∂h^<t>)
            dL_db_h += dL_dh_t
            
            # Update gradient flowing to previous timestep
            # Equation (b): ∂L^<t-1>/∂h^<t> = ∂L^<t>/∂h^<t>
            dL_da_next = dL_dh_t
        
        # Gradient clipping to prevent exploding gradients
        for g in [dL_dW_ao, dL_db_o, dL_dW_ah, dL_dW_xh, dL_db_h]:
            np.clip(g, -5, 5, out=g)   
         
        gradients = {
            'dL_dW_ao': dL_dW_ao,
            'dL_db_o': dL_db_o,
            'dL_dW_ah': dL_dW_ah,
            'dL_dW_xh': dL_dW_xh,
            'dL_db_h': dL_db_h
        }
        
        return gradients
    
    def SGD(self, gradients: Dict[str, np.ndarray]):
        """
        Update parameters using gradient descent
        
        Args:
            gradients: Dictionary containing gradients from backward pass
        """
        # Parameter update rules with learning rate η
        self.W_ao -= self.lr * gradients['dL_dW_ao']
        self.b_o -= self.lr * gradients['dL_db_o']
        self.W_ah -= self.lr * gradients['dL_dW_ah']
        self.W_xh -= self.lr * gradients['dL_dW_xh']
        self.b_h -= self.lr * gradients['dL_db_h']
    
    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute loss for a single time step
        
        Args:
            y_hat: Predicted output
            y: True output
        
        Returns:
            loss: loss for a single time step
        """
        return -np.sum(y * np.log(y_hat + 1e-8)) # Cross-entropy loss
    
    def total_loss(self, y_hat_sequence: List[np.ndarray], y_sequence: List[np.ndarray]) -> float:
        """
        Compute total loss
        
        Args:
            y_hat_sequence: List of predictions
            y_sequence: List of true labels (one-hot encoded)
        
        Returns:
            loss: Total loss L = Σ L^<t>
        """
        loss = 0
        for t in range(len(y_sequence)):
            loss += self.loss(y_hat_sequence[t], y_sequence[t])
        return loss
    
    def compute_output_error(self, y_sequence: List[np.ndarray], y_hat_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """ 
        Compute output error (∂L^<t>/∂o^<t>)
        
        Args:
            y_sequence: List of true labels
            y_hat_sequence: List of predicted labels
        
        Returns: 
            dL_do_sequence: List of gradients of loss with respect to output
        """
        # Equation (a): ∂L^<t>/∂o^<t> = ŷ^<t> - y^<t>
        return [y_hat_sequence[t] - y_sequence[t] for t in range(len(y_sequence))] 
    
    def train_step(self, x_sequence: List[np.ndarray], y_sequence: List[np.ndarray]) -> float:
        """
        One training step: forward, backward, and parameter update
        
        Args:
            x_sequence: List of input vectors
            y_sequence: List of true labels
        
        Returns:
            loss: Training loss for this sequence
        """
        # Forward pass
        y_hat_sequence, cache = self.forward(x_sequence)
        
        # Compute loss
        loss = self.total_loss(y_hat_sequence, y_sequence)
        
        # Compute deltas of loss with respect to output ∂L^<t>/∂o^<t>
        dL_do_sequence = self.compute_output_error(y_sequence, y_hat_sequence)
        
        # Backward pass
        gradients = self.backward(y_sequence, dL_do_sequence, cache)
        
        # Update parameters
        self.SGD(gradients)
        
        return loss