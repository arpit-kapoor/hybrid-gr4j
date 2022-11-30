import torch 
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
                
        super().__init__()

        # Input Dims
        self.input_dim = input_dim

        # Hidden Dims
        self.hidden_dim = hidden_dim

        # Output Dims
        self.output_dim = output_dim

        self.n_layers = n_layers

        # RNN layer
        self.lstm_layer = nn.LSTM(self.input_dim, 
                                self.hidden_dim, 
                                self.n_layers, 
                                batch_first=True)
        
        # Fully-connected output layer
        self.fc1 = nn.Linear(self.hidden_dim,
                            self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim,
                            self.output_dim)

        # Initialize weights
        self.init_weights()

    def forward(self, x):

        # Validate input shape
        assert len(x.shape)==3, f"Expected input to be 3-dim, got {len(x.shape)}"

        # Get dimensions of the input
        batch_size, seq_size, input_size = x.shape

        # Initialize hidden_state
        hidden, cell = self.init_zero_hidden(batch_size)

        # Pass through the recurrent layer
        out, (hidden, cell) = self.lstm_layer(x, (hidden, cell))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = torch.tanh(out[:, -1, :].contiguous().view(-1, self.hidden_dim))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)

        return out

    
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
                Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, requires_grad=False)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, requires_grad=False)
        return h_0, c_0
    

    def init_weights(self):
        for p in self.lstm_layer.parameters():
            nn.init.normal_(p)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)