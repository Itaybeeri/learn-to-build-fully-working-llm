"""
Module 2 demo — a tiny neural network, built and trained FROM SCRATCH.

Goal: teach a 2-input network the XOR rule:
        (0,0) -> 0   (0,1) -> 1   (1,0) -> 1   (1,1) -> 0
        "output 1 only when the two inputs DIFFER."

Why XOR? A single linear layer CANNOT solve it -- you need a hidden layer
with a non-linear bend (the sigmoid). So if this learns, it is living proof
of the Module 2 lesson "the activation's bend is what gives a network power."

Nothing is hidden here: we write the forward pass, the loss, backprop, and
the weight update BY HAND -- exactly the four steps of the training loop.
Pure Python, no libraries to install. Run it with:   python xor_from_scratch.py
"""

import math
import random

random.seed(1)  # makes the run reproducible (same "random" start every time)

# ---------------------------------------------------------------------------
# The data: the four XOR examples. Each is (inputs, true answer).
# ---------------------------------------------------------------------------
DATA = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

# ---------------------------------------------------------------------------
# The activation function: the sigmoid (our smooth 0-to-1 "bend").
#   sigmoid(z) = 1 / (1 + e^-z)
# Its slope (needed for backprop) has a famously tidy form: s * (1 - s).
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_slope(s):       # s is an ALREADY-computed sigmoid output
    return s * (1.0 - s)

# ---------------------------------------------------------------------------
# The network's WEIGHTS (and biases) -- the dials training will tune.
# Architecture: 2 inputs -> hidden layer of 2 neurons -> 1 output neuron.
#
#   A "bias" is one extra always-on number added to a neuron's weighted sum
#   (a weighted sum of a constant 1). It lets a neuron shift its threshold;
#   XOR needs it. Think of it as each neuron's built-in "lean."
#
# We start every weight RANDOM -- the model knows nothing yet.
# ---------------------------------------------------------------------------
def rand():
    return random.uniform(-1.0, 1.0)

# Hidden layer: 2 neurons, each reading the 2 inputs -> a 2x2 weight grid + 2 biases
W1 = [[rand(), rand()],   # neuron H1's weights (one per input)
      [rand(), rand()]]   # neuron H2's weights
b1 = [rand(), rand()]     # one bias per hidden neuron

# Output layer: 1 neuron reading the 2 hidden outputs -> 2 weights + 1 bias
W2 = [rand(), rand()]
b2 = rand()

LEARNING_RATE = 0.5       # the step size: how big a nudge each update (a hyperparameter)
EPOCHS = 20000            # how many full passes over the 4 examples


# ---------------------------------------------------------------------------
# FORWARD PASS: push one input vector through the network -> a prediction.
# Returns the prediction AND the hidden outputs (needed later by backprop).
# ---------------------------------------------------------------------------
def forward(x):
    # Hidden layer: each neuron does (weighted sum + bias), then the sigmoid bend.
    h = [0.0, 0.0]
    for j in range(2):                       # for each hidden neuron j
        z = b1[j]                            # start with its bias
        for i in range(2):                   # add weight*input over the 2 inputs
            z += W1[j][i] * x[i]
        h[j] = sigmoid(z)                    # the bend

    # Output neuron: weighted sum of the hidden outputs + bias, then sigmoid.
    z_out = b2
    for j in range(2):
        z_out += W2[j] * h[j]
    pred = sigmoid(z_out)                    # final prediction, a probability in (0,1)

    return pred, h


# ---------------------------------------------------------------------------
# LOSS: binary cross-entropy -- how wrong one prediction is.
#   loss = -[ y*log(pred) + (1-y)*log(1-pred) ]
# (the two-outcome version of the -log(p_correct) idea from the loss note).
# ---------------------------------------------------------------------------
def loss(pred, y):
    eps = 1e-9                               # tiny guard so we never take log(0)
    return -(y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps))


# ---------------------------------------------------------------------------
# THE TRAINING LOOP: for each epoch, for each example, do the 4 steps.
# ---------------------------------------------------------------------------
print("Training a 2-2-1 network to learn XOR (from scratch)...\n")

for epoch in range(EPOCHS + 1):
    total_loss = 0.0

    for x, y in DATA:                        # (with only 4 examples, batch = 1 example)
        # 1) FORWARD ---------------------------------------------------------
        pred, h = forward(x)
        total_loss += loss(pred, y)

        # 2) BACKWARD (backprop) -- blame flowing backward from the loss ------
        # Output neuron: for sigmoid + cross-entropy the blame is simply (pred - y).
        dz_out = pred - y                    # how the output's pre-sigmoid score is to blame

        # blame on the output layer's weights/bias = blame * what fed into it
        dW2 = [dz_out * h[0], dz_out * h[1]]
        db2 = dz_out

        # pass the blame BACK to each hidden neuron: its share = dz_out * its output weight,
        # then through its own sigmoid bend (multiply by the local slope h*(1-h)).
        dz_h = [0.0, 0.0]
        for j in range(2):
            blame_reaching_h = dz_out * W2[j]            # baton handed back from the output
            dz_h[j] = blame_reaching_h * sigmoid_slope(h[j])

        # blame on the hidden layer's weights/biases = blame * the inputs that fed them
        dW1 = [[dz_h[0] * x[0], dz_h[0] * x[1]],
               [dz_h[1] * x[0], dz_h[1] * x[1]]]
        db1 = [dz_h[0], dz_h[1]]

        # 3 & 4) UPDATE -- nudge every weight opposite its slope -------------
        for j in range(2):
            for i in range(2):
                W1[j][i] -= LEARNING_RATE * dW1[j][i]
            b1[j] -= LEARNING_RATE * db1[j]
        for j in range(2):
            W2[j] -= LEARNING_RATE * dW2[j]
        b2 -= LEARNING_RATE * db2

    # progress report every so often
    if epoch % 2000 == 0:
        print(f"  epoch {epoch:>6}   average loss = {total_loss / len(DATA):.4f}")

# ---------------------------------------------------------------------------
# RESULTS: after training, see what the network now predicts.
# ---------------------------------------------------------------------------
print("\nTrained! Final predictions (want ~0 or ~1):\n")
for x, y in DATA:
    pred, _ = forward(x)
    verdict = "OK" if round(pred) == y else "WRONG"
    print(f"  input {x}  ->  pred {pred:.3f}  (true {int(y)})  [{verdict}]")
