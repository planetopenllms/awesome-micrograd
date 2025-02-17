
> These 94 lines of code are everything that is needed to train a neural network.
> Everything else is just efficiency.
>
> This is my earlier project Micrograd. It implements a scalar-valued auto-grad engine.
> You start with some numbers at the leafs 
> (usually the input data and the neural network parameters), 
> build up a computational graph with operations like + and * that mix them, 
> and the graph ends with a single value at the very end (the loss).
> You then go backwards through the graph applying chain rule at each node to calculate the gradients. 
> The gradients tell you how to nudge your parameters to decrease the loss (and hence improve your network).
>
> Sometimes when things get too complicated, I come back to this code and just breathe a little.
> But ok ok you also do have to know what the computational graph should be (e.g. MLP -> Transformer),
> what the loss function should be (e.g. autoregressive/diffusion), 
> how to best use the gradients for a parameter update (e.g. SGD -> AdamW) etc etc. 
> But it is the core of what is mostly happening.
>
> The [1986 paper from Rumelhart, Hinton, Williams](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) that popularized and used this algorithm (backpropagation) for training neural nets,
> [micrograd on Github](https://github.com/karpathy/micrograd) 
> and [my (now somewhat old) YouTube video](https://www.youtube.com/watch?v=VMj-3S1tku0) where I very slowly build and explain.
>
>  -- Andrej Karpathy,  June 2024


# Awesome Micro Grad (micrograd)



