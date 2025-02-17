
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




# Awesome Micrograd & Friends

Yes, you can! 
Build your own auto(matic) grad(ient) engine using reverse-mode auto(matic) diff(erenation)
from scratch. 
bonus - add a pytoch-like neural network library on top. 





## Official Micrograd Versions by Andrej Karpathy

Genesis @ <https://github.com/karpathy/micrograd>

> A tiny Autograd engine (with a bite! :)). 
> Implements backpropagation (reverse-mode autodiff) over a dynamically built
> DAG and a small neural networks library on top of it with a PyTorch-like API. 
> Both are tiny, with about 100 and 50 lines of code respectively. 
> The DAG only operates over scalar values, so e.g. we chop up each neuron into all 
> of its individual tiny adds and multiplies. 
> However, this is enough to build up entire deep neural nets doing binary classification, 
> as the demo notebook shows. Potentially useful for educational purposes.

EurekaLabs follow-up started in 2024 (part of LLMs 101) @ <https://github.com/EurekaLabsAI/micrograd> 


**Neural Networks: Zero to Hero**

Lecture 1: Building micrograd - the spelled-out intro to neural networks and backpropagation

- Code Notebooks
  - [Micrograd - First Part](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb)
  - [Micrograd - Second Part](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb)
- [YouTube Video](https://www.youtube.com/watch?v=VMj-3S1tku0)




##  Micrograd Extensions

<!--
### Micrograd "V2"  

uses numpy.ndarrays (with type float32) for values, 
adds require_grad options, 
adds more operations incl. matmul, sum, and more;  
tries to stay below the 1000 lines of code limit!

see <https://github.com/planetopenllms/llms.sandbox/tree/main/micrograd_v2>

-->


###  Micrograd++ / Micrograd Plus Plus (by Parsiad Azimzadeh)

incl. tensor support (via numpy.ndarrays) 
and GPU support and more

see <https://github.com/parsiad/micrograd-pp>


###  Micograd CUDA (by Matthieu Le Cauchois)

incl. Micrograd extension with basic 2D tensors and naïve matrix multiplication for MLP,
batching,
CUDA kernel for matrix multiplication,
and more

see <https://github.com/mlecauchois/micrograd-cuda>


### Tinygrad & Teenygrad (by George Hotz et al)

Tinygrad is a about 10000 lines of code Pytorch-like library with autograd,
incl. gpu support beyond CUDA and much more,
venture capital backed by tinycorp (with a million dollar investment).


Teenygrad is a 1000 lines of code Tinygrad.

see <https://github.com/tinygrad/tinygrad>  and <https://github.com/tinygrad/teenygrad>



##  Micrograd in Language X

### Typescript

**Micrograd TS** by Oleksii Trekhleb,
see <https://github.com/trekhleb/micrograd-ts> -
about 200 lines of typescript code

<details>
<summary>Show/Hide Sample</summary>

``` ts
// Inputs x1, x2
const x1 = v(2, { label: 'x1' })
const x2 = v(0, { label: 'x2' })

// Weights w1, w2
const w1 = v(-3, { label: 'w1' })
const w2 = v(1, { label: 'w2' })

// bias of the neuron
const b = v(6.8813735870195432, { label: 'b' })

// x1w1 + x2w2 + b
const x1w1 = x1.mul(w1)
x1w1.label = 'x1w1'

const x2w2 = x2.mul(w2)
x2w2.label = 'x2w2'

const x1w1x2w2 = x1w1.add(x2w2)
x1w1x2w2.label = 'x1w1x2w2'

const n = x1w1x2w2.add(b)
n.label = 'n'

const o = n.tanh()
o.label = 'o'

o.backward()
```

</details>

### C Lang

**micrograd.c** by Jaward Sesay,
see <https://github.com/Jaykef/micrograd.c>

<details>
<summary>Show Hide/Sample</summary>

``` c 
Value* a = value_new(-4.0);
Value* b = value_new(2.0);
Value* c = value_add(a, b);
Value* d = value_add(value_mul(a, b), value_pow(b, 3));
c = value_add(c, value_add(c, value_new(1)));
c = value_add(c, value_add(value_add(value_new(1), c), value_neg(a)));
d = value_add(d, value_add(value_mul(d, value_new(2)), value_relu(value_add(b, a))));
d = value_add(d, value_add(value_mul(value_new(3), d), value_relu(value_sub(b, a))));
Value* e = value_sub(c, d);
Value* f = value_pow(e, 2);
Value* g = value_div(f, value_new(2.0));
g = value_add(g, value_div(value_new(10.0), f));
backward(g);

double tol = 1e-4; 
printf("g->data: %.6f\n", g->data);

backward(g);

printf("a->grad: %.6f\n", a->grad);
printf("b->grad: %.6f\n", b->grad);
```

</details>



### Go Lang

**go-micrograd** by Nathan Bary, 
see <https://github.com/nathan-barry/go-micrograd>


- [Intro to Autograd Engines: Karpathy's Micrograd Implemented in Go](https://nathan.rs/posts/go-micrograd/) by Nathan Bary, Nov 2023 


<details>
<summary>Show/Hide Sample</summary>

``` go
    x := New(2)
	w := New(0.4) // pretend random init
	y := New(4)

	for k := 0; k < 6; k++ {

		// forward pass
		ypred := Mul(w, x)
		loss := Pow(Sub(ypred, y), New(2))

		// backward pass
		w.Grad = 0 // zero previous gradients
		loss.Backward()

		// update weights
		w.Data += -0.1 * w.Grad

		fmt.Printf("Iter: %2v, Loss: %.4v, w: %.4v\n",
            k, loss.Data, w.Data)
	}
```

</details>


### Crystal 

**micrograd.cr** by nogginly,
see <https://github.com/nogginly/micrograd.cr>

<details>
<summary>Show/Hide Sample</summary>

``` cr
require "micrograd"

alias NNFloat = Float32
alias NNValue = MicroGrad::Value(NNFloat)

a = NNValue[-4]
b = NNValue[2]
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu
d += 3 * d + (b - a).relu
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

puts "g: #{g}" # prints 24.7041, the outcome of this forward pass
g.backward
puts "a: #{a}" # prints 138.8338, i.e. the numerical value of dg/da
puts "b: #{b}" # prints 645.5773, i.e. the numerical value of dg/db
```

</details>


### Ruby

**micrograd** by Nithin Bekal,
see <https://github.com/nithinbekal/micrograd>, [(rdocs)](https://www.rubydoc.info/gems/micrograd)

<details>
<summary>Show/Hide Sample</summary>

``` ruby
include Micrograd

a = Value.new(2.0)
b = Value.new(-3.0)
c = Value.new(10.0)
e = a * b
d = e + c
f = Value.new(-2.0)

l = d * f

# Walk through all the values and calculate gradients for them.
l.start_backward
```

</details>


**backprop** by Rick Hull,
see <https://github.com/rickhull/backprop>, [(rdocs)](https://www.rubydoc.info/gems/backprop)




### Julia

**MicroGrad.jl** by Lior Sinai,
see <https://github.com/LiorSinai/MicroGrad.jl>



##  Micrograd Articles 







## History Corner


Grokking Deep Learning by Andrew W. Trask  builds a micrograd-like engine w/ a pytorch-like library on top in 2019 (!), see 

> Chapter 13 - Introducing automatic optimization: let's build a deep learning framework.
>
> [...]
> Introduction to automatic gradient computation (autograd)
> Previously, you performed backpropagation by hand. Let's make it automatic!
> [...]

The (code) notebook is free online <https://github.com/iamtrask/Grokking-Deep-Learning>,
see [Chapter13 - Intro to Automatic Differentiation - Let's Build A Deep Learning Framework.ipynb](https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter13%20-%20Intro%20to%20Automatic%20Differentiation%20-%20Let's%20Build%20A%20Deep%20Learning%20Framework.ipynb).




