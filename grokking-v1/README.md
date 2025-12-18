#  Autograd Engine from the Book - Grokking Deep Learning

this automatic gradient computation a.k.a. autograd engine starts with / follows the
initial code from the book [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) by Andrew Trask, Jan 2019   

see Chapter 13 - titled
"Intro to Automatic Differentiation - Let's Build A Deep Learning Framework"
for the original code and write-up.



## open questions / todos - discuss

- [ ]  use raw gradient (NOT wrapped in own Tensor) - why? why not?
- [ ]   rework / recheck  - requires_grad check / propagation
- [ ]  autoadd name (if not given)   e.g. _v1, _v2, etc.  - use global count
