# CompNeuroCam


# Assignment 2:

### **Model 1: no recurrence**
For this model, $n = m$ and $W^{(1)} = 0_m$ (the $m \times m$ matrix full of zeros).

---

### **Model 2: random symmetric connectivity**
For this model, $n = m$ and  
$$
W^{(2)} = \mathcal{R}(\tilde{W} + \tilde{W}^\top, \alpha)
$$  
with $\tilde{W}_{ij} \sim \mathcal{N}(0, 1)$ i.i.d.  
The random weights will be generated once and for all before any simulation of the network dynamics with various $\theta$'s.

---

### **Model 3: symmetric ring structure**
For this model, $n = m$ and  
$$
W^{(3)} = \mathcal{R}(\tilde{W}, \alpha)
$$  
with $\tilde{W}_{ij} = \mathcal{V}(\phi_i - \phi_j)$, where $\{ \phi_i \}$ and $\mathcal{V}(\cdot)$ have been defined previously.

---

### **Model 4: balanced ring structure**
For this model, $n = 2m$, and  
$$
W^{(4)} = 
\begin{pmatrix}
\tilde{W} & -\tilde{W} \\
\tilde{W} & -\tilde{W}
\end{pmatrix}
$$  
with $\tilde{W}_{ij} = \mathcal{R}(W^{(3)}, \alpha')$.  
As $n = 2m$, we can no longer use $B = C = I$ as in Table 1. Instead, we use:

$$
B = \begin{pmatrix} I_m \\ 0_m \end{pmatrix}, \quad 
C = (I_m, 0_m)
$$

where $I_m$ denotes the $m \times m$ identity matrix.



