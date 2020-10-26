import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn

"""
used for thesis-fundamental
"""

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def relu(x):
    a = []
    for item in x:
        a.append(max(item,0))
    return a


x = np.arange(-10., 10., 0.2)
xx = np.arange(-5., 5., 0.2)

sig = sigmoid(x)
r = relu(x)
seaborn.set(style='ticks')



fig, ax = plt.subplots()
ax.plot(x, sig)
ax.set_aspect('auto')
ax.grid(True, which='both')
ax.xaxis.set_ticks([-10,-5,0,5,10])
seaborn.despine(ax=ax, offset=0)
plt.legend()

fig, ax = plt.subplots()
ax.plot(x, r)
ax.set_aspect('auto')
ax.grid(True, which='both')
ax.xaxis.set_ticks([-10,-5,0,5,10])
seaborn.despine(ax=ax, offset=0)
plt.legend()


plt.rc('legend', fontsize=24)    # legend fontsize
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=35)     # fontsize of the axes title
#plt.rc('axes', labelsize=35)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('figure', titlesize=35)  # fontsize of the figure title

fig, ax = plt.subplots()
ax.plot(xx, xx*sigmoid(xx),"b-",label="swish")
ax.plot(xx, relu(xx),"r--",label="ReLU")
ax.set_aspect('auto')
ax.grid(True, which='both')
ax.xaxis.set_ticks([-5,0,5])
seaborn.despine(ax=ax, offset=0)
plt.legend()

fig, ax = plt.subplots()
ax.plot(xx, xx*sigmoid(xx),"b--",label="swish")
ax.plot(xx, xx*relu(xx+3)/6,"g",label="h-swish")
ax.set_aspect('auto')
ax.grid(True, which='both')
ax.xaxis.set_ticks([-5,0,5])
seaborn.despine(ax=ax, offset=0)
plt.legend()



plt.show()

@article{mobilnet_actF13,
  title={Sigmoid-weighted linear units for neural network function approximation in reinforcement learning},
  author={Elfwing, Stefan and Uchibe, Eiji and Doya, Kenji},
  journal={Neural Networks},
  volume={107},
  pages={3--11},
  year={2018},
  publisher={Elsevier}
}

@article{mobilnet_actF36,
  title={Searching for activation functions. arXiv 2017},
  author={Ramachandran, P and Zoph, B and Le, QV},
  journal={arXiv preprint arXiv:1710.05941}
}








\subsection{Activation Function}
\paragraph{Linear Activation Functions}

It is theoretically possible but not suggested to use a linear activation function, which turns the the perceptron into a linear classifier. However, a single perception's representation power is too limited to handle complex problems. The linear combination of these linear classifiers is still linear, meaning it can be represented by a single linear perception without any improved complexity. Moreover, the linear activation functions' derivative is a constant and unrelated to the weighted sum. So during training, we can't apply the backpropagation (see section \ref{sec:Backpropagation}) to adapt the weights to improve its performance.

\paragraph{Nonlinear Activation Functions}
To address these two problems of linear activation functions, non-linear activation functions are widely used in neural networks. The Sigmoid and \gls{relu} are two popular choices. The \textit{sigmoid} is defined as:

\begin{equation}
    sigmoid\ x =  \frac{\mathrm{1} }{\mathrm{1} + e^{-x}}
\end{equation}

From the Figure \ref{fig:sigmoid}
we could see the \textit{sigmoid} has smooth gradient and bounded output values. which are favorable features as an activation function. However, due to the flattened curve at both ends of x-axis, is facing challenge of gradient vanishing, i.e. the output is more and more unsensitive to the inputs as the absolute value of input increases. The vanishing gradient makes it harder to train the network, especially  for the deep networks like the MobileNetV3.

\begin{figure}[!htb]
    \captionsetup{format=plain, justification = raggedright, singlelinecheck = false}
    \subfigure[Sigmoid]{
        \includegraphics[width=.45\linewidth]{Graphics/2Fundamental/sigmoid.png}
    \label{fig:sigmoid}
    }
    \subfigure[{\gls{relu}}]{
        \includegraphics[width=.45\linewidth]{Graphics/2Fundamental/relu.png}
    \label{fig:relu}
    }
    \caption{Graph of sigmoid and ReLU functions}
 \end{figure}


% more info refer to https://arxiv.org/pdf/1505.00853.pdf
The \gls{relu} is basically a piecewise linear function, pruned its negative part to zero, see Figure \ref{fig:relu}. Its unsuppressed positive part solves vanishing gradient problem \cite{xu2015empirical_relu}. Its simple form is much more computationally efficient than \textit{sigmoid}.


MobileNetV3 \cite{mobilenetv3} intends to use a nonlinearity called \textit{swish} \cite{mobilnet_actF36} \cite{mobilnet_actF13}, that could significantly improves the accuracy of neural networks as a drop-in replacement of \gls{relu}.It is defined as:
\begin{equation}
    swish\  x = x \cdot \sigma(x)
\end{equation}

However, the \textit{swish} is more expensive than ReLU mainly


 \begin{figure}[!htb]
    \subfigure[Swish \& {\gls{relu}}]{
        \includegraphics[width=.45\linewidth]{Graphics/2Fundamental/relu_swish.png}
    }
    \subfigure[H-Swish \& {\gls{relu}}]{
        \includegraphics[width=.45\linewidth]{Graphics/2Fundamental/hswish.png}
    }
    \label{dgdsfhd}
    \caption{Unaligned points detected by a rotation LiDAR on a moving vehicle}
    \label{fig:hswish}
 \end{figure}
