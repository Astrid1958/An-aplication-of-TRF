# An-aplication-of-TRF
Once the data are ready, the EEG is paired with the stimulus, which in my case is the binary vector; for this purpose, the data are iterated over subject, speed, and each audio signal:

\begin{verbatim}
if vel == "slow":
               S = np.load(f"/content/drive/MyDrive/Matrices_TRF/matriz_binaria/slow/ut{ut}.npy")
                else:
                S = np.load(f"/content/drive/MyDrive/Matrices_TRF/matriz_binaria/fast/ut{ut}_F.npy")
                # === Load EEG ===
                eeg = np.load(
               f"/content/drive/MyDrive/EEG_preprocesado_sep/{vel}/bb{sujeto}_ut{ut}.npy"
                )[:, canales]
\end{verbatim}

Then, I proceeded to match the number of samples $\tilde{T}$ as before \eqref{muestras}. To do this, I defined a simple function that compares the sample sizes and redefines them by choosing the minimum, so that the EEG data retain dimension $\R^{\tilde{T}_m\times C}$ and the stimulus retains dimension $\R^{\tilde{T}_m\times 1}$, $m=1,...,M$, $ M=\lfloor total/C \rfloor$, where the total is previously defined in \eqref{total}. Unlike the processing in Section \ref{detalles herramientas 62}, I considered the complete audio without splitting the time samples and separated all data into fast and slow conditions. For each condition, I implemented a function that randomly divides the cases, storing 80\% of them for model training and 20\% for testing. Within the training data, I further split them into 80\% training and 20\% validation. I organized these data using nested dictionaries that allow the information to be structured according to the experimental condition (stimulus speed) and subject as follows:

\begin{verbatim}
stimuli[vel][sujeto] = {"train": S_tr, "test": S_test,"val": S_val}.
responses[vel][sujeto] = {"train": EEG_tr, "test": EEG_test,"val": EEG_val}.
\end{verbatim}

The complete structure can be represented as:

\begin{equation*}
\texttt{stimuli} =
\begin{cases}
\text{slow} \rightarrow \{ \text{subject}
\rightarrow \{\text{val}, \text{train}, \text{test} \} \} ,\\
\text{fast} \rightarrow \{ \text{subject} \rightarrow \{ \text{val},\text{train}, \text{test} \} \}.
\end{cases}
\end{equation*}

Similarly, this representation applies to \texttt{responses}.

\\

For each subject and condition, the data correspond to a list of independent realizations. Let $N=\lfloor0.64M\rfloor$; the training data are defined as:

\begin{align*}
&S_{\text{train}} = \{S^{(1)}, S^{(2)}, \dots, S^{(N)}\},  \\
&E_{\text{train}} = \{E^{(1)}, E^{(2)}, \dots, E^{(N)}\},
\end{align*}

and similarly, let $L=\lfloor 0.8M\rfloor$; the validation data are defined as:

\begin{align}\label{validacion}
    &S_{\text{val}} = \{S^{(N+1)}, \dots, S^{(L)}\},  \\
&E_{\text{val}} = \{E^{(N+1)}, \dots, E^{(L)}\},
\end{align}

and the test data:

\begin{align*}
    &S_{\text{test}} = \{S^{(L+1)}, \dots, S^{(M)}\},  \\
&E_{\text{test}} = \{E^{(L+1)}, \dots, E^{(M)}\},
\end{align*}

for each $m=1,...,M$:

\begin{align*}
&S^{(m)} \in \mathbb{R}^{\tilde{T}_m \times 1},\\
&E^{(m)} \in \mathbb{R}^{\tilde{T}_ m\times C},
\end{align*}

\subsubsection*{Selection of the regularization parameter $\lambda$}

Next, the procedure implemented for selecting the ridge regression parameter $\lambda$ \eqref{ridge} using a validation set is described.

Since mTRFcrossval is a function exclusive to Matlab, I defined a new function denoted by \verb|seleccionar_lambda| to carry out parameter selection. The function \verb|seleccionar_lambda| takes as input: the training data $(S_{\text{train}}, E_{\text{train}})$, the validation data $(S_{\text{val}}, E_{\text{val}})$, the temporal parameters of the model, and a set of candidate values for $\lambda$. Within the function, for each value $\lambda_l$ in a candidate set $\{\lambda_1, \dots, \lambda_\ell\}$, a TRF model is trained:

\begin{equation*}
   \texttt{model\_l} = \texttt{trf.train}(S_{\text{train}}, E_{\text{train}}, fs, \tau_{\min}, \tau_{\max}, \lambda),
\end{equation*}

which is equivalent to solving:

\begin{align*}
\hat{W}_{\lambda_l}
=
\arg\min_W
\|E_{\text{train}} - S_{\text{train}} W\|_2^2
+
\lambda_l \|W\|_2^2.
\end{align*}

In this way, $\ell$ models are obtained, one for each $\lambda_l$. Then, each trained model is used to predict the response on the validation data. For each $\lambda_l$:

\begin{equation*}
\texttt{y\_pred\_l = trf.predict}(S_{\text{val}}, \texttt{model})
\end{equation*}

which corresponds to:

\begin{align*}
\hat{E}^{(l)}_{\text{val}} = S_{\text{val}}^{}\hat{W}_{\lambda_l}.
\end{align*}

Note that,

\begin{equation*}
\hat{E}^{(l)}_{\text{val}} = \{\hat{E}^{(l,N+1)},...,\hat{E}^{(l,L)}\}.
\end{equation*}

Then, each pair of true and predicted signals $\hat{E}^{(l,m)}$ and $E^{(m)}$ is transformed into a one-dimensional vector through:

\begin{equation*}
\texttt{yt = np.asarray(yt).ravel()}, \quad \texttt{yp = np.asarray(yp).ravel}.
\end{equation*}

This is equivalent to:

\begin{align*}
E^{(m)}_{\text{val}} \in \mathbb{R}^{T_m \times C}
&\longrightarrow \tilde{E}^{(m)} \in \mathbb{R}^{T_m C}, \\
 \hat{E}^{(l,m)}_{\text{val}}\in \mathbb{R}^{T_m \times C}
&\longrightarrow \tilde{\hat{E}}^{(l,m)} \in \mathbb{R}^{T_m C}.
\end{align*}

In this way, for each $l=1,...,\ell$, and $m = N+1, \dots, L$, I ensured that both arrays $\hat{E}^{(l,m)}$ and $E^{(m)}$ had the same dimensions, so that $\hat{E}^{(l,m)},E^{(m)} \in \R^{T_m}$, where in this framework $T_m$ is defined as the minimum number of samples between both:

\begin{equation*}
T_m= \min(\text{len}( E_{\text{val}}^{(m)} ), \text{len}(\hat{E}^{(l,m)}_{\text{val}})).
\end{equation*}

All signals are then concatenated:

\begin{align*}
Y &=
\begin{bmatrix}
\tilde{E}^{(N+1)} \\
\vdots \\
\tilde{E}^{(L)}
\end{bmatrix}
\in \mathbb{R}^{T^{(l)}}, \\
\\
\hat{Y}^{(l)} &=
\begin{bmatrix}
\tilde{\hat{E}}^{(l,N+1)} \\
\vdots \\
\tilde{\hat{E}}^{(l,L)}
\end{bmatrix}
\in \mathbb{R}^{T^{(l)}},
\end{align*}

where:

\begin{align*}
T^{(l)} = \sum_{m=N+1}^L T_m.
\end{align*}

It is important to note that the true signal $Y$ is unique and does not depend on the regularization parameter. In contrast, the predictions $\hat{Y}^{(l)}$ do depend on $\lambda_l$, so a different prediction is obtained for each trained model.

Next, an empty list was created to store the Pearson coefficients associated with each value of $\lambda$, with the goal of finding the best value of $\lambda$ that optimizes the model (i.e., maximizes the Pearson correlation). The model performance is evaluated through the Pearson correlation coefficient between both signals for each $\lambda_l$, $l=1,...,\ell$:

\begin{verbatim}
r = pearsonr(y_true, y_pred),
\end{verbatim}

which is equivalent to \eqref{ro}, maintaining the properties defined in Section \ref{detalles herramientas 62}. Each correlation value obtained for a specific $\lambda$ is stored in a list called \verb|scores|.

\begin{verbatim}
scores.append(r),
\end{verbatim}

thus obtaining a set of metrics:

\begin{align*}
\{\rho_{\lambda_1}, \rho_{\lambda_2}, \dots, \rho_{\lambda_\ell}\},
\end{align*}

To select the best lambda, once all candidate values of the regularization parameter have been evaluated, the one that maximizes the average correlation is selected:

\begin{equation*}
\lambda^{*} = \arg\max_{\lambda_l} \; \rho(\lambda_l).
\end{equation*}
