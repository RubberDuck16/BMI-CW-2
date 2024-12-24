from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import lfilter
from numpy.linalg import inv


def kalman_filter(observations, A, C, Q, R, mu0, Sigma0):
    _, time_horizon = observations.shape

    filtered_mews, filtered_sigmas = [mu0], [Sigma0]

    mew_t, sigma_t = mu0, Sigma0
    for t in range(1, time_horizon):
        x_next = observations[:, t]

        term1 = C.T @ (inv(R) @ C)
        term2 = A @ (sigma_t @ A.T) + Q
        term2 = inv(term2)
        sigma_new_inverse = term1 + term2 
        sigma_t = inv(sigma_new_inverse)

        prediction = A @ mew_t
        Kalman_gain = sigma_t @ (C.T @ inv(R)) 
        pred_err = x_next - C @ A @ mew_t
        mew_t = prediction + Kalman_gain @ pred_err
        filtered_mews.append(mew_t)
        filtered_sigmas.append(sigma_t)

    return filtered_mews, filtered_sigmas


def kalman_smooth(filtered_mews, filtered_sigmas, A, Q):
    time_horizon = len(filtered_mews)
    dim_z = A.shape[0]

    smoothed_mews = np.zeros((dim_z, time_horizon))                 # 10 x 16
    smoothed_sigmas = np.zeros((dim_z, dim_z, time_horizon))        # 10 x 10 x 16

    smoothed_mews[:, time_horizon-1] = filtered_mews[-1]
    smoothed_sigmas[:, :, time_horizon-1] = filtered_sigmas[-1]

    for t in range(time_horizon-2, -1, -1):
        P = A @ (filtered_sigmas[t] @ A.T) + Q
        G = filtered_sigmas[t] @ (A.T @ inv(P))
        smoothed_mews[:, t] = filtered_mews[t] + G @ (smoothed_mews[:, t+1] - (A @ filtered_mews[t]))
        smoothed_sigmas[:, :, t] = filtered_sigmas[t] + G @ (smoothed_sigmas[:,:,t+1] - P) @ G.T
    
    return smoothed_mews, smoothed_sigmas


def D_star(X, Z):
    N, K, _ = X.shape
    M = Z.shape[0]                   # latent dimension
    
    matrix1 = np.zeros((N, M))
    matrix2 = np.zeros((M, M))

    for k in range(K):
        x = X[:, k, :]
        z = Z[:, k, :]
        matrix1 = matrix1 + (x @ z.T)
        matrix2 = matrix2 + (z @ z.T)
    return matrix1 @ inv(matrix2)

def S_star(X, Z):
    N, K, T = X.shape
    M = Z.shape[0]   
    
    matrix1 = np.zeros((N, N))
    matrix2 = np.zeros((M, N))

    for k in range(K):
        x = X[:, k, :]
        z = Z[:, k, :]
        matrix1 = matrix1 + (x @ x.T)
        matrix2 = matrix2 + (z @ x.T)
    
    D = D_star(X, Z)
    term = matrix1 - (D @ matrix2)
    return (1/K*T) * term



# grab the data from the server
r = requests.get('http://4G10.cbl-cambridge.org/data.npz', stream = True)
data = np.load(BytesIO(r.raw.read()))

hand_train = data["hand_train"]
neural_train = data["neural_train"]
neural_test = data["neural_test"]
mu0 = data["hand_KF_mu0"].reshape(10)
Sigma0 = data["hand_KF_Sigma0"]
A = data["hand_KF_A"]
C = data["hand_KF_C"]
Q = data["hand_KF_Q"]
R = data["hand_KF_R"]


Z_smooth = np.zeros((10, 400, 16))

for k in range(400):
    observations = hand_train[:, k, :]
    f_m, f_s = kalman_filter(observations, A, C, Q, R, mu0, Sigma0)
    s_m, s_s = kalman_smooth(f_m, f_s, A, Q)
    Z_smooth[:, k, :] = s_m

# mean taken over time and trials in TRAINING SET
mean_over_time_and_conditions = np.mean(neural_train, axis=(1, 2))            
X = neural_train - mean_over_time_and_conditions[:, np.newaxis, np.newaxis]
X_test = neural_test - mean_over_time_and_conditions[:, np.newaxis, np.newaxis]

D = D_star(X, Z_smooth)
S = S_star(X, Z_smooth)

N, K, T = X_test.shape
V_predictions = np.zeros((2, K, T))

for k in range(K):
    print(k)
    observations = X_test[:, k, :]
    f_m, f_s = kalman_filter(observations, A, D, Q, S, mu0, Sigma0)
    z_k = np.asarray(f_m).T
    v_hat = C @ z_k
    V_predictions[:, k, :] = v_hat
    
np.save("v_pred2.npy", V_predictions)