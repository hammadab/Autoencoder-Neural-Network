def aeCost(We, data, params, h, y):
    # We = [w1, w2, b1, b2]
    # data = x
    # params = {'Lin': Lin, 'Lhid': Lhid, 'lambda': lamb, 'beta': beta,'rho': rho}
    # calculates the cost and its partial derivatives
    # We = [W1 W2 b1 b2], a vector containing weights for first and second layers followed by bias terms
    # data is of size (Lin x N)
    # params is a structure with the following fields Lin (Lin), Lhid (Lhid), lambda (λ), beta (β), rho (ρ)
    # params = {'Lin': , 'Lhid': , 'lambda': , 'beta': , 'rho': }
    # h is output of hidden layer, size = (Lhid x N)
    # y is output of output layer, size = (Lout x N)
    N = data.shape[1]
    rho_hat = np.nansum(h, 1) / h.shape[1]
    kld = np.nan_to_num(params['rho'] * np.log(params['rho'] / rho_hat)) + np.nan_to_num((1 - params['rho']) * np.log((1 - params['rho']) / (1 - rho_hat)))
    dkld = np.reshape(np.nan_to_num((1 - params['rho']) / (1 - rho_hat)) - np.nan_to_num(params['rho'] / rho_hat), (params['Lhid'], 1))
    J = np.nansum(np.nansum((data - y)**2, 1)/(2*data.shape[1]) + params['lambda']*(np.nansum((np.concatenate((np.ravel(We[0]), np.ravel(We[1]), np.ravel(We[2]), np.ravel(We[3]))))**2))/2 + params['beta']*np.nansum(kld)) / N
    JgradW2 = np.nan_to_num(((y - data) * y * (1 - y)) @ np.transpose(h)) / N + params['lambda'] * (We[1])
    Jgradb2 = np.reshape(np.nansum(((y - data) * y * (1 - y)), 1) / N, We[3].shape) + params['lambda'] * (We[3])
    JgradW1 = np.nan_to_num(((np.transpose(We[1]) @ ((y - data) * y * (1 - y))) * h * (1 - h)) @ np.transpose(data)) / N + params['lambda'] * (We[0]) + params['beta'] * np.nan_to_num(dkld * np.reshape((h * (1 - h)) @ np.transpose(data), We[0].shape)) / N
    Jgradb1 = np.reshape(np.nansum(((np.transpose(We[1]) @ ((y - data) * y * (1 - y))) * h * (1 - h)), 1) / N, We[2].shape) + params['lambda'] * (We[2]) + params['beta'] * np.reshape(np.nansum(dkld * (h * (1 - h)), 1) / N, (params['Lhid'], 1))
    # J = np.nansum(np.nansum((data - y)**2, 1)/(2*data.shape[1]) + params['lambda']*(np.nansum((np.concatenate((np.ravel(We[0]), np.ravel(We[1]))))**2))/2 + params['beta']*np.nansum(kld))/N
    # JgradW2 = np.nan_to_num(((y - data) * y * (1 - y)) @ np.transpose(h)) / N + params['lambda'] * (We[1])
    # Jgradb2 = np.reshape(np.nansum(((y - data) * y * (1 - y)), 1) / N, We[3].shape)
    # JgradW1 = np.nan_to_num(((np.transpose(We[1]) @ ((y - data) * y * (1 - y))) * h * (1 - h)) @ np.transpose(data)) / N + params['lambda'] * (We[0]) + params['beta'] * np.nan_to_num(dkld * np.reshape((h * (1 - h)) @ np.transpose(data), We[0].shape)) / N
    # Jgradb1 = np.reshape(np.nansum(((np.transpose(We[1]) @ ((y - data) * y * (1 - y))) * h * (1 - h)), 1) / N, We[2].shape) + params['beta'] * np.reshape(np.nansum(dkld * (h * (1 - h)), 1) / N, (params['Lhid'], 1))
    return [J, JgradW2, Jgradb2, JgradW1, Jgradb1]

with h5py.File('/assign3_data1.h5', 'r') as f:
    data = np.asarray(f[list(f.keys())[0]])
    invXForm = np.asarray(f[list(f.keys())[1]])
    xForm = np.asarray(f[list(f.keys())[2]])

# Preprocessor
X = 0.2126 * data[:, 0, :, :] + 0.7152 * data[:, 1, :, :] + 0.0722 * data[:, 2, :, :]
for i in range(0, len(X)):  # for each image
    X[i] -= np.mean(X[i])  # normalize
X[np.where(X > 3 * np.std(X))] = 3 * np.std(X)  # clip
X[np.where(X < -3 * np.std(X))] = -3 * np.std(X)  # clip
for i in range(0, len(X)):  # for each image
    X[i] = (0.8 * (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))) + 0.1  # map to [0.1 0.9]

# # Display sample data
# for i in np.random.randint(0, data.shape[0]-1, 200):
#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(
#         np.transpose((data[i, :, :, :] - np.min(data[i, :, :, :])) / (
#                 np.max(data[i, :, :, :]) - np.min(data[i, :, :, :]))))
#     axs[0].set_title("RGB")
#     axs[1].imshow(np.transpose(X[i, :, :]))
#     axs[1].set_title("normalized")
#     fig.suptitle("image " + str(i) + " before and after preprocessing")
#     fig.savefig(str(i + 1) + ".png")

X = np.transpose(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))

# Parameters
Lin = X.shape[0]  # input layer
Lhid = 64  # hidden layer
Lout = Lin  # output layer
lamb = 5 * 10 ** -4
beta = 0.001  # tune by experiment
rho = 0.4  # tune by experiment
n = 0.005  # learning rate
epochs = 25
mini_batch = 128

# initialize weights
b1 = np.random.uniform(-(6 / (Lin + Lhid)) ** 0.5, (6 / (Lin + Lhid)) ** 0.5,
                        (Lhid, 1))  # bias for hidden layer
w1 = np.random.uniform(-(6 / (Lin + Lhid)) ** 0.5, (6 / (Lin + Lhid)) ** 0.5,
                        (Lhid, Lin))  # weights for hidden layer
b2 = np.random.uniform(-(6 / (Lout + Lhid)) ** 0.5, (6 / (Lout + Lhid)) ** 0.5,
                        (Lout, 1))  # bias for output layer
w2 = np.random.uniform(-(6 / (Lout + Lhid)) ** 0.5, (6 / (Lout + Lhid)) ** 0.5,
                        (Lout, Lhid))  # weights for output layer

J = np.zeros(epochs)
JJ = np.inf
for epoch in range(0, epochs):
    ir = np.random.randint(0, X.shape[1]-1, X.shape[1])
    for mb in range(0, np.ceil(X.shape[1]/mini_batch).astype(np.int64)):
        x = X[:, ir[mb*mini_batch:(mb*mini_batch)+mini_batch-1]]
        # Forward propagation
        z1 = b1 + w1 @ x
        h = np.nan_to_num(1 / (1 + np.exp(-z1)))
        z2 = b2 + w2 @ h
        y = np.nan_to_num(1 / (1 + np.exp(-z2)))

        # Backpropagation
        [J[epoch], JgradW2, Jgradb2, JgradW1, Jgradb1] = aeCost([w1, w2, b1, b2], x,
                                            {'Lin': Lin, 'Lhid': Lhid, 'lambda': lamb, 'beta': beta,'rho': rho},
                                                                h, y)
        b2 = b2 - n * np.nan_to_num(Jgradb2)
        w2 = w2 - n * np.nan_to_num(JgradW2)
        b1 = b1 - n * np.nan_to_num(Jgradb1)
        w1 = w1 - n * np.nan_to_num(JgradW1)

        if J[epoch] < JJ:
            bw1 = w1
            bw2 = w2
            bb1 = b1
            bb2 = b2
            JJ = J[epoch]
plt.plot(J)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
print("blah")
z1 = bb1 + bw1 @ X
h = np.nan_to_num(1 / (1 + np.exp(-z1)))
z2 = bb2 + bw2 @ h
y = np.nan_to_num(1 / (1 + np.exp(-z2)))
h = np.resize(np.transpose(h), (h.shape[1], np.int(np.sqrt(h.shape[0])), np.int(np.sqrt(h.shape[0]))))
y = np.resize(np.transpose(y), (y.shape[1], np.int(np.sqrt(y.shape[0])), np.int(np.sqrt(y.shape[0]))))
X = np.resize(np.transpose(X), (X.shape[1], np.int(np.sqrt(X.shape[0])), np.int(np.sqrt(X.shape[0]))))
# Display samples
for i in np.random.randint(0, data.shape[0] - 1, 10):
    fig, axs = plt.subplots(2, 2)
    axs[0][0].imshow(
        np.transpose((data[i, :, :, :] - np.min(data[i, :, :, :])) / (
                np.max(data[i, :, :, :]) - np.min(data[i, :, :, :]))))
    axs[0][0].set_title("RGB")
    axs[0][1].imshow(np.transpose(X[i, :, :]))
    axs[0][1].set_title("normalized")
    axs[1][0].imshow(np.transpose(y[i, :, :]))
    axs[1][0].set_title("output layer")
    axs[1][1].imshow(np.transpose(h[i, :, :]))
    axs[1][1].set_title("hidden layer")
    fig.suptitle("image " + str(i) + " before and after preprocessing")
    fig.savefig(str(i + 1) + ".png")
# Display weights
w1 = np.resize(w1, (w1.shape[0], np.int(np.sqrt(w1.shape[1])), np.int(np.sqrt(w1.shape[1]))))
fig, axs = plt.subplots(8, 8)
for a in range(0, 8):
    for b in range(0, 8):
        axs[a][b].imshow(np.transpose(w1[(a*8+b), :, :]))
        axs[a][b].set_title("neuron " + str(a*8+b+1))
fig.suptitle("weights to hidden layer without bias")
fig.set_size_inches(13, 15)
fig.savefig("weights to hidden layer without bias.png")