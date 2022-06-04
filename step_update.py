def step_update(x_pred, E_pred,  Z, R_seg, mapa, g):

    if Z.shape[1]==0:
        x_up = x_pred
        E_up = E_pred

    v, H, R_seg = matching(x_pred, E_pred, Z, R_seg, mapa, g)

    #mudar formato de v, H e R para usar nas equacoes
    y = np.reshape(v, (v.shape[0]*v.shape[1],1), 'F')

    Hreshape = np.zeros((H.shape[0] * H.shape[2], 3))
    cenoura = 0
    for batata in range(0, H.shape[2]):
        Hreshape[cenoura, :] = H[0, :, batata]
        Hreshape[cenoura + 1, :] = H[1, :, batata]
        cenoura = cenoura + 2

    R_seg1 = R_seg[:, :, 0]
    for bruh in range(1, R_seg.shape[2]):
        R_seg1 = scipy.linalg.block_diag(R_seg1, R_seg[:, :, bruh])


    S = Hreshape @ E_pred @ np.transpose(Hreshape) + R_seg1
    K = E_pred @ np.transpose(Hreshape) @ (np.linalg.inv(S))

    E_up = E_pred - K @ S @ np.transpose(K)
    x_up = x_pred + K @ y

    return x_up, E_up